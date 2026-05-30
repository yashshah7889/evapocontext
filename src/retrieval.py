"""
evapoContext: Hardware-Aware Stateful Context Router
"""

import os
import re
import math
import json
import logging
import sqlite3
import threading
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

# Configure logging
logger = logging.getLogger("EvapoContextRetrieval")

DEFAULT_CATEGORY_WEIGHTS = {
    "system_rule": 1.5,
    "tool_schema": 1.2,
    "memory": 1.1,
    "conversation": 1.0
}

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each", 
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", 
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", 
    "its", "itself", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", 
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", 
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", 
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", 
    "through", "to", "too", "under", "until", "up", "very", "was", "were", "what", "when", 
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours", 
    "yourself", "yourselves"
}


class TextSplitter:
    """
    Utility class for chunking large text documents into overlapping segments.
    """

    @staticmethod
    def split_text(text: str, chunk_size: int = 150, chunk_overlap: int = 25) -> List[str]:
        """
        Splits a string document into a list of word-level chunks with overlaps.
        """
        words = text.strip().split()
        if not words:
            return []
            
        chunks = []
        i = 0
        while i < len(words):
            segment_words = words[i : i + chunk_size]
            chunks.append(" ".join(segment_words))
            
            i += (chunk_size - chunk_overlap)
            if i >= len(words) or chunk_size <= chunk_overlap:
                break
                
        return chunks


class EmbeddingGenerator:
    """
    Handles local tokenization and ONNX inference to generate dense vectors
    using the bge-small-en-v1.5 model. Configured with CPU thread tuning.
    """

    def __init__(
        self,
        model_dir: str = "src/model",
        repo_id: str = "BAAI/bge-small-en-v1.5",
        use_gpu: bool = False
    ):
        """
        Initializes the embedding generator. Downloads files from Hugging Face if missing.
        """
        self.model_dir = os.path.abspath(model_dir)
        self.repo_id = repo_id
        self.use_gpu = use_gpu

        self.model_path = os.path.join(self.model_dir, "onnx", "model.onnx")
        self.tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")

        self._lock = threading.Lock()
        self.session = None
        self.tokenizer = None

        logger.info(f"Initializing EmbeddingGenerator (Model Dir: {self.model_dir})")
        self._ensure_model_files()
        self._load_model()

    def _ensure_model_files(self):
        """
        Checks for local model files and downloads them if they do not exist.
        """
        with self._lock:
            # Check model file
            if not os.path.exists(self.model_path):
                logger.info(f"Model file missing. Downloading from repo: {self.repo_id}...")
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename="onnx/model.onnx",
                    local_dir=self.model_dir
                )

            # Check tokenizer file
            if not os.path.exists(self.tokenizer_path):
                logger.info(f"Tokenizer configuration missing. Downloading from repo: {self.repo_id}...")
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename="tokenizer.json",
                    local_dir=self.model_dir
                )

    def _load_model(self):
        """
        Loads the tokenizer and compiles the ONNX session with hardware optimizations.
        """
        with self._lock:
            logger.info("Loading tokenizer configuration...")
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
            
            self.tokenizer.enable_padding(direction="right", pad_id=0, pad_token="[PAD]")
            self.tokenizer.enable_truncation(max_length=512)

            logger.info("Configuring ONNX session options...")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            logger.info("Initializing ONNX inference session...")
            providers = ["CPUExecutionProvider"]
            if self.use_gpu:
                available_providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in available_providers:
                    providers.insert(0, "CUDAExecutionProvider")
                elif "CoreMLExecutionProvider" in available_providers:
                    providers.insert(0, "CoreMLExecutionProvider")

            self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=providers)
            logger.info(f"ONNX session loaded successfully with providers: {self.session.get_providers()}")

    def generate_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Generates dense 384-dimensional mathematical arrays for input texts.
        Processed in small batches of 16 to avoid ONNX memory allocation failures.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        processed_texts = []
        for text in texts:
            if is_query:
                prefix = "Represent this sentence for searching relevant passages: "
                if not text.startswith(prefix):
                    processed_texts.append(f"{prefix}{text}")
                else:
                    processed_texts.append(text)
            else:
                processed_texts.append(text)

        batch_size = 16
        all_embeddings = []
        
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i : i + batch_size]
            with self._lock:
                encodings = self.tokenizer.encode_batch(batch)
                
                input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
                attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)
                token_type_ids = np.array([enc.type_ids for enc in encodings], dtype=np.int64)

                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                }
                outputs = self.session.run(None, inputs)
                token_embeddings = outputs[0]

                input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
                sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
                sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), 1e-9, None)
                embeddings = sum_embeddings / sum_mask

                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                normalized_embeddings = embeddings / np.maximum(norms, 1e-12)
                all_embeddings.append(normalized_embeddings)

        return np.vstack(all_embeddings)


class BM25Index:
    """
    A pure-Python implementation of the Best Matching 25 (BM25) ranking algorithm
    with Stopword Filtering for clean term weight assignments.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.lock = threading.Lock()
        
        self.doc_count = 0
        self.doc_lengths: List[int] = []
        self.avg_doc_length = 0.0
        
        self.df: Dict[str, int] = {}
        self.tf_list: List[Dict[str, int]] = []
        self.docs: List[Dict[str, Any]] = []

    def tokenize(self, text: str) -> List[str]:
        """
        Splits text into lower-cased alphanumeric word tokens, filtering out English stopwords.
        """
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if t not in STOPWORDS]

    def add_document(self, doc_id: Any, text: str):
        """
        Adds a single document to the BM25 index.
        """
        with self.lock:
            tokens = self.tokenize(text)
            self.docs.append({"id": doc_id, "text": text, "tokens": tokens})
            self.doc_count += 1
            
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.tf_list.append(tf)
            
            for token in tf:
                self.df[token] = self.df.get(token, 0) + 1
                
            self.doc_lengths.append(len(tokens))
            self.avg_doc_length = sum(self.doc_lengths) / self.doc_count

    def clear(self):
        """
        Clears the BM25 index.
        """
        with self.lock:
            self.doc_count = 0
            self.doc_lengths.clear()
            self.avg_doc_length = 0.0
            self.df.clear()
            self.tf_list.clear()
            self.docs.clear()

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Computes BM25 relevance scores for all documents and returns the top_k.
        """
        with self.lock:
            if self.doc_count == 0:
                return []

            query_tokens = self.tokenize(query)
            if not query_tokens:
                return [(i, 0.0) for i in range(self.doc_count)][:top_k]

            scores = []
            for i in range(self.doc_count):
                score = 0.0
                tf = self.tf_list[i]
                doc_len = self.doc_lengths[i]
                
                for token in query_tokens:
                    if token not in self.df:
                        continue
                    
                    df_token = self.df[token]
                    idf = math.log(1.0 + (self.doc_count - df_token + 0.5) / (df_token + 0.5))
                    f_q = tf.get(token, 0)
                    
                    numerator = f_q * (self.k1 + 1.0)
                    denominator = f_q + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_length))
                    score += idf * (numerator / denominator)
                    
                scores.append((i, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]


class HybridRetrievalStore:
    """
    A thread-safe hybrid database store that integrates BM25 sparse search and
    dense vector cosine similarity re-ranking.
    
    Supports SQLite persistence, Category-based Metadata Boosting, 
    Deduplication vector reuse, and dual-layer Query Caching.
    """

    def __init__(self, embedding_generator: EmbeddingGenerator, db_path: Optional[str] = None):
        self.embedding_generator = embedding_generator
        self.bm25_index = BM25Index()
        self.lock = threading.Lock()
        self.db_path = db_path
        
        # Internal chunk storage
        self.chunks: List[Dict[str, Any]] = []

        # Dual-Layer Query Caching
        # 1. Exact string cache maps raw query strings -> retrieved outputs
        self.exact_query_cache: Dict[str, List[Dict[str, Any]]] = {}
        # 2. Semantic cache stores list of dicts: {'query': str, 'embedding': np.ndarray, 'results': list}
        self.semantic_query_cache: List[Dict[str, Any]] = []
        self.max_cache_size = 50

        if self.db_path:
            self._init_sqlite()
            self._load_from_sqlite()

    def _init_sqlite(self):
        """
        Creates the database table if it doesn't already exist.
        """
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    is_pinned INTEGER NOT NULL,
                    pinning_level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _load_from_sqlite(self):
        """
        Loads cached chunks and their vector embeddings from the SQLite store.
        """
        logger.info(f"Loading cached contexts from SQL store: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, text, token_count, is_pinned, pinning_level, category, embedding, metadata FROM chunks")
            rows = cursor.fetchall()
            
            loaded_count = 0
            for row in rows:
                chunk_id, text, token_count, is_pinned, pinning_level, category, emb_blob, meta_str = row
                
                embedding = np.frombuffer(emb_blob, dtype=np.float32)
                metadata = json.loads(meta_str)

                chunk_record = {
                    "id": chunk_id,
                    "text": text,
                    "token_count": token_count,
                    "is_pinned": bool(is_pinned),
                    "pinning_level": pinning_level,
                    "category": category,
                    "embedding": embedding,
                    "metadata": metadata
                }

                self.chunks.append(chunk_record)
                self.bm25_index.add_document(doc_id=chunk_id, text=text)
                loaded_count += 1
                
            logger.info(f"Loaded {loaded_count} cached chunks from database.")
        finally:
            conn.close()

    def _invalidate_caches(self):
        """
        Invalidates query caches when database contents change to ensure accurate results.
        """
        self.exact_query_cache.clear()
        self.semantic_query_cache.clear()

    def add_chunks(self, raw_chunks: List[Dict[str, Any]]):
        """
        Inserts chunks into the index. Deduplicates insertions by checking if the chunk
        already exists with identical text, reusing cached embeddings to skip ONNX execution.
        """
        if not raw_chunks:
            return

        # Invalidate query caches since corpus changed
        self._invalidate_caches()

        texts_to_embed = []
        indices_to_embed = []
        chunks_metadata = []

        existing_by_text = {}
        existing_by_id = {}
        with self.lock:
            for item in self.chunks:
                existing_by_text[item["text"]] = item["embedding"]
                existing_by_id[item["id"]] = item

        for idx, chunk in enumerate(raw_chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue
            
            chunk_id = chunk.get("id", f"chunk_{len(self.chunks) + len(chunks_metadata)}")
            is_pinned = chunk.get("is_pinned", False)
            pinning_level = chunk.get("pinning_level", "critical" if is_pinned else "none")
            category = chunk.get("category", chunk.get("metadata", {}).get("category", "conversation"))
            metadata = chunk.get("metadata", {})
            if "category" not in metadata:
                metadata = metadata.copy()
                metadata["category"] = category

            # Compute exact token count using the tokenizer
            encoded = self.embedding_generator.tokenizer.encode(text)
            token_count = len(encoded.ids)

            meta = {
                "id": chunk_id,
                "text": text,
                "token_count": token_count,
                "is_pinned": is_pinned,
                "pinning_level": pinning_level,
                "category": category,
                "metadata": metadata,
                "embedding": None
            }

            # Optimization Check: If text is already embedded, reuse the vector
            if text in existing_by_text:
                meta["embedding"] = existing_by_text[text]
            elif chunk_id in existing_by_id and existing_by_id[chunk_id]["text"] == text:
                meta["embedding"] = existing_by_id[chunk_id]["embedding"]
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(len(chunks_metadata))

            chunks_metadata.append(meta)

        # Generate embeddings in batch for entirely new chunks
        if texts_to_embed:
            new_embeddings = self.embedding_generator.generate_embeddings(texts_to_embed, is_query=False)
            for new_idx, chunk_idx in enumerate(indices_to_embed):
                chunks_metadata[chunk_idx]["embedding"] = new_embeddings[new_idx]

        # Write to SQLite database
        if self.db_path:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                for meta in chunks_metadata:
                    cursor.execute("""
                        INSERT OR REPLACE INTO chunks 
                        (id, text, token_count, is_pinned, pinning_level, category, embedding, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        meta["id"],
                        meta["text"],
                        meta["token_count"],
                        1 if meta["is_pinned"] else 0,
                        meta["pinning_level"],
                        meta["category"],
                        meta["embedding"].tobytes(),
                        json.dumps(meta["metadata"])
                    ))
                conn.commit()
            finally:
                conn.close()

        # Update in-memory collections
        with self.lock:
            for meta in chunks_metadata:
                self.chunks = [c for c in self.chunks if c["id"] != meta["id"]]
                self.chunks.append(meta)
                self.bm25_index.add_document(doc_id=meta["id"], text=meta["text"])

        logger.info(f"Indexed {len(chunks_metadata)} chunks. Model inference was bypassed for {len(chunks_metadata) - len(texts_to_embed)} chunks.")

    def clear(self):
        """
        Flushes all chunks and resets index structures.
        """
        self._invalidate_caches()
        if self.db_path:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chunks")
                conn.commit()
            finally:
                conn.close()

        with self.lock:
            self.bm25_index.clear()
            self.chunks.clear()
        logger.info("Cleared HybridRetrievalStore database.")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        bm25_candidates: int = 50,
        hybrid_weight: float = 0.3,
        category_weights: Optional[Dict[str, float]] = None,
        system_pressure: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid two-tiered search with category boosting and dual-layer caching.
        Integrates Adaptive Hardware-Aware Retrieval Gating (AHARG) to scale and bypass
        computation under resource strain.
        
        1. Exact Cache Check (lexical check, 0.001ms)
        2. Vector Bypass Check (if system_pressure >= 0.90, bypass ONNX execution)
        3. Semantic Cache Check (vector comparison, 0.05ms)
        4. Dynamic Candidate Scaling & Standard Two-Tiered Recall (BM25 + Cosine Re-ranking)
        """
        weights = category_weights if category_weights is not None else DEFAULT_CATEGORY_WEIGHTS
        weights_tuple = tuple(sorted(weights.items()))
        query_cleaned = query.strip()
        
        # Round system pressure to 1 decimal place to prevent caching wiggles
        pressure_bucket = round(system_pressure, 1)
        cache_key = (query_cleaned, top_k, bm25_candidates, hybrid_weight, weights_tuple, pressure_bucket)

        # --- OPTIMIZATION 1: EXACT CACHE CHECK ---
        with self.lock:
            if cache_key in self.exact_query_cache:
                # Return deep copy of cached records to prevent mutation
                return [c.copy() for c in self.exact_query_cache[cache_key][:top_k]]

        # --- HARDWARE-AWARE OPTIMIZATION 2: VECTOR INFERENCE BYPASS ---
        if system_pressure >= 0.90:
            logger.warning(
                f"Critical System Pressure ({system_pressure * 100:.1f}%) detected! "
                "Bypassing ONNX vector search and falling back to pure BM25 search to save CPU cycles."
            )
            with self.lock:
                total_docs = len(self.chunks)
                if total_docs == 0:
                    return []
                    
            bm25_results = self.bm25_index.search(query_cleaned, top_k=top_k)
            if not bm25_results:
                return []
                
            candidate_records = []
            max_bm25_score = max(score for _, score in bm25_results) if bm25_results else 0.0
            
            with self.lock:
                chunk_by_id = {c["id"]: c for c in self.chunks}
                for doc_idx, score in bm25_results:
                    target_doc = self.bm25_index.docs[doc_idx]
                    chunk_id = target_doc["id"]
                    
                    if chunk_id in chunk_by_id:
                        chunk = chunk_by_id[chunk_id].copy()
                        chunk["bm25_score"] = round(score, 6)
                        chunk["cosine_similarity"] = 0.0
                        
                        norm_bm25 = (score / max_bm25_score) if max_bm25_score > 0.0 else 0.0
                        category = chunk.get("category", "conversation")
                        boost = weights.get(category, 1.0)
                        boosted_score = norm_bm25 * boost
                        
                        chunk["similarity"] = round(boosted_score, 6)
                        chunk["vector_bypass_active"] = True
                        chunk.pop("embedding", None)
                        candidate_records.append(chunk)
                        
            candidate_records.sort(key=lambda x: x["similarity"], reverse=True)
            for rank_idx, chunk in enumerate(candidate_records):
                chunk["retrieval_rank"] = rank_idx + 1
                
            with self.lock:
                self.exact_query_cache[cache_key] = candidate_records
                
            return candidate_records[:top_k]

        # Step 3: Encode user query (needed for both semantic cache and vector search)
        query_embedding = self.embedding_generator.generate_embeddings([query_cleaned], is_query=True)[0]

        # --- OPTIMIZATION 3: SEMANTIC CACHE CHECK ---
        with self.lock:
            best_semantic_match = None
            best_semantic_sim = 0.0
            
            for cached in self.semantic_query_cache:
                if (
                    cached["top_k"] == top_k
                    and cached["bm25_candidates"] == bm25_candidates
                    and abs(cached["hybrid_weight"] - hybrid_weight) < 1e-6
                    and cached["weights_tuple"] == weights_tuple
                    and cached.get("pressure_bucket", 0.0) == pressure_bucket
                ):
                    sim = float(np.dot(cached["embedding"], query_embedding))
                    if sim > best_semantic_sim:
                        best_semantic_sim = sim
                        best_semantic_match = cached
                    
            # If queries match semantically at 96%+, return cached list
            if best_semantic_sim >= 0.96 and best_semantic_match is not None:
                # Save to exact cache for future lexical hits
                self.exact_query_cache[cache_key] = best_semantic_match["results"]
                return [c.copy() for c in best_semantic_match["results"][:top_k]]

        # --- OPTIMIZATION 4: STANDARD TWO-TIERED SEARCH WITH CANDIDATE SCALING ---
        # Dynamic Candidate Scaling based on pressure: scales candidates smoothly
        effective_candidates = max(5, int(bm25_candidates * (1.0 - system_pressure)))
        
        with self.lock:
            total_docs = len(self.chunks)
            if total_docs == 0:
                return []

        candidates_limit = min(effective_candidates, total_docs)
        bm25_results = self.bm25_index.search(query_cleaned, top_k=candidates_limit)
        
        if not bm25_results:
            return []

        candidate_records = []
        candidate_embeddings = []
        max_bm25_score = 0.0

        with self.lock:
            chunk_by_id = {c["id"]: c for c in self.chunks}
            for doc_idx, score in bm25_results:
                target_doc = self.bm25_index.docs[doc_idx]
                chunk_id = target_doc["id"]
                
                if chunk_id in chunk_by_id:
                    chunk = chunk_by_id[chunk_id].copy()
                    chunk["bm25_score"] = score
                    candidate_records.append(chunk)
                    candidate_embeddings.append(chunk["embedding"])
                    if score > max_bm25_score:
                        max_bm25_score = score

        if not candidate_records:
            return []

        # Dense vector similarity calculation
        candidate_matrix = np.array(candidate_embeddings, dtype=np.float32)
        cosine_similarities = np.dot(candidate_matrix, query_embedding)

        # Combined hybrid score and category boosting
        for idx, chunk in enumerate(candidate_records):
            bm25_val = chunk["bm25_score"]
            norm_bm25 = (bm25_val / max_bm25_score) if max_bm25_score > 0.0 else 0.0
            
            cosine_val = float(cosine_similarities[idx])
            clamped_cosine = max(0.0, min(1.0, cosine_val))
            
            base_score = (hybrid_weight * norm_bm25) + ((1.0 - hybrid_weight) * clamped_cosine)
            
            category = chunk.get("category", "conversation")
            boost = weights.get(category, 1.0)
            boosted_score = base_score * boost
            
            chunk["bm25_score"] = round(bm25_val, 6)
            chunk["cosine_similarity"] = round(clamped_cosine, 6)
            chunk["similarity"] = round(boosted_score, 6)
            chunk.pop("embedding", None)

        candidate_records.sort(key=lambda x: x["similarity"], reverse=True)

        for rank_idx, chunk in enumerate(candidate_records):
            chunk["retrieval_rank"] = rank_idx + 1

        # Save to Caches
        with self.lock:
            # 1. Update Lexical Cache
            self.exact_query_cache[cache_key] = candidate_records
            
            # 2. Update Semantic Cache (evict oldest if full)
            if len(self.semantic_query_cache) >= self.max_cache_size:
                self.semantic_query_cache.pop(0)
            self.semantic_query_cache.append({
                "query": query_cleaned,
                "embedding": query_embedding,
                "top_k": top_k,
                "bm25_candidates": bm25_candidates,
                "hybrid_weight": hybrid_weight,
                "weights_tuple": weights_tuple,
                "pressure_bucket": pressure_bucket,
                "results": candidate_records
            })

        return candidate_records[:top_k]
