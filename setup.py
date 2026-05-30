"""
evapoContext: Hardware-Aware Stateful Context Router
"""

from setuptools import setup, find_packages

setup(
    name="evapocontext",
    version="1.0.0",
    description="A hardware-aware stateful context router middleware implementing Model Context Protocol.",
    author="EvapoContext Engineering Team",
    author_email="support@evapocontext.local",
    url="https://github.com/evapocontext/evapocontext",
    packages=find_packages(),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.6",
        "psutil>=5.8.0",
        "onnxruntime>=1.14.0",
        "tokenizers>=0.13.0",
        "huggingface_hub>=0.14.0",
        "jsonschema>=3.2.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License"
    ],
    entry_points={
        "console_scripts": [
            "evapocontext-daemon=src.server:run_daemon"
        ]
    }
)
