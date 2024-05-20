# setup.py
#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="src",
    version="0.0.1",
    description="Cross-Encoder for reranking passage",
    author="",
    author_email="",
    url="https://github.com/kien-vu-uet/VNR-Chatbot",
    install_requires=["lightning==1.9"],
    packages=find_packages('./src'),
)