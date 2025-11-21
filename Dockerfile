FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# IMPORTANT: pin versions that match mordecai3's expectations
RUN pip install --no-cache-dir \
      "spacy==3.5.0" \
      "spacy-transformers==1.1.8" \
      "transformers==4.26.1" \
      "textacy==0.13.3" \
      "mordecai3"

# Download the transformer-based English model
RUN python -m spacy download en_core_web_trf

