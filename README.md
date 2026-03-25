# 🔍 Python Basic Search Engine: BSBI, BM25 & WAND Retrieval

This repository contains a from-scratch implementation of a basic search engine using the standard Python library. It was developed as a programming assignment 2 for the Information Retrieval (Temu Balik Informasi) course. The system is designed to process, index, and retrieve documents efficiently using Blocked Sort-Based Indexing (BSBI) along with advanced scoring and retrieval algorithms.

## ✨ Key Features Implemented

* **Blocked Sort-Based Indexing (BSBI):** A scalable indexing technique that constructs the inverted index in blocks, writes them to disk, and merges them, ensuring memory efficiency for large collections.
* **Bit-Level Compression (Elias-Gamma):** Implemented a custom bit-level compression algorithm (Elias-Gamma) to significantly reduce the disk space required for postings lists and term frequencies, outperforming the standard Variable-Byte Encoding (VBE).
* **BM25 Scoring Model:** Upgraded the baseline TF-IDF scoring to the Okapi BM25 ranking function. This provides more accurate document retrieval by incorporating term frequency saturation and document length normalization.
* **WAND (Weak AND) Top-K Retrieval:** Implemented a Document-at-a-Time (DaaT) processing strategy with the WAND algorithm. By utilizing pre-computed upper bounds for each term, the engine performs early termination, significantly accelerating query response times without sacrificing accuracy.
* **Comprehensive Evaluation Metrics:** Integrated and evaluated search effectiveness using industry-standard metrics, including:
  * Rank-Biased Precision (RBP)
  * Discounted Cumulative Gain (DCG)
  * Normalized Discounted Cumulative Gain (NDCG)
  * Average Precision (AP)

## 📁 Repository Structure

* `collection/`: Directory containing the raw text documents to be indexed.
* `index/`: Directory where the final compressed inverted index binaries and metadata are stored.
* `tmp/`: Temporary directory used during the BSBI indexing process before merging.
* `bsbi.py`: The core module handling the BSBI logic, BM25 scoring, and WAND retrieval.
* `index.py`: Handles the reading and writing (I/O) of the inverted index to disk.
* `compression.py`: Contains standard, VBE, and Elias-Gamma compression schemes.
* `util.py`: Utility functions, including ID mapping and postings merging.
* `search.py`: A script to perform interactive searches on the built index.
* `evaluation.py`: Evaluates the search engine's performance using `qrels.txt` and `queries.txt`.

## 🚀 How to Run

### 1. Indexing the Collection
Before searching, you must build the inverted index. Ensure your documents are placed inside the `collection/` directory, then run:
```bash
python bsbi.py
```
*This process will parse the documents, create intermediate indices in `tmp/`, merge them, and generate the final compressed index in the `index/` folder.*

### 2. Evaluating the Search Engine
To test the system's accuracy against the provided 30 queries (`queries.txt`) and view the evaluation scores (RBP, DCG, NDCG, AP), run:
```bash
python evaluation.py
```

### 3. Running Interactive Searches
You can modify the sample query list inside `search.py` and run the script to see the Top-K retrieved documents along with their BM25 scores:
```bash
python search.py
```

## 👤 Author
**Belva Ghani Abhinaya**, Faculty of Computer Science, Universitas Indonesia

***
