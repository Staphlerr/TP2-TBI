# Python Basic Search Engine: BSBI, BM25, WAND & LSI-FAISS Semantic Retrieval

This repository contains a from-scratch implementation of a basic search engine using the standard Python library. It was developed as Programming Assignment 2 (TP2) for the Information Retrieval (Temu Balik Informasi) course. The system processes, indexes, and retrieves documents efficiently using Blocked Sort-Based Indexing (BSBI) along with advanced scoring, retrieval algorithms, and modern vector-based semantic search.

## Key Features Implemented

* **Blocked Sort-Based Indexing (BSBI):** A scalable indexing technique that constructs the inverted index in blocks, writes them to disk, and merges them, ensuring memory efficiency for large document collections.
* **Bit-Level Compression (Elias-Gamma):** Implemented a custom bit-level compression algorithm (Elias-Gamma) to significantly reduce the disk space required for postings lists and term frequencies, outperforming the standard Variable-Byte Encoding (VBE). *Includes a +1 shift mechanism to handle zero-values mathematically.*
* **BM25 Scoring Model:** Upgraded the baseline TF-IDF scoring to the Okapi BM25 ranking function. This provides more accurate document retrieval by incorporating term frequency saturation and document length normalization (using pre-computed average document lengths).
* **WAND (Weak AND) Top-K Retrieval:** Implemented a Document-at-a-Time (DaaT) processing strategy with the WAND algorithm. By utilizing pre-computed upper bounds (Max TF) for each term, the engine performs early termination, significantly accelerating query response times while yielding the exact same accuracy as standard BM25.
* **Latent Semantic Indexing (LSI) & FAISS Vector Search:** Implemented a semantic search engine pipeline to capture context and synonyms beyond exact keyword matches. 
  * Constructs a memory-efficient sparse Term-Document matrix using `scipy.sparse`.
  * Applies **Truncated SVD** to project documents and queries into a lower-dimensional latent semantic space.
  * Integrates Facebook's **FAISS** (`IndexFlatIP`) for lightning-fast vector indexing and Cosine Similarity retrieval.
* **Automated Multi-Model Evaluation:** Integrated and evaluated search effectiveness using industry-standard metrics across TF-IDF, BM25, WAND, and LSI-FAISS concurrently:
  * Rank-Biased Precision (RBP)
  * Discounted Cumulative Gain (DCG)
  * Normalized Discounted Cumulative Gain (NDCG)
  * Average Precision (AP)

## Repository Structure

* `collection/`: Directory containing the raw text documents to be indexed.
* `index/`: Directory where the final compressed inverted index binaries, FAISS indices, and metadata are stored (tracked via `.gitkeep`).
* `tmp/`: Temporary directory used during the BSBI indexing process before merging (tracked via `.gitkeep`).
* `bsbi.py`: The core module handling the BSBI logic, BM25 scoring, WAND retrieval, and LSI matrix construction.
* `index.py`: Handles the reading and writing (I/O) of the inverted index to disk, storing term upper bounds and document lengths.
* `compression.py`: Contains Elias-Gamma compression schemes (alongside standard and VBE).
* `util.py`: Utility functions, including ID mapping and postings merging.
* `trie.py`: Implementation of the Trie data structure.
* `search.py`: A script to perform interactive Top-K searches comparing all implemented retrieval methods simultaneously.
* `evaluation.py`: Evaluates the search engine's performance dynamically using `qrels.txt` and `queries.txt`.

## Dependencies

To run the semantic search (LSI-FAISS) evaluation and indexing, you need to install the following external libraries:
```bash
pip install faiss-cpu scipy numpy tqdm
```

## How to Change Compression Methods

By default, the engine is set to use the custom `EliasGammaPostings`. If you wish to test the baseline `StandardPostings` or `VBEPostings`, you need to modify the `postings_encoding` parameter.

1. Open `bsbi.py`, `search.py`, and `evaluation.py`.
2. Change the import statement at the top:
   ```python
   from compression import StandardPostings  # or VBEPostings
   ```
3. Change the `postings_encoding` argument when instantiating `BSBIIndex`:
   ```python
   BSBI_instance = BSBIIndex(data_dir = 'collection', \
                             postings_encoding = StandardPostings, \
                             output_dir = 'index')
   ```
4. **Important:** Whenever you change the compression method, you **must** rebuild the index by running `python bsbi.py` before running searches or evaluations.

## How to Run

### 1. Indexing the Collection
Before searching, you must build the inverted index and the FAISS vector index. Ensure your documents are placed inside the `collection/` directory, then run:
```bash
python bsbi.py
```
*This process will parse the documents, create intermediate indices in `tmp/`, merge them, and then run SVD to generate the FAISS semantic index in the `index/` folder.*

### 2. Evaluating the Search Engine
To test the system's accuracy against the provided 30 queries (`queries.txt`) and view the comparative evaluation scores (RBP, DCG, NDCG, AP) for TF-IDF, BM25, WAND, and LSI-FAISS, run:
```bash
python evaluation.py
```

### 3. Running Interactive Searches
You can modify the sample query list inside `search.py` and run the script to see the Top-10 retrieved documents along with their scores across all retrieval methods:
```bash
python search.py
```

## Author
**Belva Ghani Abhinaya** (2306203526)  
Faculty of Computer Science, Universitas Indonesia