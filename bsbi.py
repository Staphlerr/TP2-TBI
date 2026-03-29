import os
import pickle
import contextlib
import heapq
import time
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import faiss

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                for token in f.read().split():
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.5, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time)
        menggunakan model scoring BM25.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi
        k: int
            Jumlah top dokumen yang dikembalikan
        k1: float
            Parameter k1 untuk BM25 (default 1.5)
        b: float
            Parameter b untuk BM25 (default 0.75)

        Result
        ------
        List[(float, str)]
            List of tuple: elemen pertama adalah score BM25, dan yang
            kedua adalah nama dokumen.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Filter term yang ada di query dan ada di collection (menghindari KeyError)
        terms = []
        for word in query.split():
            if word in self.term_id_map.str_to_id:
                terms.append(self.term_id_map[word])

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            scores = {}
            N = len(merged_index.doc_length)
            avgdl = merged_index.avg_doc_length

            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    # Menggunakan rumus IDF standar untuk BM25
                    idf = math.log(N / df) 
                    
                    postings, tf_list = merged_index.get_postings_list(term)
                    
                    for i in range(len(postings)):
                        doc_id = postings[i]
                        tf = tf_list[i]
                        doc_len = merged_index.doc_length[doc_id]
                        
                        if doc_id not in scores:
                            scores[doc_id] = 0.0
                            
                        # Rumus BM25
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                        scores[doc_id] += idf * (numerator / denominator)

            # Sortir Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]
    
    def retrieve_wand(self, query, k=10, k1=1.5, b=0.75):
        """
        Melakukan Ranked Retrieval dengan algoritma WAND (Weak AND)
        menggunakan model scoring BM25.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split() if word in self.term_id_map.str_to_id]
        if not terms:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avgdl = merged_index.avg_doc_length

            pointers = {} 
            postings = {}
            tf_lists = {}
            upper_bounds = {}
            
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    max_tf = merged_index.postings_dict[term][4] 
                    idf = math.log(N / df)
                    
                    ub_score = idf * ((max_tf * (k1 + 1)) / (max_tf + k1 * (1 - b)))
                    
                    p_list, t_list = merged_index.get_postings_list(term)
                    
                    pointers[term] = 0
                    postings[term] = p_list
                    tf_lists[term] = t_list
                    upper_bounds[term] = ub_score

            active_terms = list(postings.keys())
            if not active_terms:
                return []

            heap = [] 
            threshold = 0.0

            while True:
                active_terms = [t for t in active_terms if pointers[t] < len(postings[t])]
                if not active_terms:
                    break
    
                active_terms.sort(key=lambda t: postings[t][pointers[t]])
                sum_ub = 0.0
                pivot_term = None
                
                for t in active_terms:
                    sum_ub += upper_bounds[t]
                    if sum_ub > threshold:
                        pivot_term = t
                        break
                
                if pivot_term is None:
                    break
                    
                pivot_doc = postings[pivot_term][pointers[pivot_term]]
                first_term = active_terms[0]
                first_doc = postings[first_term][pointers[first_term]]
                
                if first_doc == pivot_doc:
                    score = 0.0
                    doc_len = merged_index.doc_length[pivot_doc]
                    
                    for t in active_terms:
                        if postings[t][pointers[t]] == pivot_doc:
                            tf = tf_lists[t][pointers[t]]
                            df = merged_index.postings_dict[t][1]
                            idf = math.log(N / df)
                            numerator = tf * (k1 + 1)
                            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                            score += idf * (numerator / denominator)
                            
                            pointers[t] += 1 
                            
                    if len(heap) < k:
                        heapq.heappush(heap, (score, pivot_doc))
                        if len(heap) == k:
                            threshold = heap[0][0] 
                    elif score > threshold:
                        heapq.heappushpop(heap, (score, pivot_doc))
                        threshold = heap[0][0]
                        
                else:
                    while pointers[first_term] < len(postings[first_term]) and postings[first_term][pointers[first_term]] < pivot_doc:
                        pointers[first_term] += 1

            docs = [(score, self.doc_id_map[doc_id]) for score, doc_id in sorted(heap, reverse=True)]
            return docs
    
    def build_lsi(self, k_dim=100):
        """
        Membangun model Latent Semantic Indexing (LSI) dan meng-index 
        vektor dokumen menggunakan FAISS.
        """
        
        row_indices = [] 
        col_indices = []
        data_values = [] 
        
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            for term_id in tqdm(merged_index.postings_dict.keys()):
                df = merged_index.postings_dict[term_id][1]
                idf = math.log(N / df)
                
                postings, tf_list = merged_index.get_postings_list(term_id)
                for i in range(len(postings)):
                    doc_id = postings[i]
                    tf = tf_list[i]
                    weight = (1 + math.log(tf)) * idf
                    
                    row_indices.append(doc_id)
                    col_indices.append(term_id)
                    data_values.append(weight)
        
        num_docs = max(row_indices) + 1
        num_terms = max(col_indices) + 1
        sparse_matrix = csr_matrix((data_values, (row_indices, col_indices)), 
                                   shape=(num_docs, num_terms), dtype=np.float32)
        
        k_dim = min(k_dim, min(num_docs, num_terms) - 1) 
        U, Sigma, VT = svds(sparse_matrix, k=k_dim)
        doc_vectors = np.dot(U, np.diag(Sigma)).astype('float32')
        
        faiss.normalize_L2(doc_vectors) 
        self.faiss_index = faiss.IndexFlatIP(k_dim)
        self.faiss_index.add(doc_vectors)
        
        self.VT = VT.astype('float32')
        self.Sigma_inv = np.diag(1.0 / Sigma).astype('float32')
        
        faiss.write_index(self.faiss_index, os.path.join(self.output_dir, 'lsi.index'))
        with open(os.path.join(self.output_dir, 'svd_components.pkl'), 'wb') as f:
            pickle.dump((self.VT, self.Sigma_inv, num_terms, N), f)
    
    def retrieve_lsi(self, query, k=10):
        """
        Melakukan pencarian semantik menggunakan LSI dan FAISS.
        """
        if not hasattr(self, 'faiss_index'):
            self.faiss_index = faiss.read_index(os.path.join(self.output_dir, 'lsi.index'))
            with open(os.path.join(self.output_dir, 'svd_components.pkl'), 'rb') as f:
                self.VT, self.Sigma_inv, self.num_terms, self.N_docs = pickle.load(f)
            self.load() 

        query_vector = np.zeros((1, self.num_terms), dtype=np.float32)
        
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            for word in query.split():
                if word in self.term_id_map.str_to_id:
                    term_id = self.term_id_map[word]
                    df = merged_index.postings_dict[term_id][1]
                    idf = math.log(self.N_docs / df)
                    query_vector[0, term_id] = idf 
                    
        query_lsi = np.dot(np.dot(query_vector, self.VT.T), self.Sigma_inv)
        if np.all(query_vector == 0):
            return []

        faiss.normalize_L2(query_lsi)
        scores, doc_ids = self.faiss_index.search(query_lsi, k)
        results = []
        for i in range(len(doc_ids[0])):
            did = int(doc_ids[0][i])
            score = float(scores[0][i])
            
            if did != -1 and did < len(self.doc_id_map.id_to_str): 
                results.append((score, self.doc_id_map[did]))
                
        return results

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)
                self.build_lsi(k_dim=100)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = EliasGammaPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
