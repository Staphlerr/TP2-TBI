import re
from bsbi import BSBIIndex
from compression import VBEPostings
import math
from compression import EliasGammaPostings

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def dcg(ranking):
    """
    Discounted Cumulative Gain
    Rumus: sum(rel_i / log2(i + 1)) untuk i dari 1 sampai rank
    """
    score = 0.0
    for i, rel in enumerate(ranking):
        score += rel / math.log2(i + 2) 
    return score

def ndcg(ranking):
    """
    Normalized Discounted Cumulative Gain
    Rumus: DCG / IDCG
    IDCG adalah DCG dari ranking yang ideal (diurutkan dari relevan ke tidak relevan)
    """
    actual_dcg = dcg(ranking)
    ideal_ranking = sorted(ranking, reverse=True)
    ideal_dcg = dcg(ideal_ranking)
    
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

def ap(ranking):
    """
    Average Precision
    Rumus: Rata-rata dari Precision@K tiap kali kita menemukan dokumen relevan
    """
    relevant_count = 0
    sum_precision = 0.0
    
    for i, rel in enumerate(ranking):
        if rel == 1:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            sum_precision += precision_at_i
            
    if relevant_count == 0:
        return 0.0
    return sum_precision / relevant_count

######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = EliasGammaPostings, \
                          output_dir = 'index')

  queries = []
  with open(query_file) as file:
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])
      queries.append((qid, query))

  models = [
      ("TF-IDF Baseline", BSBI_instance.retrieve_tfidf),
      ("BM25 Scoring", BSBI_instance.retrieve_bm25),
      ("WAND Top-K Retrieval", BSBI_instance.retrieve_wand)
  ]

  print("==========================================================")
  print("        EVALUASI SEARCH ENGINE BSBI (30 QUERIES)          ")
  print("==========================================================")

  for model_name, retrieve_func in models:
      rbp_scores, dcg_scores, ndcg_scores, ap_scores = [], [], [], []
      
      for qid, query in queries:
          ranking = []
          for (score, doc) in retrieve_func(query, k = k):
              did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
              ranking.append(qrels[qid][did])
              
          rbp_scores.append(rbp(ranking))
          dcg_scores.append(dcg(ranking))
          ndcg_scores.append(ndcg(ranking))
          ap_scores.append(ap(ranking))

      print(f"--- Hasil {model_name} ---")
      print(f"RBP score  = {sum(rbp_scores) / len(rbp_scores):.4f}")
      print(f"DCG score  = {sum(dcg_scores) / len(dcg_scores):.4f}")
      print(f"NDCG score = {sum(ndcg_scores) / len(ndcg_scores):.4f}")
      print(f"AP score   = {sum(ap_scores) / len(ap_scores):.4f}")
      print()

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  eval(qrels)