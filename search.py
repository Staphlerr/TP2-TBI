from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = EliasGammaPostings, \
                          output_dir = 'index')

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]

models = {
    "TF-IDF": BSBI_instance.retrieve_tfidf,
    "BM25": BSBI_instance.retrieve_bm25,
    "WAND": BSBI_instance.retrieve_wand
}
           
for query in queries:
    print("==========================================================")
    print(f"Query : '{query}'")
    print("==========================================================")
    
    for model_name, retrieve_func in models.items():
        print(f"\n[{model_name} Results - Top 10]")
        results = retrieve_func(query, k = 10) 
        if not results:
            print("Tidak ada dokumen yang ditemukan.")
            continue
            
        for (score, doc) in results:
            print(f"{doc:30} {score:>.3f}")
    print("\n")