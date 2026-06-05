import os
import time
from Bio import Entrez
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

Entrez.email = "helenice06@gmail.com"

GOLD_DIR    = "/projects/F202600026AIVLABDEUCALION/helenice/reranking/data/sa-eval/smafirac"
PMIDS_SEEDS = ["16850029", "19735549", "21494637", "24204323"]

def fetch_abstract(pmid):
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        text = handle.read()
        handle.close()
        time.sleep(0.4)
        return text
    except Exception as e:
        print(f"  Erro ao buscar {pmid}: {e}")
        return ""

def load_gold(pmid):
    filepath = os.path.join(GOLD_DIR, f"eval_gold_{pmid}.txt")
    gold = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                gold[parts[0]] = int(parts[1])
    return gold

def precision_at_k(ranked, gold, k=20):
    top_k = ranked[:k]
    relevant = sum(1 for pmid in top_k if gold.get(pmid, 0) == 1)
    return relevant / k

def recall_at_k(ranked, gold, k=20):
    top_k = ranked[:k]
    total_relevant = sum(gold.values())
    if total_relevant == 0:
        return 0
    relevant = sum(1 for pmid in top_k if gold.get(pmid, 0) == 1)
    return relevant / total_relevant

def r_precision(ranked, gold):
    total_relevant = sum(gold.values())
    if total_relevant == 0:
        return 0
    top_r = ranked[:total_relevant]
    relevant = sum(1 for pmid in top_r if gold.get(pmid, 0) == 1)
    return relevant / total_relevant

def ndcg_at_k(ranked, gold, k=20):
    top_k = ranked[:k]
    dcg = sum(gold.get(pmid, 0) / np.log2(i + 2) for i, pmid in enumerate(top_k))
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0

print("=" * 60)
print("TF-IDF SMAFIRA-c — Abstract Completo vs Candidatos")
print("=" * 60)

results = []

for seed_pmid in PMIDS_SEEDS:
    print(f"\nSeed: {seed_pmid}")

    # 1. Buscar abstract completo do seed
    print(f"  A buscar abstract do seed...")
    seed_text = fetch_abstract(seed_pmid)
    print(f"  Abstract: {len(seed_text.split())} palavras")

    # 2. Carregar ground truth
    gold = load_gold(seed_pmid)
    candidate_pmids = list(gold.keys())
    print(f"  Candidatos: {len(candidate_pmids)} | Relevantes: {sum(gold.values())}")

    # 3. Buscar abstracts dos candidatos
    print(f"  A baixar abstracts dos candidatos...")
    candidate_texts = []
    for pmid in candidate_pmids:
        abstract = fetch_abstract(pmid)
        candidate_texts.append(abstract)

    # 4. TF-IDF
    all_texts = [seed_text] + candidate_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    seed_vec = tfidf_matrix[0]
    candidate_vecs = tfidf_matrix[1:]

    similarities = cosine_similarity(seed_vec, candidate_vecs)[0]

    # 5. Ranking
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_pmids = [candidate_pmids[i] for i in ranked_indices]

    # 6. Métricas
    p20  = precision_at_k(ranked_pmids, gold, k=20)
    r20  = recall_at_k(ranked_pmids, gold, k=20)
    rp   = r_precision(ranked_pmids, gold)
    ndcg = ndcg_at_k(ranked_pmids, gold, k=20)

    print(f"  Precision@20: {p20:.4f}")
    print(f"  Recall@20:    {r20:.4f}")
    print(f"  R-Precision:  {rp:.4f}")
    print(f"  NDCG@20:      {ndcg:.4f}")

    results.append((seed_pmid, p20, r20, rp, ndcg))

print("\n" + "=" * 60)
print("RESULTADOS FINAIS (média dos 4 seeds)")
print("=" * 60)
avg_p    = np.mean([r[1] for r in results])
avg_r    = np.mean([r[2] for r in results])
avg_rp   = np.mean([r[3] for r in results])
avg_ndcg = np.mean([r[4] for r in results])
print(f"  Precision@20: {avg_p:.4f}")
print(f"  Recall@20:    {avg_r:.4f}")
print(f"  R-Precision:  {avg_rp:.4f}")
print(f"  NDCG@20:      {avg_ndcg:.4f}")
print("\nConcluido!")
