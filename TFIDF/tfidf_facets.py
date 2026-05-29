from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import itertools

FACETS_DIR = "/projects/F202600026AIVLABDEUCALION/helenice/reranking/data/facets"
PMIDS = ["16850029", "19735549", "21494637", "24204323"]

def load_facets(pmid, mode="intersect"):
    filepath = os.path.join(FACETS_DIR, f"{pmid}_{mode}.txt")
    facets = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t", 1)
                if len(parts) == 2 and parts[0] == "1":
                    facets.append(parts[1])
    return " ".join(facets)

def load_full_abstract(pmid):
    filepath = os.path.join(FACETS_DIR, f"{pmid}.txt")
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    return " ".join(sentences)

print("=" * 60)
print("COMPARAÇÃO COM FACETS (interseção dos 3 anotadores)")
print("=" * 60)

facets_docs = {pmid: load_facets(pmid) for pmid in PMIDS}
full_docs   = {pmid: load_full_abstract(pmid) for pmid in PMIDS}

for pmid, text in facets_docs.items():
    n_words = len(text.split())
    print(f"  PMID {pmid}: {n_words} palavras nos facets")

def compare_documents(docs_dict, label):
    print(f"\n{'=' * 60}")
    print(f"SIMILARIDADE TF-IDF — {label}")
    print("=" * 60)
    pmids = list(docs_dict.keys())
    texts = list(docs_dict.values())
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    for i, j in itertools.combinations(range(len(pmids)), 2):
        sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
        print(f"  PMID {pmids[i]} vs PMID {pmids[j]}: {sim:.4f}")

compare_documents(facets_docs, "Usando só os FACETS")
compare_documents(full_docs,   "Usando o ABSTRACT COMPLETO")

print("\nConcluido!")
