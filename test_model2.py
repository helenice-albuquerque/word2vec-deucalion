from gensim.models import Word2Vec

model = Word2Vec.load("word2vec_final.model")

print("=" * 50)
print("ANALOGIAS")
print("=" * 50)

print("\nking + woman - man =")
print(model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=5))

print("\ndoctor + woman - man =")
print(model.wv.most_similar(positive=["doctor", "woman"], negative=["man"], topn=5))

print("\nparis + germany - france =")
print(model.wv.most_similar(positive=["paris", "germany"], negative=["france"], topn=5))

print("\n" + "=" * 50)
print("SIMILARIDADES")
print("=" * 50)

pairs = [
    ("cancer", "tumor"),
    ("cancer", "banana"),
    ("heart", "cardiac"),
    ("brain", "neural"),
    ("virus", "bacteria"),
]
for a, b in pairs:
    print(f"  {a} <-> {b}: {model.wv.similarity(a, b):.4f}")

print("\n" + "=" * 50)
print("PALAVRA ESTRANHA")
print("=" * 50)

groups = [
    ["cancer", "tumor", "melanoma", "banana"],
    ["neuron", "synapse", "cortex", "ocean"],
    ["paris", "london", "berlin", "liver"],
]
for g in groups:
    odd = model.wv.doesnt_match(g)
    print(f"  {g} → estranha: '{odd}'")

print("\n" + "=" * 50)
print("DOMINIO BIOMEDICO")
print("=" * 50)

for word in ["diabetes", "neuron", "vaccine", "genome", "surgery"]:
    similar = model.wv.most_similar(word, topn=5)
    words_only = [w for w, _ in similar]
    print(f"\n  {word}: {words_only}")

