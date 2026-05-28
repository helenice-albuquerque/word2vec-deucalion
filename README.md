# word2vec-deucalion

Treino de embeddings biomédicos com Word2Vec utilizando artigos científicos do PubMed e Wikipedia no cluster Deucalion (HPC).

O projecto inclui:

* treino de Word2Vec biomédico
* testes de similaridade semântica
* scripts SLURM para execução no cluster
* integração com modelos de classificação biomédica (HSLN)

Tecnologias:

* Python
* Gensim
* Word2Vec
* SLURM
* HPC / Deucalion
* NLP biomédico

Files:
1. word2vec_med.py
Treino do Word2Vec com Wikipedia + PubMed. O principal código do projecto — gerou o modelo com 3.3M palavras.
2. job_w2v.sh
Script SLURM para submeter o treino do Word2Vec no cluster sem perder por desconexão.
3. test_model.py
Teste básico do Word2Vec — palavras similares a "cancer" e "heart".
4. test_model2.py
Teste avançado do Word2Vec — analogias, similaridades, palavras estranhas, domínio biomédico.
5. HSLN-Joint-Sentence-Classification/ (repositório clonado)
Classificador hierárquico de frases biomédicas — treinado com PubMed-20k-RCT. Inclui train.py, build_data.py e model/config.py (que modificámos).
6. job_hsln.sh
Script SLURM para submeter o treino do HSLN no cluster.
7. tfidf_facets.py
Módulo TF-IDF simples — comparou os 4 artigos seed do SMAFIRA-c entre si (teste do módulo).
8. tfidf_smafira.py
Módulo TF-IDF completo — baixou abstracts do PubMed, calculou similaridade coseno com os facets e avaliou com as métricas da Tabela 2 (precision, recall, r-p, ndcg).

Estado atual:

Estado atual:
* Word2Vec treinado — 3.3M palavras, 92% cobertura biomédica
* Classificador HSLN treinado — 92.05% accuracy no PubMed-20k-RCT
* Módulo TF-IDF implementado — NDCG@20 de 0.5588 (paper: 0.57)
