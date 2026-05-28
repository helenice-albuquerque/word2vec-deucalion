import os
import gc
import glob
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.utils import simple_preprocess

# ── Limitar threads (cluster) ─────────────────────────────────────────────
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("train_w2v.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Checkpoint a cada epoch ───────────────────────────────────────────────
class CheckpointCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    def on_epoch_end(self, model):
        self.epoch += 1
        path = f"word2vec_epoch{self.epoch}.model"
        model.save(path)
        log.info(f"Checkpoint salvo: {path}")

# ── Iterador ──────────────────────────────────────────────────────────────
class Sentences:
    def __iter__(self):

        # -------- WIKIPEDIA --------
        wiki_files = sorted(glob.glob(
            "/projects/F202600026AIVLABDEUCALION/helenice/data/wikipedia/train_part_*.parquet"
        ))
        log.info(f"Wikipedia: {len(wiki_files)} ficheiros encontrados")

        for fpath in wiki_files:
            log.info(f"  -> A ler: {os.path.basename(fpath)}")
            try:
                pf = pq.ParquetFile(fpath)
                for batch in pf.iter_batches(batch_size=500, columns=["text"]):
                    for text in batch["text"].to_pylist():
                        if text:
                            yield simple_preprocess(text)
            except Exception as e:
                log.warning(f"Erro em {fpath}: {e} — a saltar")
            gc.collect()

        # -------- PUBMED --------
        pubmed_dir = "/projects/F202600026AIVLABDEUCALION/helenice/data/pubmed"
        log.info(f"A processar PubMed em: {pubmed_dir}")

        for root, _, files in os.walk(pubmed_dir):
            for file in files:                          # <-- linha que faltava
                if file.endswith(".txt"):
                    fpath = os.path.join(root, file)
                    log.info(f"  -> PubMed: {file}")
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    yield simple_preprocess(line)
                    except Exception as e:
                        log.warning(f"Erro em {fpath}: {e} — a saltar")

# ── Treino ────────────────────────────────────────────────────────────────
log.info("A iniciar treino...")
sentences = Sentences()

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=1,
    epochs=5,
    callbacks=[CheckpointCallback()]
)

# ── Guardar ───────────────────────────────────────────────────────────────
model.save("word2vec_final.model")
model.wv.save_word2vec_format("word2vec_final.bin", binary=True)
log.info("Treino concluido! Modelos guardados.")
