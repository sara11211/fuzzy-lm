import sys
import os
import pickle

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fuzzy_ngram.ngram import NGram
from fuzzy_ngram.data import preprocess_corpus, load_corpus
from fuzzy_ngram.corrector import Corrector
from fuzzy_ngram.cli import run_cli

# Cache folder
CACHE_DIR = os.path.join(project_root, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

PREPROCESSED_DATA_FILE = os.path.join(CACHE_DIR, "preprocessed_data.pkl")
MODEL_FILE = os.path.join(CACHE_DIR, "ngram_model.pkl")


def main():
    # 1. Load or preprocess data
    if os.path.exists(PREPROCESSED_DATA_FILE):
        print("[cache] Loading preprocessed data...")
        with open(PREPROCESSED_DATA_FILE, "rb") as f:
            data = pickle.load(f)
    else:
        print("[cache] Preprocessing corpus (this may take a while)...")
        corpus = load_corpus()
        data = preprocess_corpus(corpus)
        with open(PREPROCESSED_DATA_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"[cache] Saved preprocessed data to {PREPROCESSED_DATA_FILE}")

    # 2. Load or train model
    if os.path.exists(MODEL_FILE):
        print("[cache] Loading trained model...")
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
    else:
        print("[cache] Training model (this may take a while)...")
        model = NGram(alpha=0.1, fuzzy=True, n=3)
        model.fit(data, min_freq=5)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        print(f"[cache] Saved trained model to {MODEL_FILE}")

    # 3. Create corrector
    corrector = Corrector(model, top_k=5, lam=0.6, min_sim=0.6)

    # 4. Run CLI
    run_cli(corrector, verbose=False)


if __name__ == "__main__":
    main()