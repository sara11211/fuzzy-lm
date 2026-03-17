import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from fuzzy_ngram.ngram import NGram
from fuzzy_ngram.data import preprocess_corpus, load_corpus, load_corpus_subset
from fuzzy_ngram.corrector import Corrector
from fuzzy_ngram.cli import run_cli

# 1. Load and preprocess data
corpus = load_corpus_subset(n_samples=20000)
data = preprocess_corpus(corpus)

# 2. Train model
model = NGram(alpha=0.1, fuzzy=False, n=3)  
model.fit(data, min_freq=2)
    
# 3. Create corrector
corrector = Corrector(model, top_k=5, lam=0.6, min_sim=0.6)

# 4. Run CLI
run_cli(corrector, verbose=False)