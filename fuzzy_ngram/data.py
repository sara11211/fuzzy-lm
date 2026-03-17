import re
from typing import List
from datasets import load_dataset
import random

def tokenize_text(text: str) -> List[str]:
    """Splits text into lowercase words, removes punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", "", text)
    return text.split()

def preprocess_corpus(corpus: List[str]) -> List[List[str]]:
    """Tokenizes a list of sentences."""
    return [tokenize_text(s) for s in corpus]

def load_corpus(path="roneneldan/TinyStories", split="validation"):
    ds = load_dataset(path, split=split)
    return [example["text"] for example in ds]

def load_corpus_subset(path="roneneldan/TinyStories", split="train", n_samples=2000000):
    corpus = load_corpus(path, split)
    if n_samples and n_samples < len(corpus):
        corpus = random.sample(corpus, n_samples)
    return corpus