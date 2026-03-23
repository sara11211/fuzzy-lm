import re
import random
from typing import List
from datasets import load_dataset


def tokenize_text(text: str) -> List[str]:
    """Lowercase + remove punctuation while keeping apostrophes."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", "", text)
    return text.split()


def preprocess_corpus(corpus: List[str]) -> List[List[str]]:
    """Tokenize a list of sentences."""
    return [tokenize_text(s) for s in corpus]


def load_corpus(path="roneneldan/TinyStories", split="validation"):
    ds = load_dataset(path, split=split)
    return [example["text"] for example in ds]


def load_corpus_subset(
    path="roneneldan/TinyStories",
    split="train",
    n_samples=2119719
):
    """Load a random subset of the dataset."""
    ds = load_dataset(path, split=split)

    print(len(ds))
    if n_samples and n_samples < len(ds):
        indices = random.sample(range(len(ds)), n_samples)
        ds = ds.select(indices)

    return [example["text"] for example in ds]