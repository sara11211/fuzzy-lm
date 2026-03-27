# fuzzy-lm

Sentence correction using N-Gram language models with optional fuzzy matching.

<img src="https://i.imgur.com/nl6CNw9.png" width="80%" align="center"/>

## About

> This project is a **tweaked version** of the N-Gram NLP lab done as part of the NLP course at **ESI**. Trained on a subset of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) from Hugging Face for demonstration purpose. The subset is relatively small so correction may not always be accurate and the model is **not highly efficient** for large-scale text.  

## Features

- N-Gram model implementation
- Unknown word handling via `<unk>` token
- Fuzzy matching for out-of-vocabulary words
- Model caching — train once, reload instantly
- Interactive CLI with normal and verbose modes

## Installation
```bash
git clone https://github.com/sara11211/fuzzy-lm.git
cd fuzzy-lm
pip install -r requirements.txt
```

## Usage
```bash
python scripts/run.py
```

The model trains and caches on first run. Subsequent runs load instantly from `cache/`.
