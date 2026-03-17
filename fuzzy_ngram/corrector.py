from typing import List, Tuple, Dict
from .ngram import NGram
from .data import tokenize_text


# Represents the correction info for a single word
WordResult = Dict  # keys: original, corrected, changed, candidates


class Corrector:
    """
    Wraps an NGram model to correct a full sentence word by word.
    For each unknown word, generates candidates ranked by combined
    string similarity + context probability score.
    """

    def __init__(self, model: NGram, top_k: int = 5, lam: float = 0.5, min_sim: float = 0.5):
        """
        Args:
            model (NGram): A trained NGram model.
            top_k (int): Number of candidates to show per corrected word.
            lam (float): Balance between similarity and context (0.0 to 1.0).
            min_sim (float): Minimum similarity score to accept a candidate.
                             If no candidate meets this, word is marked as unknown.
        """
        self.model = model
        self.top_k = top_k
        self.lam = lam
        self.min_sim = min_sim

    def correct_sentence(self, sentence: str) -> List[WordResult]:
        """
        Corrects a sentence and returns per-word results.

        Args:
            sentence (str): Raw input sentence (may contain typos).

        Returns:
            List of dicts, one per word, each containing:
                - original (str): the word as typed
                - corrected (str): the best correction (or original if already known)
                - changed (bool): whether the word was changed
                - unknown (bool): True if no good candidate was found
                - candidates (list): top-k list of (word, sim, ctx, combined)
        """
        tokens = tokenize_text(sentence)
        results = []
        corrected_context = []  # grows as we process left to right

        for word in tokens:
            # Word is already in vocabulary — no correction needed
            if word in self.model.vocab:
                results.append({
                    "original": word,
                    "corrected": word,
                    "changed": False,
                    "unknown": False,
                    "candidates": []
                })
                corrected_context.append(word)
                continue

            # Word is unknown — get candidates
            candidates = self.model.get_candidates(
                word,
                corrected_context,
                top_k=self.top_k,
                lam=self.lam
            )

            # Check if best candidate meets minimum similarity threshold
            if candidates and candidates[0][1] >= self.min_sim:
                best = candidates[0][0]
                results.append({
                    "original": word,
                    "corrected": best,
                    "changed": True,
                    "unknown": False,
                    "candidates": candidates
                })
                corrected_context.append(best)
            else:
                # No confident match found
                results.append({
                    "original": word,
                    "corrected": word,
                    "changed": False,
                    "unknown": True,
                    "candidates": candidates
                })
                corrected_context.append(word)

        return results