from typing import List, Dict
from .ngram import NGram
from .data import tokenize_text

WordResult = Dict


class Corrector:
    def __init__(self, model: NGram, top_k: int = 5, lam: float = 0.5, min_sim: float = 0.5):
        self.model = model
        self.top_k = top_k
        self.lam = lam
        self.min_sim = min_sim

    def correct_sentence(self, sentence: str) -> List[WordResult]:
        tokens = tokenize_text(sentence)
        results = []
        corrected_context = []

        for word in tokens:

            # Get candidate corrections ranked by combined score
            candidates = self.model.get_candidates(
                word,
                corrected_context,
                top_k=self.top_k,
                lam=self.lam
            )

            # Add the original word only if it isn't already in the list,
            # so it shows up in the candidates panel but doesn't force itself to the top
            candidate_words = {c[0] for c in candidates}
            if word not in candidate_words:
                candidates.append((word, 1.0, 0.0, 0.0))

            # Pick best candidate by combined score (candidates are already sorted)
            best_word = word
            changed = False
            unknown = False

            if candidates:
                top_word, top_sim, top_ctx, top_combined = candidates[0]

                if top_sim < self.min_sim:
                    # Model isn't confident enough — leave the word as-is
                    unknown = True
                    best_word = word
                else:
                    best_word = top_word
                    if best_word != word:
                        changed = True
            else:
                unknown = True

            results.append({
                "original": word,
                "corrected": best_word,
                "changed": changed,
                "unknown": unknown,
                "candidates": candidates[:self.top_k]
            })

            corrected_context.append(best_word)

        return results