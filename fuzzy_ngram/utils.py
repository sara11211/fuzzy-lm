from typing import Tuple, List
from termcolor import colored

#=============================================================================
#                             Functions
#=============================================================================

def levenshtein(w1:str, w2:str, sub:int=2) -> int:
    """Calculates Levenshtein distance between two words.
    The function's words are interchangeable; i.e. levenshtein(w1, w2) = levenshtein(w2, w1)

    Args:
        w1 (str): First word.
        w2 (str): Second word.
        sub (int, optional): Substitution's cost. Defaults to 2.

    Returns:
        int: distance
    """

    if len(w1) * len(w2) == 0:
        return max([len(w1), len(w2)])

    D = []
    D.append([i for i in range(len(w2) + 1)])
    for i in range(len(w1)):
        l = [i+1]
        for j in range(len(w2)):
            s = D[i][j] + (0 if w1[i] == w2[j] else sub)
            m = min([s, D[i][j+1] + 1, l[j] + 1])
            l.append(m)
        D.append(l)

    return D[-1][-1]


def distance_similarity(d: int, l1: int, l2: int) -> float:
    """
    Converts a distance into a similarity score between 0 and 1.
    """
    m = max(l1, l2)
    return (m-d)/m

def get_word_corrections(ngram, sentence):
    corrections = []
    corrected_sentence = []

    for word in sentence:
        # If word is already known, keep it
        if word in ngram.vocab:
            corrected_sentence.append(word)
            continue

        # Otherwise try fuzzy match
        sim_word, sim_score = ngram.similar_word(word)

        # Apply threshold (IMPORTANT FIX)
        if sim_word and sim_score > 0.6:
            corrections.append((word, sim_word, sim_score))
            corrected_sentence.append(sim_word)
        else:
            # Keep original word if not confident
            corrected_sentence.append(word)

    return corrections, corrected_sentence


def top_k_similar_words(word: str, vocab: set, k: int = 5) -> List[Tuple[str, float]]:
    """
    Returns the top-k most similar words from vocab to the given word,
    ranked by similarity score (highest first).

    Args:
        word (str): The input word to match.
        vocab (set): The vocabulary to search in.
        k (int): Number of top candidates to return.

    Returns:
        List[Tuple[str, float]]: List of (vocab_word, similarity_score) pairs.
    """
    candidates = [w for w in vocab if abs(len(w) - len(word)) <= 3]
    scored = []
    for vocab_word in candidates:
        dist = levenshtein(word, vocab_word, sub=1)
        sim = distance_similarity(dist, len(word), len(vocab_word))
        if sim > 0:
            scored.append((vocab_word, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]