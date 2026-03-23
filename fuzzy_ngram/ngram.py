import math
from typing import List, Dict, Tuple
from .utils import levenshtein, distance_similarity


class NGram:
    def __init__(self, alpha: float = 0.0, fuzzy: bool = False, n: int = 1):
        if n < 1 or n > 10:
            raise Exception(f"n={n} must be between 1 and 10")
        if not (0.0 <= alpha <= 1.0):
            raise Exception(f"alpha={alpha} must be between 0.0 and 1.0")
        
        self.n = n
        self.grams: Dict[str, int] = {}
        self.vocab = set()
        self.alpha = alpha
        self.fuzzy = fuzzy

    def fit(self, data: List[List[str]], min_freq: int = 1):
        if min_freq < 1:
            raise Exception("min_freq must be >= 1")
        
        self.grams = {}
        self.vocab = set()
        
        # Count word frequencies
        word_counts = {}
        for sentence in data:
            for word in sentence:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Build vocabulary and decide if <unk> is needed
        use_unk = False
        for word, count in word_counts.items():
            if count >= min_freq:
                self.vocab.add(word)
            else:
                use_unk = True
        if use_unk:
            self.vocab.add("<unk>")
        if self.n > 1:
            self.vocab.add("<s>")
            self.vocab.add("</s>")
        
        # Process sentences and extract n-grams
        for sentence in data:
            sent = sentence.copy()
            if use_unk:
                sent = [w if w in self.vocab else "<unk>" for w in sent]
            if self.n > 1:
                sent = ["<s>"] * (self.n - 1) + sent + ["</s>"] * (self.n - 1)

            if self.n == 1:
                for word in sent:
                    self.grams[word] = self.grams.get(word, 0) + 1
            else:
                for i in range(len(sent) - self.n + 1):
                    ngram_list = sent[i:i + self.n]
                    ngram_str = " ".join(ngram_list)
                    self.grams[ngram_str] = self.grams.get(ngram_str, 0) + 1

                    context_str = " ".join(ngram_list[:-1])
                    self.grams[context_str] = self.grams.get(context_str, 0) + 1

    def similar_word(self, word: str) -> Tuple[str, float]:
        best_word = ""
        best_sim = 0.0
        candidates = [w for w in self.vocab if abs(len(w) - len(word)) <= 3]
        for vocab_word in sorted(candidates):
            dist = levenshtein(word, vocab_word, sub=1)
            sim = distance_similarity(dist, len(word), len(vocab_word))
            if sim > best_sim:
                best_sim = sim
                best_word = vocab_word
        return best_word, best_sim

    def log_cond_prob(self, current_word: str, past_words: List[str]) -> float:
        discount = 1.0
        curr = current_word
        past = past_words.copy()
        
        if curr not in self.vocab:
            if self.fuzzy:
                sim_w, sim_s = self.similar_word(curr)
                if sim_s > 0:
                    curr = sim_w
                    discount *= sim_s
                else:
                    curr = '<unk>'
            else:
                curr = '<unk>'
        
        for i, w in enumerate(past):
            if w not in self.vocab:
                if self.fuzzy:
                    sim_w, sim_s = self.similar_word(w)
                    if sim_s > 0:
                        past[i] = sim_w
                        discount *= sim_s
                    else:
                        past[i] = '<unk>'
                else:
                    past[i] = '<unk>'
        
        context = past[-(self.n - 1):] if len(past) >= self.n - 1 else ['<s>'] * (self.n - 1 - len(past)) + past

        if self.n == 1:
            count_w = self.grams.get(curr, 0)
            total_count = sum(self.grams.values())
            V = len(self.vocab)
            prob = (count_w + self.alpha) / (total_count + self.alpha * V)
        else:
            context_str = " ".join(context)
            ngram_str = " ".join(context + [curr])
            count_ngram = self.grams.get(ngram_str, 0)
            count_context = self.grams.get(context_str, 0)
            V = len(self.vocab)
            prob = (count_ngram + self.alpha) / (count_context + self.alpha * V)

        final_prob = prob * discount
        return float('-inf') if final_prob <= 0 else math.log(final_prob)

    def log_text_prob(self, text: List[str]) -> float:
        text = text.copy() + ['</s>']
        log_prob = 0.0
        context = []
        for word in text:
            lcp = self.log_cond_prob(word, context)
            if lcp == float('-inf'):
                return float('-inf')
            log_prob += lcp
            context.append(word)
            if len(context) > self.n - 1:
                context = context[-(self.n - 1):]
        return log_prob

    def get_candidates(self, word: str, context: List[str], top_k: int = 5, lam: float = 0.5) -> List[Tuple[str, float, float, float]]:
        from .utils import top_k_similar_words
        similar = top_k_similar_words(word, self.vocab, k=top_k * 3)
        results = []
        for candidate, sim in similar:
            log_p = self.log_cond_prob(candidate, context)
            ctx_score = 0.0 if log_p == float('-inf') else math.exp(max(log_p, -20))
            combined = lam * sim + (1 - lam) * ctx_score
            results.append((candidate, sim, ctx_score, combined))
        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]

    def export_json(self):
        return self.__dict__.copy()

    def import_json(self, data):
        for key in data:
            self.__dict__[key] = data[key]