import math
from typing import List, Dict, Tuple
from .utils import levenshtein, distance_similarity

#=============================================================================
#                             The N-gram Class
#=============================================================================
class NGram:

    def __init__(self, alpha: float = 0.0, fuzzy: bool = False, n: int = 1):
        if n < 1:
            raise Exception(f"n={n}; n must be >= 1")
        if n > 10:
            raise Exception(f"n={n}; we limited n to 10 at most")
        
        if not (0.0 <= alpha <= 1.):
            raise Exception(f"alpha={alpha} must be between 0.0 and 1.0")
        
        self.n = n
        self.grams: Dict[str, int] = {} 
        self.vocab = set()
        self.alpha = alpha
        self.fuzzy = fuzzy


    def fit(self, data:List[List[str]], min_freq: int = 1):
        """
        Trains the n-gram model on the given data.

        Args:
            data (List[List[str]]): A list of sentences, where each sentence is itself a list of words (tokens).
                Example: [['the', 'cat', 'sat'], ['a', 'dog', 'ran']]
            min_freq (int): Minimum frequency threshold for n-grams to be included.
        Returns:
            None: This method updates the model in-place and does not return anything.
        """
        if min_freq < 1:
            raise Exception("min_freq must be 1 or plus")
        
        self.grams = {}
        self.vocab = set()
        
        # ---------------------------
        # 1. Count word frequencies
        # ---------------------------
        word_counts = {}
        for sentence in data:
            for word in sentence:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # ---------------------------
        # 2. Build vocabulary
        # ---------------------------
        use_unk = False
        for word, count in word_counts.items():
            if count >= min_freq:
                self.vocab.add(word)
            else:
                use_unk = True

        # <unk> is added to the vocab 
        # only if a token was replaced by it
        if use_unk:
            self.vocab.add("<unk>")

        # add <s> and </s> if needed
        if self.n > 1:
            self.vocab.add("<s>")
            self.vocab.add("</s>")
        
        # ---------------------------
        # 3. Process sentences
        # ---------------------------
        for sentence in data:

            sent = sentence.copy()

            # replace rare words
            if use_unk:
                sent = [w if w in self.vocab else "<unk>" for w in sent]

            # add boundaries
            if self.n > 1:
                sent = ["<s>"] * (self.n - 1) + sent + ["</s>"] * (self.n - 1)

            # -----------------------------------
            # 4. Extract n-grams and (n-1)-grams
            # -----------------------------------
            # For unigrams (n=1), count words          
            if self.n == 1:
                for word in sent:
                    self.grams[word] = self.grams.get(word, 0) + 1
            else:
                # For n-grams with n > 1, count the n-gram and the (n-1) gram
                for i in range(len(sent) - self.n + 1):
                    ngram_list = sent[i: i + self.n]
                    ngram_str = " ".join(ngram_list)
                    self.grams[ngram_str] = self.grams.get(ngram_str, 0) + 1

                    # The context is the n-gram without the last word
                    context_list = ngram_list[:-1]
                    context_str = " ".join(context_list)
                    self.grams[context_str] = self.grams.get(context_str, 0) + 1


    def similar_word(self, word: str) -> Tuple[str, float]:
        """
            Finds the most similar word in the vocabulary to the given word.
            
            Args:
                word (str): The word to find a similar match for.
                
            Returns:
                Tuple[str, float]: The most similar word from vocab and its similarity score.
            """
        best_word = ""
        best_sim = 0.0
        
        # ---------------------------
        # 1. Process vocab words
        # ---------------------------
        candidates = [w for w in self.vocab if abs(len(w) - len(word)) <= 3]
        for vocab_word in sorted(candidates):
            dist = levenshtein(word, vocab_word, sub=1)
            sim = distance_similarity(dist, len(word), len(vocab_word))
            
            # ---------------------------
            # 2. Find most similar word
            # ---------------------------
            if sim > best_sim:
                best_sim = sim
                best_word = vocab_word
        
        return best_word, best_sim


    def log_cond_prob(self, current_word: str, past_words: List[str]) -> float:
        """
        Calculates the conditional log probability of the current_word given the past_words context.

        Args:
            current_word (str): The word for which the probability is to be computed.
            past_words (List[str]): The list of previous words (context).

        Returns:
            float: The conditional log probability of current_word given past_words.
        """
        
        discount = 1.0
        curr = current_word
        past = past_words.copy()
        
        # ---------------------------
        # 1. Fuzzy logic
        # ---------------------------
        if curr not in self.vocab:
            # Handle current word replacement
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
            # Handle past words replacement
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
                 
        # ---------------------------
        # 2. Context Construction
        # ---------------------------   
        context = past
        needed_len = self.n - 1
        current_len = len(context)

        # If the context is too short, pad with <s>
        if current_len < needed_len:
            num_padding = needed_len - current_len
            context = ['<s>'] * num_padding + context
        
        # If it's too long, keep only the last (n-1) words
        elif current_len > needed_len:
            context = context[-needed_len:]
        
        
        # ---------------------------
        # 3. Probability Calculation
        # ---------------------------
        # Case Unigram (n=1)
        if self.n == 1:
            count_w = self.grams.get(curr, 0)
            total_count = sum(self.grams.values())

            V = len(self.vocab)
            prob = (count_w + self.alpha) / (total_count + self.alpha * V)

        else:
            # Case N-gram > 1
            context_str = " ".join(context)
            ngram_str = " ".join(context + [curr])

            count_ngram = self.grams.get(ngram_str, 0)
            count_context = self.grams.get(context_str, 0)

            V = len(self.vocab)

            prob = (count_ngram + self.alpha) / (count_context + self.alpha * V)

        # Final probability = probability * similarity_score
        final_prob = prob * discount

        if final_prob <= 0:
            return float('-inf')

        return math.log(final_prob)


    def log_text_prob(self, text: List[str]) -> float:
        """
        Calculates the log probability of a given text sequence.
        Args:
            text (List[str]): A list of words/tokens representing the text sequence.
        Returns:
            float: The log probability of the text sequence.
        """
        text = text.copy()
        
        # Append </s> to the text
        text.append('</s>')

        log_prob = 0.0
        context = []

        for word in text:
            lcp = self.log_cond_prob(word, context)

            if lcp == float('-inf'):
                return float('-inf')
            
            # Sum the log probability of each word giving its context
            log_prob += lcp

            context.append(word)
            if len(context) > self.n - 1:
                # Update the context for the next word
                context = context[-(self.n - 1):]

        return log_prob
    
    def get_candidates(
        self,
        word: str,
        context: List[str],
        top_k: int = 5,
        lam: float = 0.5
    ) -> List[Tuple[str, float, float, float]]:
        """
        Returns ranked correction candidates for an unknown word,
        combining string similarity and context probability.

        Args:
            word (str): The misspelled word.
            context (List[str]): The preceding words (already corrected).
            top_k (int): Number of candidates to return.
            lam (float): Weight for similarity vs context. 
                        1.0 = pure similarity, 0.0 = pure context.

        Returns:
            List of (candidate, similarity, context_prob, combined_score)
            sorted by combined_score descending.
        """
        from .utils import top_k_similar_words

        similar = top_k_similar_words(word, self.vocab, k=top_k * 3)

        results = []
        for candidate, sim in similar:
            log_p = self.log_cond_prob(candidate, context)

            # Convert log prob to a 0-1 score
            # log_p ranges roughly from -inf to 0, so we exponentiate
            # and clip to avoid -inf causing issues
            if log_p == float('-inf'):
                ctx_score = 0.0
            else:
                # Normalize: e^log_p gives raw prob, we scale it to [0,1]
                # by capping at a reasonable minimum log value
                ctx_score = math.exp(max(log_p, -20))

            combined = lam * sim + (1 - lam) * ctx_score
            results.append((candidate, sim, ctx_score, combined))

        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]
    
    # -------------------
    # Persistence
    # -------------------
    def export_json(self):
        return self.__dict__.copy()

    def import_json(self, data):
        for cle in data:
            self.__dict__[cle] = data[cle]