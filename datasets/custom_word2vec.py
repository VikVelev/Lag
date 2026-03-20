import numpy as np
import random
from typing import List, Optional
import os
import pickle
import csv
from collections import defaultdict
import re


class SimpleWord2VecEmbedder:
    """
    The fully maximized pure-Python Word2Vec.
    Features: Pre-compiled integer corpus, Subsampling, LR Decay, Vectorization, and Pre-computed Negatives.
    """

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1,
                 learning_rate: float = 0.025, epochs: int = 5, negative_samples: int = 5,
                 subsample_threshold: float = 1e-3):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.subsample_threshold = subsample_threshold # New!

        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = defaultdict(int)
        self.vocab_size = 0
        self.sampling_probs = None 
        
        self.neg_table = None
        self.neg_table_size = 10_000_000
        self.neg_pointer = 0

        self.W1 = None  
        self.W2 = None  
        
        self.compiled_corpus = [] # New! Stores our pre-processed integer text

    def _build_vocab_and_corpus(self, sentences: List[List[str]]) -> None:
        """Builds vocab, calculates subsampling, and pre-compiles the corpus into integers."""
        self.word_freq.clear()
        self.word_to_idx.clear()
        self.idx_to_word.clear()
        self.compiled_corpus.clear()
        
        total_words_in_corpus = 0
        
        for sentence in sentences:
            for word in sentence:
                self.word_freq[word] += 1
                total_words_in_corpus += 1

        vocab = {word: freq for word, freq in self.word_freq.items() if freq >= self.min_count}
        self.word_to_idx = {word: i for i, word in enumerate(vocab.keys())}
        self.idx_to_word = {i: word for word, i in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        if self.vocab_size > 0:
            # 1. Unigram distribution for Negative Sampling
            freqs = np.array([vocab[self.idx_to_word[i]] for i in range(self.vocab_size)])
            pow_freqs = freqs ** 0.75
            self.sampling_probs = pow_freqs / np.sum(pow_freqs)
            
            print("Pre-computing negative sample table...")
            self.neg_table = np.random.choice(
                self.vocab_size, size=self.neg_table_size, p=self.sampling_probs
            )
            
            # 2. Subsampling Probabilities (Standard Word2Vec Formula)
            # Higher frequency = higher chance to be dropped
            word_fractions = freqs / total_words_in_corpus
            keep_probs = (np.sqrt(word_fractions / self.subsample_threshold) + 1) * (self.subsample_threshold / word_fractions)
            
            print("Pre-compiling and subsampling corpus to integers...")
            words_kept = 0
            words_dropped = 0
            
            for sentence in sentences:
                int_sentence = []
                for word in sentence:
                    if word in self.word_to_idx:
                        idx = self.word_to_idx[word]
                        # Subsampling: Roll the dice to see if we keep this word
                        if keep_probs[idx] >= 1.0 or random.random() < keep_probs[idx]:
                            int_sentence.append(idx)
                            words_kept += 1
                        else:
                            words_dropped += 1
                            
                if len(int_sentence) > 1: # Only keep sentences with at least 2 words left
                    self.compiled_corpus.append(np.array(int_sentence, dtype=np.int32))
                    
            print(f"Subsampling dropped {words_dropped} highly frequent words. Kept {words_kept}.")

        print(f"Vocabulary built: {self.vocab_size} words")

    def train(self, sentences: List[List[str]]) -> None:
        """Highly optimized training loop using pre-compiled corpus and LR decay."""
        print("Preparing data...")
        self._build_vocab_and_corpus(sentences)

        if self.vocab_size <= 1:
            raise ValueError("Not enough words found in vocabulary.")

        print("Initializing weights...")
        self.W1 = np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size, (self.vocab_size, self.vector_size))
        self.W2 = np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size, (self.vector_size, self.vocab_size))

        print(f"Training for {self.epochs} epochs...")

        labels = np.zeros(1 + self.negative_samples)
        labels[0] = 1.0 
        
        # Determine total words for accurate learning rate decay
        total_word_count = sum(len(sent) for sent in self.compiled_corpus) * self.epochs
        words_processed_global = 0
        current_lr = self.learning_rate

        for epoch in range(self.epochs):
            epoch_loss = 0
            pairs_count = 0

            # Iterate over our pre-compiled integer arrays! No more dictionary lookups!
            for indices in self.compiled_corpus:
                seq_len = len(indices)

                for i, center_idx in enumerate(indices):
                    # Linear Learning Rate Decay
                    words_processed_global += 1
                    if words_processed_global % 10000 == 0:
                        progress = words_processed_global / total_word_count
                        # Decay LR, but don't let it go below 0.0001
                        current_lr = max(self.learning_rate * (1.0 - progress), 0.0001)

                    # Dynamic Window Size (Standard Word2Vec trick for better context)
                    actual_window = random.randint(1, self.window)
                    start = max(0, i - actual_window)
                    end = min(seq_len, i + actual_window + 1)
                    
                    h = self.W1[center_idx] 

                    for j in range(start, end):
                        if i == j:  
                            continue
                        
                        context_idx = indices[j]

                        if self.neg_pointer + self.negative_samples > self.neg_table_size:
                            self.neg_pointer = 0 
                            
                        neg_indices = self.neg_table[self.neg_pointer : self.neg_pointer + self.negative_samples]
                        self.neg_pointer += self.negative_samples

                        target_indices = np.concatenate(([context_idx], neg_indices))
                        target_W2 = self.W2[:, target_indices] 
                        
                        z = np.dot(h, target_W2)
                        z = np.clip(z, -500, 500)
                        preds = 1.0 / (1.0 + np.exp(-z))

                        errors = (preds - labels) * current_lr
                        
                        grad_W2 = np.outer(h, errors)
                        grad_W1 = np.dot(target_W2, errors)

                        self.W2[:, target_indices] -= grad_W2
                        self.W1[center_idx] -= grad_W1

                        loss = -(labels * np.log(preds + 1e-10) + (1 - labels) * np.log(1 - preds + 1e-10))
                        epoch_loss += np.sum(loss)
                        pairs_count += 1 + self.negative_samples

            avg_loss = epoch_loss / pairs_count if pairs_count > 0 else 0
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.5f}")


    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        if word not in self.word_to_idx:
            return None
        return self.W1[self.word_to_idx[word]].copy()

    def get_similar_words(self, word: str, topn: int = 10) -> List[tuple]:
        if word not in self.word_to_idx:
            return []

        target_vec = self.get_embedding(word)
        similarities = []

        for other_word, idx in self.word_to_idx.items():
            if other_word != word:
                other_vec = self.W1[idx]
                dot_product = np.dot(target_vec, other_vec)
                norm_a = np.linalg.norm(target_vec)
                norm_b = np.linalg.norm(other_vec)
                similarity = (
                    dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
                )
                similarities.append((other_word, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

    def save_model(self, filepath: str) -> None:
        model_data = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "word_freq": dict(self.word_freq),
            "vocab_size": self.vocab_size,
            "vector_size": self.vector_size,
            "sampling_probs": self.sampling_probs,
            "W1": self.W1,
            "W2": self.W2,
            "config": {
                "window": self.window,
                "min_count": self.min_count,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "negative_samples": self.negative_samples,
            },
        }
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        self.word_to_idx = model_data["word_to_idx"]
        self.idx_to_word = model_data["idx_to_word"]
        self.word_freq = defaultdict(int, model_data["word_freq"])
        self.vocab_size = model_data["vocab_size"]
        self.vector_size = model_data["vector_size"]
        self.sampling_probs = model_data.get("sampling_probs")
        self.W1 = model_data["W1"]
        self.W2 = model_data["W2"]
        config = model_data["config"]
        self.window = config["window"]
        self.min_count = config["min_count"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.negative_samples = config["negative_samples"]
        print(f"Model loaded from {filepath}")

    def export_to_csv(
        self, output_path: str, words: Optional[List[str]] = None
    ) -> None:
        if words is None:
            words = list(self.word_to_idx.keys())
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["word"] + [f"dim_{i}" for i in range(self.vector_size)])
            exported_count = 0
            for word in words:
                embedding = self.get_embedding(word)
                if embedding is not None:
                    writer.writerow([word] + embedding.tolist())
                    exported_count += 1
        print(f"Successfully exported {exported_count} embeddings")

    def export_vocab_to_csv(self, output_path: str) -> None:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["word", "frequency"])
            for word in self.word_to_idx.keys():
                writer.writerow([word, self.word_freq[word]])
        print(f"Vocabulary exported to {output_path}")


def load_and_preprocess_corpus(filepath: str) -> List[List[str]]:
    sentences = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        for raw_sentence in re.split(r"[.!?]+", text):
            words = re.sub(r"[^\w\s]", "", raw_sentence.lower()).split()
            if words:
                sentences.append(words)
        print(f"Loaded {len(sentences)} sentences from '{filepath}'.")
        return sentences
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")
        return []


if __name__ == "__main__":
    corpus_sentences = load_and_preprocess_corpus("word2vec.train")
    if corpus_sentences:
        embedder = SimpleWord2VecEmbedder(
            vector_size=50, window=6, min_count=2, epochs=100
        )
        embedder.train(corpus_sentences)

        embedder.save_model("models/space_word2vec.pkl")
        embedder.export_to_csv("exports/space_embeddings.csv")
        embedder.export_vocab_to_csv("exports/space_vocab.csv")

        for word in ["planet", "gravity", "star", "trade", "astronomers"]:
            similar = embedder.get_similar_words(word, topn=3)
            print(f"\nWords similar to '{word}':")
            for w, score in similar:
                print(f"  - {w} ({score:.4f})")
