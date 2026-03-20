import numpy as np
from typing import List, Optional
import os
import pickle
import csv
from collections import defaultdict
import re

class SimpleWord2VecEmbedder:
    """
    A simplified Word2Vec implementation using skip-gram with negative sampling.
    Refactored for mathematical stability, memory safety, and unigram sampling.
    """

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1,
                 learning_rate: float = 0.01, epochs: int = 5, negative_samples: int = 5):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.negative_samples = negative_samples

        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = defaultdict(int)
        self.vocab_size = 0
        self.sampling_probs = None # Unigram distribution

        # Initialize weight matrices
        self.W1 = None  # Input to hidden
        self.W2 = None  # Hidden to output

    def _build_vocab(self, sentences: List[List[str]]) -> None:
        """Build vocabulary and unigram distribution from sentences."""
        # Reset state in case of retraining
        self.word_freq.clear()
        self.word_to_idx.clear()
        self.idx_to_word.clear()
        
        for sentence in sentences:
            for word in sentence:
                self.word_freq[word] += 1

        # Filter by min_count
        vocab = {word: freq for word, freq in self.word_freq.items() if freq >= self.min_count}

        self.word_to_idx = {word: i for i, word in enumerate(vocab.keys())}
        self.idx_to_word = {i: word for word, i in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        if self.vocab_size > 0:
            # Build Unigram Distribution (freq ^ 0.75) for Negative Sampling
            freqs = np.array([vocab[self.idx_to_word[i]] for i in range(self.vocab_size)])
            pow_freqs = freqs ** 0.75
            self.sampling_probs = pow_freqs / np.sum(pow_freqs)

        print(f"Vocabulary built: {self.vocab_size} words")

    def _initialize_weights(self) -> None:
        """Initialize weight matrices."""
        self.W1 = np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size,
                                   (self.vocab_size, self.vector_size))
        self.W2 = np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size,
                                   (self.vector_size, self.vocab_size))

    def _get_negative_samples(self, target_idx: int) -> List[int]:
        """Get negative samples using the unigram distribution."""
        if self.vocab_size <= 1:
            return []

        negatives = []
        # Oversample slightly to account for collisions with target_idx
        raw_samples = np.random.choice(self.vocab_size, size=self.negative_samples + 2, p=self.sampling_probs)
        
        for neg in raw_samples:
            if neg != target_idx:
                negatives.append(neg)
            if len(negatives) == self.negative_samples:
                break
                
        # Fallback if we still need more
        while len(negatives) < self.negative_samples:
            neg = np.random.choice(self.vocab_size, p=self.sampling_probs)
            if neg != target_idx:
                negatives.append(int(neg))
                
        return negatives

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function with overflow protection."""
        x = np.clip(x, -500, 500) # Prevent RuntimeWarning
        return 1.0 / (1.0 + np.exp(-x))

    def _train_pair(self, input_idx: int, output_idx: int, is_positive: bool) -> float:
        """Train a single input-output pair."""
        # Forward pass
        hidden = self.W1[input_idx]
        output = self.W2[:, output_idx]
        prediction = self._sigmoid(np.dot(hidden, output))

        # Calculate loss
        target = 1.0 if is_positive else 0.0
        loss = -(target * np.log(prediction + 1e-10) + (1 - target) * np.log(1 - prediction + 1e-10))

        # Backward pass
        error = prediction - target

        # Calculate gradients FIRST to avoid mutating views
        grad_W2 = self.learning_rate * error * hidden
        grad_W1 = self.learning_rate * error * output

        # Apply updates
        self.W2[:, output_idx] -= grad_W2
        self.W1[input_idx] -= grad_W1

        return loss

    def train(self, sentences: List[List[str]]) -> None:
        """Train the Word2Vec model."""
        print("Building vocabulary...")
        self._build_vocab(sentences)

        if self.vocab_size <= 1:
            raise ValueError("Not enough words found in vocabulary after filtering (need at least 2).")

        print("Initializing weights...")
        self._initialize_weights()

        print(f"Training for {self.epochs} epochs...")

        total_pairs = 0
        total_loss = 0

        for epoch in range(self.epochs):
            epoch_loss = 0
            pairs_count = 0

            for sentence in sentences:
                indices = [self.word_to_idx[word] for word in sentence if word in self.word_to_idx]

                for i, center_idx in enumerate(indices):
                    start = max(0, i - self.window)
                    end = min(len(indices), i + self.window + 1)

                    for j in range(start, end):
                        if i != j:  
                            context_idx = indices[j]

                            # Positive sample
                            loss = self._train_pair(center_idx, context_idx, True)
                            epoch_loss += loss
                            pairs_count += 1

                            # Negative samples
                            negatives = self._get_negative_samples(context_idx)
                            for neg_idx in negatives:
                                loss = self._train_pair(center_idx, neg_idx, False)
                                epoch_loss += loss
                                pairs_count += 1

            avg_loss = epoch_loss / pairs_count if pairs_count > 0 else 0
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

            total_loss += epoch_loss
            total_pairs += pairs_count

        avg_total_loss = total_loss / total_pairs if total_pairs > 0 else 0
        print(f"Training completed. Total pairs processed: {total_pairs}, Average loss: {avg_total_loss:.4f}")

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding for a word."""
        if word not in self.word_to_idx:
            return None
        return self.W1[self.word_to_idx[word]].copy()

    def get_similar_words(self, word: str, topn: int = 10) -> List[tuple]:
        """Get most similar words using cosine similarity."""
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
                similarity = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
                similarities.append((other_word, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

    def save_model(self, filepath: str) -> None:
        """Save model to file safely."""
        model_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'word_freq': dict(self.word_freq),
            'vocab_size': self.vocab_size,
            'vector_size': self.vector_size,
            'sampling_probs': self.sampling_probs,
            'W1': self.W1,
            'W2': self.W2,
            'config': {
                'window': self.window,
                'min_count': self.min_count,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
                'negative_samples': self.negative_samples
            }
        }

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.word_to_idx = model_data['word_to_idx']
        self.idx_to_word = model_data['idx_to_word']
        self.word_freq = defaultdict(int, model_data['word_freq'])
        self.vocab_size = model_data['vocab_size']
        self.vector_size = model_data['vector_size']
        self.sampling_probs = model_data.get('sampling_probs')
        self.W1 = model_data['W1']
        self.W2 = model_data['W2']

        config = model_data['config']
        self.window = config['window']
        self.min_count = config['min_count']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.negative_samples = config['negative_samples']

        print(f"Model loaded from {filepath}")

    def export_to_csv(self, output_path: str, words: Optional[List[str]] = None) -> None:
        """Export embeddings to CSV."""
        if words is None:
            words = list(self.word_to_idx.keys())

        print(f"Exporting {len(words)} embeddings to {output_path}")

        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ['word'] + [f'dim_{i}' for i in range(self.vector_size)]
            writer.writerow(header)

            exported_count = 0
            for word in words:
                embedding = self.get_embedding(word)
                if embedding is not None:
                    row = [word] + embedding.tolist()
                    writer.writerow(row)
                    exported_count += 1

        print(f"Successfully exported {exported_count} embeddings")

    def export_vocab_to_csv(self, output_path: str) -> None:
        """Export strictly the filtered vocabulary with frequencies."""
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['word', 'frequency'])

            # Only export words that survived min_count filtering
            for word in self.word_to_idx.keys():
                writer.writerow([word, self.word_freq[word]])

        print(f"Vocabulary exported to {output_path}")

def load_and_preprocess_corpus(filepath: str) -> List[List[str]]:
    """Reads a text file and tokenizes it into a list of sentences (lists of words)."""
    sentences = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        raw_sentences = re.split(r'[.!?]+', text)
        for raw_sentence in raw_sentences:
            clean_sentence = re.sub(r'[^\w\s]', '', raw_sentence.lower())
            words = clean_sentence.split()
            if words:
                sentences.append(words)
        print(f"Loaded {len(sentences)} sentences from '{filepath}'.")
        return sentences
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")
        return []

# Example usage/testing block
if __name__ == "__main__":
    # Make sure you have a 'word2vec.train' file in the same directory, 
    # or replace this with a list of sample sentences to test immediately.
    corpus_sentences = load_and_preprocess_corpus("word2vec.train")

    if corpus_sentences:
        embedder = SimpleWord2VecEmbedder(vector_size=50, window=6, min_count=2, epochs=500)
        
        # Train
        embedder.train(corpus_sentences)
        
        # Export and Save
        embedder.save_model("models/space_word2vec.pkl")
        embedder.export_to_csv("exports/space_embeddings.csv")
        embedder.export_vocab_to_csv("exports/space_vocab.csv")
        
        # Test similarities
        test_words = ["planet", "gravity", "star", "trade", "astronomers"]
        for word in test_words:
            similar = embedder.get_similar_words(word, topn=3)
            print(f"\nWords similar to '{word}':")
            for w, score in similar:
                print(f"  - {w} ({score:.4f})")
    else:
        print("No training data found. Create 'word2vec.train' to run the full pipeline.")