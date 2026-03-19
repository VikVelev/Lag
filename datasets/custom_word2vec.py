import numpy as np
from typing import List, Optional
import os
import pickle
import csv
from collections import defaultdict
import random

class SimpleWord2VecEmbedder:
    """
    A simplified word2vec implementation using skip-gram with negative sampling.
    More stable than Gensim for certain environments.
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

        # Initialize weight matrices
        self.W1 = None  # Input to hidden
        self.W2 = None  # Hidden to output

    def _build_vocab(self, sentences: List[List[str]]) -> None:
        """Build vocabulary from sentences."""
        for sentence in sentences:
            for word in sentence:
                self.word_freq[word] += 1

        # Filter by min_count
        vocab = {word: freq for word, freq in self.word_freq.items() if freq >= self.min_count}

        self.word_to_idx = {word: i for i, word in enumerate(vocab.keys())}
        self.idx_to_word = {i: word for word, i in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)

        print(f"Vocabulary built: {self.vocab_size} words")

    def _initialize_weights(self) -> None:
        """Initialize weight matrices."""
        self.W1 = np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size,
                                   (self.vocab_size, self.vector_size))
        self.W2 = np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size,
                                   (self.vector_size, self.vocab_size))

    def _get_negative_samples(self, target_idx: int) -> List[int]:
        """Get negative samples for training."""
        negatives = []
        while len(negatives) < self.negative_samples:
            neg = random.randint(0, self.vocab_size - 1)
            if neg != target_idx:
                negatives.append(neg)
        return negatives

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

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

        # Update weights
        self.W2[:, output_idx] -= self.learning_rate * error * hidden
        self.W1[input_idx] -= self.learning_rate * error * output

        return loss

    def train(self, sentences: List[List[str]]) -> None:
        """Train the Word2Vec model."""
        print("Building vocabulary...")
        self._build_vocab(sentences)

        if self.vocab_size == 0:
            raise ValueError("No words found in vocabulary after filtering")

        print("Initializing weights...")
        self._initialize_weights()

        print(f"Training for {self.epochs} epochs...")

        total_pairs = 0
        total_loss = 0

        for epoch in range(self.epochs):
            epoch_loss = 0
            pairs_count = 0

            for sentence in sentences:
                # Convert words to indices, skip unknown words
                indices = [self.word_to_idx[word] for word in sentence if word in self.word_to_idx]

                for i, center_idx in enumerate(indices):
                    # Define context window
                    start = max(0, i - self.window)
                    end = min(len(indices), i + self.window + 1)

                    for j in range(start, end):
                        if i != j:  # Skip center word
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
                # Cosine similarity
                dot_product = np.dot(target_vec, other_vec)
                norm_a = np.linalg.norm(target_vec)
                norm_b = np.linalg.norm(other_vec)
                similarity = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
                similarities.append((other_word, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        model_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'word_freq': dict(self.word_freq),
            'vocab_size': self.vocab_size,
            'vector_size': self.vector_size,
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

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
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

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            header = ['word'] + [f'dim_{i}' for i in range(self.vector_size)]
            writer.writerow(header)

            # Write embeddings
            exported_count = 0
            for word in words:
                embedding = self.get_embedding(word)
                if embedding is not None:
                    row = [word] + embedding.tolist()
                    writer.writerow(row)
                    exported_count += 1

        print(f"Successfully exported {exported_count} embeddings")

    def export_vocab_to_csv(self, output_path: str) -> None:
        """Export vocabulary with frequencies."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['word', 'frequency'])

            for word, freq in sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([word, freq])

        print(f"Vocabulary exported to {output_path}")


# Simple tokenizer
def simple_tokenizer(text: str) -> List[str]:
    """Simple whitespace-based tokenizer."""
    return text.lower().split()


# Example usage
if __name__ == "__main__":
    # Create embedder
    embedder = SimpleWord2VecEmbedder(vector_size=50, window=3, min_count=1, epochs=10)

    # Sample training data
    sentences = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "ran", "in", "the", "park"],
        ["cats", "and", "dogs", "are", "pets"],
        ["machine", "learning", "is", "powerful"],
        ["word", "embeddings", "represent", "meaning"]
    ]

    # Train
    embedder.train(sentences)

    # Export
    embedder.export_to_csv("datasets/embeddings.csv")
    embedder.export_vocab_to_csv("datasets/vocabulary.csv")

    # Test
    similar = embedder.get_similar_words("cat", topn=3)
    print(f"Words similar to 'cat': {similar}")

    print("Training completed!")
