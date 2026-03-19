import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import csv

class LocalWordEmbedder:
    def __init__(self, vocab=None):
        self.vectorizer = CountVectorizer(vocabulary=vocab)

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()


def generate_dataset(N, embedder, texts=None):
    if texts is None:
        # Generate N random sentences
        words = ["cat", "dog", "fish", "bird", "apple", "car", "tree", "house", "book", "phone"]
        texts = [" ".join(np.random.choice(words, size=np.random.randint(3, 7))) for _ in range(N)]
    embeddings = embedder.fit_transform(texts)
    return embeddings, texts

# Example usage:
if __name__ == "__main__":
    N = 1000 # Number of embedding vectors
    embedder = LocalWordEmbedder()
    embeddings, sentences = generate_dataset(N, embedder)
    print("Sentences:", sentences)
    print("Embeddings:\n", embeddings)

    # Save to CSV
    with open("embeddings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["sentence"] + [f"dim_{i}" for i in range(embeddings.shape[1])])
        # Write rows
        for sent, emb in zip(sentences, embeddings):
            writer.writerow([sent] + list(emb))
