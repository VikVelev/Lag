import numpy as np
from typing import List, Optional
import gensim.downloader as api
from gensim.models import KeyedVectors
import csv
import os
import gensim.downloader as api
import csv
import os


class PretrainedEmbedder:
    """
    A wrapper around Gensim to load and use massive pretrained Word2Vec/GloVe models.
    """

    def __init__(self, model_name: str = "glove-wiki-gigaword-50"):
        """
        Initializes the embedder and downloads/loads the pretrained model.

        Popular model_name options via gensim downloader:
        - "glove-wiki-gigaword-50" (Fast, 50 dimensions, 400k words, ~65MB)
        - "word2vec-google-news-300" (Massive, 300 dimensions, 3M words, ~1.6GB)
        """
        self.model_name = model_name
        self.model: Optional[KeyedVectors] = None
        self.vector_size = 0
        self.vocab_size = 0

    def load_pretrained(self) -> None:
        """Downloads (if necessary) and loads the pretrained model into memory."""
        print(f"Loading pretrained model '{self.model_name}'...")
        print("(This may take a while if it's downloading for the first time...)")

        # This automatically fetches the model from Gensim's remote servers
        # and caches it locally on your machine.
        self.model = api.load(self.model_name)

        self.vector_size = self.model.vector_size
        self.vocab_size = len(self.model.key_to_index)

        print(f"Successfully loaded '{self.model_name}'!")
        print(f"Vocabulary Size: {self.vocab_size:,} words")
        print(f"Vector Dimensions: {self.vector_size}")

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get the numpy vector for a specific word."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_pretrained() first.")

        # Gensim is case-sensitive, and most models are entirely lowercase
        word = word.lower()

        if word in self.model:
            return self.model[word].copy()
        return None

    def get_similar_words(self, word: str, topn: int = 10) -> List[tuple]:
        """Get the most similar words using Gensim's highly optimized backend."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_pretrained() first.")

        word = word.lower()

        if word not in self.model:
            print(f"Word '{word}' not found in the pretrained vocabulary.")
            return []

        # Gensim handles the cosine similarity math internally (and very fast)
        return self.model.most_similar(word, topn=topn)

    def solve_analogy(
        self, positive: List[str], negative: List[str], topn: int = 3
    ) -> List[tuple]:
        """
        The classic Word2Vec party trick!
        e.g., King - Man + Woman = Queen
        (positive=['king', 'woman'], negative=['man'])
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        positive = [w.lower() for w in positive]
        negative = [w.lower() for w in negative]

        try:
            return self.model.most_similar(
                positive=positive, negative=negative, topn=topn
            )
        except KeyError as e:
            print(f"One of the words in the analogy is missing from the vocab: {e}")
            return []

    def export_subset_to_csv(
        self, output_path: str, words_to_export: List[str]
    ) -> None:
        """
        Exports a specific subset of words to a CSV so you can still use
        your PCA visualization script! (Exporting the whole 3M word vocab would crash it).
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        exported_count = 0
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["word"] + [f"dim_{i}" for i in range(self.vector_size)])

            for word in words_to_export:
                word = word.lower()
                if word in self.model:
                    writer.writerow([word] + self.model[word].tolist())
                    exported_count += 1

        print(f"Exported {exported_count} vectors to {output_path}")


# --- Testing the Pretrained Model ---
def test():
    # Using the 50-dimensional GloVe model (trained on Wikipedia) for a fast download.
    # glove-wiki-gigaword-50
    # Change to "word2vec-google-news-300" if you want the absolute highest quality.
    embedder = PretrainedEmbedder(model_name="word2vec-google-news-300")

    # 1. Load the model (will trigger a download on the first run)
    embedder.load_pretrained()

    # 2. Test standard similarity
    for test_word in ["planet", "gravity", "star", "trade", "astronomers"]:
        print(f"\nWords similar to '{test_word}':")
        for w, score in embedder.get_similar_words(test_word, topn=5):
            print(f"  - {w} ({score:.4f})")

    # 3. Test the famous vector math analogy
    print("\nSolving Analogy: King - Man + Woman = ?")
    analogy = embedder.solve_analogy(
        positive=["king", "woman"], negative=["man"], topn=1
    )
    print(f"Result: {analogy[0][0]} ({analogy[0][1]:.4f})")



def export_in_chunks(model_name: str = "word2vec-google-news-300", 
                     output_dir: str = "exports/google_news_chunks", 
                     chunk_size: int = 100000):
    
    print(f"Loading '{model_name}'... (Grab a coffee, this takes a minute or two)")
    model = api.load(model_name)
    
    # Create the dedicated folder for our chunks
    os.makedirs(output_dir, exist_ok=True)
        
    vector_size = model.vector_size
    total_words = len(model.index_to_key)
    
    print(f"Model loaded. Total vocabulary: {total_words:,} words")
    print(f"Exporting in chunks of {chunk_size:,}...")

    # Calculate exactly how many files we need to make
    num_chunks = (total_words + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_words)
        
        # Name the file neatly: e.g., chunk_001_0_to_100000.csv
        filename = f"chunk_{chunk_idx + 1:03d}_{start_idx}_to_{end_idx}.csv"
        filepath = os.path.join(output_dir, filename)
        
        print(f"Writing {filename}...")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write the header
            header = ['word'] + [f'dim_{i}' for i in range(vector_size)]
            writer.writerow(header)
            
            # Write the rows specifically for this chunk
            for i in range(start_idx, end_idx):
                word = model.index_to_key[i]
                vector = model[word]
                writer.writerow([word] + vector.tolist())
                
    print("\nSuccess! All 3 million words have been safely exported into bite-sized CSVs.")

if __name__ == "__main__":
    # Run the chunker!
    export_in_chunks(
        model_name="word2vec-google-news-300", 
        output_dir="exports/google_news_chunks", 
        chunk_size=100000 
    )
