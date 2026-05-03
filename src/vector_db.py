import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            self.documents = json.load(f)
        
        # Prepare text for embedding (Hybrid approach: Semantic + Keywords)
        texts = []
        for doc in self.documents:
            # Smart expansion using title + description + keywords
            keywords_str = " ".join(doc.get("keywords", []))
            text = f"Title: {doc['title']}. Description: {doc['description']}. Keywords: {keywords_str}"
            texts.append(text)
        
        # Create embeddings
        embeddings = self.model.encode(texts)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) # Using Inner Product for Cosine Similarity (requires normalized vectors)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"Loaded {len(self.documents)} documents into Vector DB.")

    def search(self, query, top_k=10):
        # Embed the query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['confidence'] = float(distances[0][i])
                results.append(doc)
                
        return results

if __name__ == "__main__":
    db = VectorDB()
    db.load_data("../data/standards.json")
    print(db.search("cement for construction"))
