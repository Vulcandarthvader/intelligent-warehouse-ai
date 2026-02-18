import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class WarehouseRAG:
    def __init__(self, docs_path="rag/documents"):
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.documents = []
        self.doc_names = []

        print("Loading documents...")

        for filename in os.listdir(docs_path):
            filepath = os.path.join(docs_path, filename)

            with open(filepath, "r") as f:
                text = f.read()

                self.documents.append(text)
                self.doc_names.append(filename)

        print("Embedding documents...")
        self.embeddings = self.model.encode(self.documents)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)

        self.index.add(np.array(self.embeddings))
        print("FAISS index created.")

    def query(self, user_query, top_k=1):
        query_embedding = self.model.encode([user_query])
        distances, indices = self.index.search(
            np.array(query_embedding), top_k
        )

        retrieved_docs = []

        for idx in indices[0]:
            retrieved_docs.append(
                {
                    "document": self.doc_names[idx],
                    "content": self.documents[idx],
                }
            )

        return retrieved_docs


if __name__ == "__main__":
    rag = WarehouseRAG()

    while True:
        query = input("\nEnter your query (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = rag.query(query)

        print("\nTop Retrieved Documents:")
        for r in results:
            print("\n---", r["document"], "---")
            print(r["content"])

