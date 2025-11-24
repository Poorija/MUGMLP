import os
import json
import pickle
import logging
import asyncio
from typing import List, Optional
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import networkx as nx
from rank_bm25 import BM25Okapi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    # WARNING: This implementation is a Singleton and currently shares the index across all users.
    # In a production multi-tenant environment, the index should be sharded by tenant/project ID.
    def __init__(self, index_path="rag_storage"):
        self.storage_path = index_path
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.encoder = None
        self.faiss_index = None
        self.documents = [] # List of {"content": str, "metadata": dict}
        self.bm25 = None
        self.graph = nx.Graph()

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

        self._load_state()

    def _initialize_models(self):
        if not self.encoder:
            logger.info("Loading RAG Embedding Model...")
            self.encoder = SentenceTransformer(self.embedding_model_name)
            if self.faiss_index is None:
                # Use CPU for FAISS for broad compatibility
                self.dimension = 384 # Dimension of all-MiniLM-L6-v2
                self.faiss_index = faiss.IndexFlatL2(self.dimension)

    def _save_state(self):
        # Save FAISS index
        if self.faiss_index:
            faiss.write_index(self.faiss_index, os.path.join(self.storage_path, "index.faiss"))

        # Save Metadata, Documents, Graph
        state = {
            "documents": self.documents,
            "graph": nx.node_link_data(self.graph)
        }
        with open(os.path.join(self.storage_path, "state.pkl"), "wb") as f:
            pickle.dump(state, f)

        # Save BM25 (it's not easily serializable, but we can re-init it or pickle it if recursion limit permits)
        # BM25Okapi is pickleable
        if self.bm25:
            with open(os.path.join(self.storage_path, "bm25.pkl"), "wb") as f:
                pickle.dump(self.bm25, f)

    def _load_state(self):
        try:
            # Load FAISS
            index_file = os.path.join(self.storage_path, "index.faiss")
            if os.path.exists(index_file):
                self.faiss_index = faiss.read_index(index_file)

            # Load Metadata
            state_file = os.path.join(self.storage_path, "state.pkl")
            if os.path.exists(state_file):
                with open(state_file, "rb") as f:
                    state = pickle.load(f)
                    self.documents = state.get("documents", [])
                    if "graph" in state:
                        self.graph = nx.node_link_graph(state["graph"])

            # Load BM25
            bm25_file = os.path.join(self.storage_path, "bm25.pkl")
            if os.path.exists(bm25_file):
                with open(bm25_file, "rb") as f:
                    self.bm25 = pickle.load(f)

        except Exception as e:
            logger.error(f"Failed to load RAG state: {e}")

    def ingest_document(self, file_path: str, doc_id: str):
        """Parses and ingests a document (PDF or Text)."""
        self._initialize_models()

        text_content = ""
        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()

        # Simple chunking (Split by paragraphs or roughly 500 chars)
        # For production, use langchain.RecursiveCharacterTextSplitter
        chunks = [c.strip() for c in text_content.split("\n\n") if len(c.strip()) > 50]

        if not chunks:
            # If split by paragraphs failed, split by fixed size
            chunks = [text_content[i:i+500] for i in range(0, len(text_content), 500)]

        # 1. Update Vectors
        embeddings = self.encoder.encode(chunks)
        self.faiss_index.add(np.array(embeddings).astype("float32"))

        # 2. Update Storage
        start_idx = len(self.documents)
        for i, chunk in enumerate(chunks):
            self.documents.append({
                "content": chunk,
                "source": doc_id,
                "chunk_id": start_idx + i
            })

        # 3. Update BM25
        tokenized_corpus = [doc["content"].split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 4. Update Graph (Naive Entity Extraction)
        # Extract capitalized words as "entities" and link adjacent ones
        for chunk in chunks:
            words = chunk.split()
            entities = [w.strip(".,") for w in words if w[0].isupper() and len(w) > 2]
            for i in range(len(entities) - 1):
                self.graph.add_edge(entities[i], entities[i+1], relation="adjacent")

        self._save_state()
        return len(chunks)

    def hybrid_search(self, query: str, k=5) -> List[dict]:
        self._initialize_models()

        # 1. Dense Search (FAISS)
        q_emb = self.encoder.encode([query])
        # Retrieve slightly more for fusion (k*2)
        D, I = self.faiss_index.search(np.array(q_emb).astype("float32"), k * 2)
        dense_hits = I[0] # List of indices

        # 2. Sparse Search (BM25)
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        # Get top k*2 indices
        sparse_hits = np.argsort(doc_scores)[::-1][:k * 2]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_score = {}

        def add_rank_score(hits):
            for rank, doc_idx in enumerate(hits):
                if doc_idx < 0 or doc_idx >= len(self.documents):
                    continue
                if doc_idx not in rrf_score:
                    rrf_score[doc_idx] = 0.0
                rrf_score[doc_idx] += 1.0 / (rank + 60) # 60 is standard constant

        add_rank_score(dense_hits)
        add_rank_score(sparse_hits)

        # Sort by RRF score
        sorted_docs = sorted(rrf_score.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_idx, score in sorted_docs[:k]:
            results.append(self.documents[doc_idx])

        return results

    def graph_retrieval(self, query: str, depth=2) -> List[str]:
        """Finds entities in query and retrieves graph neighbors."""
        query_entities = [w.strip(".,") for w in query.split() if w[0].isupper()]
        knowledge = []

        for entity in query_entities:
            if entity in self.graph:
                # Get neighbors
                neighbors = list(nx.single_source_shortest_path_length(self.graph, entity, cutoff=depth).keys())
                knowledge.append(f"Entity '{entity}' is related to: {', '.join(neighbors[:5])}")

        return knowledge

    def query(self, query_text: str) -> dict:
        """Main interface for RAG."""
        docs = self.hybrid_search(query_text)
        graph_context = self.graph_retrieval(query_text)

        context_str = "\n".join([d["content"] for d in docs])
        graph_str = "\n".join(graph_context)

        # In a real system, we would pass this to an LLM.
        # Here we return the constructed prompt/context.
        return {
            "retrieved_docs": docs,
            "graph_knowledge": graph_context,
            "constructed_context": f"Graph Knowledge:\n{graph_str}\n\nDocuments:\n{context_str}"
        }

# Singleton
rag_engine = RAGEngine()
