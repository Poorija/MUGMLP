import os
import json
import pickle
import logging
import asyncio
from typing import List, Optional
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import networkx as nx
from rank_bm25 import BM25Okapi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, index_path="rag_storage"):
        self.storage_path = index_path
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.encoder = None
        self.reranker = None
        # Stores: {scope_id: {'faiss': index, 'docs': [], 'bm25': obj, 'graph': nx.Graph}}
        self.stores = {}
        self.dimension = 384

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

        self._load_state()

    def _initialize_models(self):
        if not self.encoder:
            logger.info("Loading RAG Embedding Model...")
            self.encoder = SentenceTransformer(self.embedding_model_name)
        if not self.reranker:
            logger.info("Loading RAG Reranker Model...")
            self.reranker = CrossEncoder(self.reranker_model_name)

    def _get_store(self, scope_id: str):
        if scope_id not in self.stores:
            self.stores[scope_id] = {
                'faiss': faiss.IndexFlatL2(self.dimension),
                'docs': [],
                'bm25': None,
                'graph': nx.Graph()
            }
        return self.stores[scope_id]

    def _save_state(self):
        # We save each scope in a subdirectory
        for scope_id, store in self.stores.items():
            scope_dir = os.path.join(self.storage_path, str(scope_id))
            if not os.path.exists(scope_dir):
                os.makedirs(scope_dir)

            # Save FAISS
            if store['faiss']:
                faiss.write_index(store['faiss'], os.path.join(scope_dir, "index.faiss"))

            # Save Metadata
            state = {
                "documents": store['docs'],
                "graph": nx.node_link_data(store['graph'])
            }
            with open(os.path.join(scope_dir, "state.pkl"), "wb") as f:
                pickle.dump(state, f)

            # Save BM25
            if store['bm25']:
                with open(os.path.join(scope_dir, "bm25.pkl"), "wb") as f:
                    pickle.dump(store['bm25'], f)

    def _load_state(self):
        # Walk directories in storage_path
        if not os.path.exists(self.storage_path):
            return

        for scope_id in os.listdir(self.storage_path):
            scope_dir = os.path.join(self.storage_path, scope_id)
            if not os.path.isdir(scope_dir):
                continue

            try:
                store = {
                    'faiss': faiss.IndexFlatL2(self.dimension),
                    'docs': [],
                    'bm25': None,
                    'graph': nx.Graph()
                }

                # Load FAISS
                index_file = os.path.join(scope_dir, "index.faiss")
                if os.path.exists(index_file):
                    store['faiss'] = faiss.read_index(index_file)

                # Load Metadata
                state_file = os.path.join(scope_dir, "state.pkl")
                if os.path.exists(state_file):
                    with open(state_file, "rb") as f:
                        state = pickle.load(f)
                        store['docs'] = state.get("documents", [])
                        if "graph" in state:
                            store['graph'] = nx.node_link_graph(state["graph"])

                # Load BM25
                bm25_file = os.path.join(scope_dir, "bm25.pkl")
                if os.path.exists(bm25_file):
                    with open(bm25_file, "rb") as f:
                        store['bm25'] = pickle.load(f)

                self.stores[scope_id] = store
            except Exception as e:
                logger.error(f"Failed to load RAG state for scope {scope_id}: {e}")

    def ingest_document(self, file_path: str, doc_id: str, scope_id: str):
        """Parses and ingests a document (PDF or Text)."""
        self._initialize_models()
        store = self._get_store(scope_id)

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
        store['faiss'].add(np.array(embeddings).astype("float32"))

        # 2. Update Storage
        start_idx = len(store['docs'])
        for i, chunk in enumerate(chunks):
            store['docs'].append({
                "content": chunk,
                "source": doc_id,
                "chunk_id": start_idx + i
            })

        # 3. Update BM25
        tokenized_corpus = [doc["content"].split(" ") for doc in store['docs']]
        store['bm25'] = BM25Okapi(tokenized_corpus)

        # 4. Update Graph (Naive Entity Extraction)
        # Extract capitalized words as "entities" and link adjacent ones
        for chunk in chunks:
            words = chunk.split()
            entities = [w.strip(".,") for w in words if w[0].isupper() and len(w) > 2]
            for i in range(len(entities) - 1):
                store['graph'].add_edge(entities[i], entities[i+1], relation="adjacent")

        self._save_state()
        return len(chunks)

    def hybrid_search(self, query: str, scope_id: str, k=5) -> List[dict]:
        self._initialize_models()
        store = self._get_store(scope_id)

        if not store['docs']:
            return []

        # 1. Dense Search (FAISS)
        q_emb = self.encoder.encode([query])
        # Retrieve extended candidate pool for reranking (k*3)
        pool_size = min(k * 3, len(store['docs']))
        D, I = store['faiss'].search(np.array(q_emb).astype("float32"), pool_size)
        dense_hits = I[0] # List of indices

        # 2. Sparse Search (BM25)
        sparse_hits = []
        if store['bm25']:
            tokenized_query = query.split(" ")
            doc_scores = store['bm25'].get_scores(tokenized_query)
            # Get top k*3 indices
            sparse_hits = np.argsort(doc_scores)[::-1][:pool_size]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_score = {}

        def add_rank_score(hits):
            for rank, doc_idx in enumerate(hits):
                if doc_idx < 0 or doc_idx >= len(store['docs']):
                    continue
                if doc_idx not in rrf_score:
                    rrf_score[doc_idx] = 0.0
                rrf_score[doc_idx] += 1.0 / (rank + 60) # 60 is standard constant

        add_rank_score(dense_hits)
        add_rank_score(sparse_hits)

        # Sort by RRF score (Top candidates for reranking)
        sorted_docs = sorted(rrf_score.items(), key=lambda item: item[1], reverse=True)

        # 4. Cross-Encoder Reranking
        # Take top 2*k from fusion to rerank
        rerank_candidates_indices = [idx for idx, score in sorted_docs[:k*2]]
        rerank_inputs = []
        for idx in rerank_candidates_indices:
            rerank_inputs.append([query, store['docs'][idx]['content']])

        if rerank_inputs:
            scores = self.reranker.predict(rerank_inputs)
            # Combine index and score
            scored_candidates = list(zip(rerank_candidates_indices, scores))
            # Sort by Cross-Encoder score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            final_indices = [idx for idx, score in scored_candidates[:k]]
        else:
            final_indices = rerank_candidates_indices[:k]

        results = []
        for idx in final_indices:
            results.append(store['docs'][idx])

        return results

    def graph_retrieval(self, query: str, scope_id: str, depth=2) -> List[str]:
        """Finds entities in query and retrieves graph neighbors."""
        store = self._get_store(scope_id)
        query_entities = [w.strip(".,") for w in query.split() if w[0].isupper()]
        knowledge = []

        for entity in query_entities:
            if entity in store['graph']:
                # Get neighbors
                neighbors = list(nx.single_source_shortest_path_length(store['graph'], entity, cutoff=depth).keys())
                knowledge.append(f"Entity '{entity}' is related to: {', '.join(neighbors[:5])}")

        return knowledge

    def query(self, query_text: str, scope_id: str) -> dict:
        """Main interface for RAG."""
        docs = self.hybrid_search(query_text, scope_id)
        graph_context = self.graph_retrieval(query_text, scope_id)

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
