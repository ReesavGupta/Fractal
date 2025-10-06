import os
import uuid
import hashlib
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from google import genai
from google.genai import types
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

class RAGService:
    """
    Hybrid RAG service using:
      • Dense embeddings (Gemini → Qdrant)
      • Sparse BM25 keyword retriever
      • LLM-based reranker (OpenAI GPT-4o)
    """
    def __init__(
        self,
        llm,
        embedding_model,
        project_name: str,
        embedding_dim: int = 3072,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.collection_name = f"fractal_{project_name}_{uuid.uuid4().hex[:6]}" # i think what i can do is rather than having fractal_project_name_uuid i can have fractal_project_name where project_name will be the path to the current working dir

        self.qdrant_client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url)

        # Create collection if not exists
        existing = [col.name for col in self.qdrant_client.get_collections().collections]
        if self.collection_name not in existing:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=embedding_dim, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )

        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

        self.sparse_retriever = None
        self.hybrid_retriever = None

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    def chunk_code_file(self, content: str) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(content)
        return [Document(page_content=chunk) for chunk in chunks]

    # ------------------------------------------------------------------
    # Ingest: embed & store documents
    # ------------------------------------------------------------------
    def embed_chunks_and_store(self, docs: List[Document]):
        """Embed + store documents in Qdrant + initialize BM25 retriever."""
        if not docs:
            raise ValueError("No documents to embed/store.")
        
        self.vector_store.add_documents(docs)
        print(f"Stored {len(docs)} documents in Qdrant collection '{self.collection_name}'")

        self.sparse_retriever = BM25Retriever.from_documents(docs)

        dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, self.sparse_retriever],
            weights=[0.6, 0.4],
        )

        print("Hybrid retriever (dense + sparse) initialized.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """Hybrid retrieval: dense + sparse."""
        if not self.hybrid_retriever:
            raise RuntimeError("Hybrid retriever not initialized. Call embed_chunks_and_store() first.")
        results = self.hybrid_retriever.invoke(query)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Rerank with LLM
    # ------------------------------------------------------------------
    def rerank(self, docs: List[Document], query: str) -> str:
        """Use LLM to rerank and summarize retrieved documents."""
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""
            You are an expert assistant. Given the query below, rank and summarize the most relevant parts from the retrieved context.

            Query:
            {query}

            Context:
            {context}

            Summarize concisely and prioritize relevant parts.
        """
        response = self.llm.invoke(prompt)
        return response.content

###########################################################################################################################################################
###########################################################################################################################################################
#helper functions for checking the diffs and when to reembed and what should be reembeded
###########################################################################################################################################################
###########################################################################################################################################################
    def _compute_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _load_index(self, root: Path) -> dict:
        index_file = root / ".fractal_index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_index(self, root: Path, data: dict):
        with open(root / ".fractal_index.json", "w") as f:
            json.dump(data, f, indent=2) 

    def detect_changes(self, root_dir: str) -> list[Path]:
        """Compare hashes to detect modified or new files."""
        root = Path(root_dir)
        current_index = self._load_index(root)
        updated_index = {}
        changed_files = []

        code_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.php', '.rb']
        
        for file in root.rglob('*'):
            if file.is_file() and file.suffix.lower() in code_extensions:
                try:
                    content = file.read_text(encoding="utf-8")
                    file_hash = self._compute_hash(content)
                    updated_index[file.as_posix()] = file_hash

                    if file.as_posix() not in current_index or current_index[file.as_posix()] != file_hash:
                        changed_files.append(file)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    continue

        self._save_index(root, updated_index)
        return changed_files

    def reembed_changed_files(self, root_dir: str):
        changed_files = self.detect_changes(root_dir)
        if not changed_files:
            print("No code changes detected.")
            return

        print(f"Re-embedding {len(changed_files)} modified files...")

        all_new_docs = []

        for file in changed_files:
            try:
                content = file.read_text(encoding="utf-8")
                docs = self.chunk_code_file(content)

                for doc in docs:
                    doc.metadata["source_file"] = str(file)
                    doc.metadata["file_hash"] = self._compute_hash(content)
                
                all_new_docs.extend(docs)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
        
        if all_new_docs:
            self.vector_store.add_documents(all_new_docs)

            self._rebuild_retrievers()

            print(f"Re-embedding complete. Added {len(all_new_docs)} new chunks.")

    def _rebuild_retrievers(self):
        """Rebuild the hybrid retriever efficiently with all documents"""
        try:
            all_docs = self.vector_store.similarity_search("", k=10000)
            
            if all_docs:
                self.sparse_retriever = BM25Retriever.from_documents(all_docs)
                
                dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
                
                self.hybrid_retriever = EnsembleRetriever(
                    retrievers=[dense_retriever, self.sparse_retriever],
                    weights=[0.6, 0.4],
                )
                
                print("Hybrid retriever rebuilt successfully.")
        except Exception as e:
            print(f"Error rebuilding retrievers: {e}")