import os
import uuid
import hashlib
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


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
        embedding_dim: int = 768,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.collection_name = f"fractal_{project_name.replace('/', '_').replace(' ', '_')}"

        self.qdrant_client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url, https=True, prefer_grpc=False, timeout=30)

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
    def chunk_code_file(self, content: str, source_file: str) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=150,
            separators=["\n\n", "\nclass ", "\ndef ", "\n", " ", ""]
        )
        chunks = splitter.split_text(content)
        
        # Create documents with unique IDs
        docs = []
        for i, chunk in enumerate(chunks):
            doc = Document(page_content=chunk)
            # We'll set the ID in the calling method where we have source_file info
            docs.append(doc)
        
        return docs
    

    
    # ------------------------------------------------------------------
    # Initial indexing
    # ------------------------------------------------------------------
    async def index_codebase(self, root_dir: str):
        """Index the entire codebase for the first time"""
        root = Path(root_dir)
        code_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.php', '.rb']
        
        all_docs = []
        indexed_files = []
        
        for file in root.rglob('*'):
            if file.is_file() and file.suffix.lower() in code_extensions:
                if any(skip in file.parts for skip in ['.git', 'node_modules', '__pycache__', '.venv', 'venv']):
                    continue
                    
                try:
                    content = file.read_text(encoding="utf-8")
                    docs = self.chunk_code_file(content, str(file.relative_to(root)))
                    
                    for i, doc in enumerate(docs):
                        doc.metadata["source_file"] = str(file.relative_to(root))
                        doc.metadata["file_hash"] = self._compute_hash(content)
                        doc.metadata["chunk_id"] = self._generate_chunk_id(
                            str(file.relative_to(root)), i, doc.page_content
                        )
                    
                    all_docs.extend(docs)
                    indexed_files.append(str(file.relative_to(root)))
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    continue
        
        if all_docs:
            print(f"Indexing {len(indexed_files)} files with {len(all_docs)} chunks...")
            await self.embed_chunks_and_store(all_docs)
            
            # Save index
            index_data = {f: self._compute_hash(Path(root, f).read_text(encoding="utf-8")) for f in indexed_files if (Path(root) / f).exists()}

            self._save_index(root, index_data)
            
            print(f"✓ Indexed {len(indexed_files)} files successfully")
        else:
            print("No code files found to index")
    
    # ------------------------------------------------------------------
    # Ingest: embed & store documents
    # ------------------------------------------------------------------
    async def embed_chunks_and_store(self, docs: List[Document]):
        """Embed + store documents in Qdrant + initialize BM25 retriever."""
        if not docs:
            raise ValueError("No documents to embed/store.")
        
        await self.vector_store.aadd_documents(docs)
        print(f"Stored {len(docs)} chunks in Qdrant collection '{self.collection_name}'")

        # Get all documents for BM25
        all_docs = self._get_all_documents()
        if all_docs:
            self.sparse_retriever = BM25Retriever.from_documents(all_docs)
            # Optimize dense retriever for speed
            dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

            self.hybrid_retriever = EnsembleRetriever(
                retrievers=[dense_retriever, self.sparse_retriever],
                weights=[0.7, 0.3],  # Favor dense retrieval for better semantic matching
            )
            print("Hybrid retriever initialized")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """Hybrid retrieval: dense + sparse. Optimized for speed."""
        if not self.hybrid_retriever:
            # Try to initialize if not already done
            all_docs = self._get_all_documents()
            if all_docs:
                self.sparse_retriever = BM25Retriever.from_documents(all_docs)
                # Reduce k for faster dense retrieval
                dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
                self.hybrid_retriever = EnsembleRetriever(
                    retrievers=[dense_retriever, self.sparse_retriever],
                    weights=[0.7, 0.3],  # Favor dense retrieval for better semantic matching
                )
            else:
                raise RuntimeError("No documents in collection. Call index_codebase() first.")
        
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

    def _generate_chunk_id(self, source_file: str, chunk_index: int, content: str) -> str:
        """Generate unique ID for a chunk"""
        # Create a unique ID based on file path, chunk index, and content hash
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        return f"{source_file.replace('/', '_')}_{chunk_index}_{content_hash}"

    def _compute_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _load_index(self, root: Path) -> dict:
        index_file = root / ".fractal_index.json"
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _save_index(self, root: Path, data: dict):
        with open(root / ".fractal_index.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2) 

    def _get_all_documents(self) -> List[Document]:
        """Retrieve all documents from the vector store"""
        try:
            # Use a broad query to get many documents
            results = self.vector_store.similarity_search("", k=10000)
            return results
        except Exception:
            return []

    def detect_changes(self, root_dir: str) -> list[Path]:
        """Compare hashes to detect modified or new files."""
        root = Path(root_dir)
        current_index = self._load_index(root)
        updated_index = {}
        changed_files = []

        code_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.php', '.rb']
        
        for file in root.rglob('*'):
            if file.is_file() and file.suffix.lower() in code_extensions:    
                
                if any(skip in file.parts for skip in ['.git', 'node_modules', '__pycache__', '.venv', 'venv']):
                    continue
                    
                try:
                    content = file.read_text(encoding="utf-8")
                    file_hash = self._compute_hash(content)
                    rel_path = str(file.relative_to(root))
                    updated_index[rel_path] = file_hash

                    if rel_path not in current_index or current_index[rel_path] != file_hash:
                        changed_files.append(file)

                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    continue

        self._save_index(root, updated_index)
        return changed_files


    def _delete_chunks_by_source_files(self, source_files: List[str]) -> int:
        """Delete chunks from vector store by source file paths"""
        try:
            deleted_count = 0
            
            for source_file in source_files:
                try:                    
                    metadata_filter = Filter(
                        must=[
                            FieldCondition(
                                key="source_file",
                                match=MatchValue(value=source_file)
                            )
                        ]
                    )
                    
                    self.qdrant_client.delete(
                        collection_name=self.collection_name,
                        points_selector=metadata_filter
                    )
                    
                    deleted_count += 1
                    print(f"Deleted chunks from file: {source_file}")
                    
                except Exception as e:
                    print(f"Error deleting chunks for {source_file}: {e}")
                    continue
            
            if deleted_count > 0:
                print(f"Successfully deleted chunks from {deleted_count} modified files")
            
            return deleted_count
            
        except Exception as e:
            print(f"Error in deletion process: {e}")
            return 0

    def reembed_changed_files(self, root_dir: str):
        """Re-embed only changed files - properly delete old chunks first"""
        changed_files = self.detect_changes(root_dir)
        if not changed_files:
            print("No code changes detected.")
            return

        print(f"Re-embedding {len(changed_files)} modified files...")
        root = Path(root_dir)
        
        changed_file_paths = [str(file.relative_to(root)) for file in changed_files]
        deleted_count = self._delete_chunks_by_source_files(changed_file_paths)
        if deleted_count > 0:
            print(f"Deleted {deleted_count} old chunks from modified files")
        
        all_new_docs = []
        for file in changed_files:
            try:
                content = file.read_text(encoding="utf-8")
                docs = self.chunk_code_file(content, str(file.relative_to(root)))
                
                for i, doc in enumerate(docs):
                    doc.metadata["source_file"] = str(file.relative_to(root))
                    doc.metadata["file_hash"] = self._compute_hash(content)
                    doc.metadata["chunk_id"] = self._generate_chunk_id(
                        str(file.relative_to(root)), i, doc.page_content
                    )
                
                all_new_docs.extend(docs)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
        
        if all_new_docs:
            self.vector_store.add_documents(all_new_docs)
            self._rebuild_retrievers()
            print(f"✓ Re-embedded {len(changed_files)} files ({len(all_new_docs)} new chunks)")

    def _rebuild_retrievers(self):
        """Rebuild the hybrid retriever with all documents"""
        try:
            all_docs = self._get_all_documents()
            
            if all_docs:
                self.sparse_retriever = BM25Retriever.from_documents(all_docs)
                # Optimize for speed
                dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
                
                self.hybrid_retriever = EnsembleRetriever(
                    retrievers=[dense_retriever, self.sparse_retriever],
                    weights=[0.7, 0.3],  # Favor dense retrieval for better semantic matching
                )
                
        except Exception as e:
            print(f"Error rebuilding retrievers: {e}")

    def cleanup_deleted_files(self, root_dir: str):
        """Remove chunks from files that no longer exist"""
        try:
            root = Path(root_dir)
            all_docs = self._get_all_documents()
            
            files_to_cleanup = []
            for doc in all_docs:
                source_file = doc.metadata.get("source_file")
                if source_file:
                    file_path = root / source_file
                    if not file_path.exists() and source_file not in files_to_cleanup:
                        files_to_cleanup.append(source_file)
            
            deleted_count = 0
            for source_file in files_to_cleanup:
                try:
                    metadata_filter = Filter(
                        must=[
                            FieldCondition(
                                key="source_file",
                                match=MatchValue(value=source_file)
                            )
                        ]
                    )
                    
                    self.qdrant_client.delete(
                        collection_name=self.collection_name,
                        points_selector=metadata_filter
                    )
                    
                    deleted_count += 1
                    print(f"Cleaned up chunks from deleted file: {source_file}")
                    
                except Exception as e:
                    print(f"Error cleaning up {source_file}: {e}")
                    continue
            
            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} orphaned files")
            
            return deleted_count
            
        except Exception as e:
            print(f"Error cleaning up deleted files: {e}")
            return 0