import os
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
        collection_name: str = "code_chunks",
        embedding_dim: int = 3072,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.collection_name = collection_name

        self.qdrant_client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url)

        # Create collection if not exists
        existing = [col.name for col in self.qdrant_client.get_collections().collections]
        if collection_name not in existing:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
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
            collection_name=collection_name,
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


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_EMBEDDING_API_KEY")

    llm = init_chat_model(
        "gpt-4o",
        model_provider="openai",
        temperature=0.2,
        api_key=openai_api_key,
    )
    if not google_api_key:
        raise ValueError("no google api")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=SecretStr(google_api_key)
    )

    rag = RAGService(llm, embedding_model)

    code_content = """
    def add(a, b):
        return a + b

    def multiply(a, b):
        return a * b

    def divide(a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    """

    chunks = rag.chunk_code_file(code_content)
    rag.embed_chunks_and_store(chunks)

    results = rag.search("function that multiplies numbers")
    print("\n🔍 Retrieved documents:")
    for d in results:
        print("-", d.page_content[:100], "...")

    summary = rag.rerank(results, "function that multiplies numbers")
    print("\n🧠 Reranked summary:\n", summary)