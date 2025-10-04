from langchain_core.documents import Document
class RAGService:
    def __init__(self, llm, embedding, vector_store) -> None:
        pass
    
    def chunk_code_file(self, content: str):
        pass
    
    def embed_chunks_and_store(self, chunks: list[Document]):
        pass
    
    def search_vector_store(self, query: str):
        pass
    
    def rerank(self, docs, query):
        pass     