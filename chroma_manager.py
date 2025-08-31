import uuid
import chromadb
from chromadb.utils import embedding_functions
import config

class ChromaMemoryManager:
    def __init__(self, collection_name: str = "chat_history"):
        client = chromadb.PersistentClient(path=config.CHROMADB_DIR)
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = client.get_or_create_collection(name=collection_name, embedding_function=embed_fn)

    def add_message(self, role: str, content: str, session_id: str):
        self.collection.add(
            documents=[content],
            metadatas={"role": role, "session_id": session_id},
            ids=[str(uuid.uuid4())]
        )

    def retrieve_relevant_memories(self, query: str, session_id: str, k: int = 5, allowed_speaker_types: list = None) -> list[str]:
        where_clause = {"session_id": session_id}
        if allowed_speaker_types:
            where_clause["role"] = {"$in": allowed_speaker_types}

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_clause,
            include=["documents", "metadatas"]
        )
        
        recalled_memories = []
        if results and results['documents']:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                recalled_memories.append(f"{meta['role'].upper()}: {doc}")
        
        return recalled_memories