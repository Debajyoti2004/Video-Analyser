import chromadb
from chromadb.utils import embedding_functions
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
import config

class ChromaMemoryManager:
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.console = Console()
        
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name="video_analysis_memory",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def add_message(self, speaker_type: str, content: str, session_id: str):
        doc_count = self.collection.count()
        doc_id = f"{session_id}_{doc_count + 1}"

        self.collection.add(
            documents=[content],
            metadatas=[{"role": speaker_type, "session_id": session_id}],
            ids=[doc_id]
        )

    def retrieve_relevant_memories(
        self, 
        query: str, 
        session_id: str, 
        n_results: int = 5,
        allowed_speaker_types: Optional[List[str]] = None
    ) -> List[str]:
        try:
            where_filter = {
                "$and": [
                    {"session_id": {"$eq": session_id}},
                    {"role": {"$in": allowed_speaker_types or ["owner", "agent"]}}
                ]
            }

            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            return results.get('documents', [[]])[0]
        
        except Exception as e:
            error_message = f"[bold]Error querying ChromaDB:[/bold]\n\n{str(e)}"
            self.console.print(
                Panel(
                    error_message, 
                    title="[bold red]Memory System Error[/]", 
                    border_style="red"
                )
            )
            return []