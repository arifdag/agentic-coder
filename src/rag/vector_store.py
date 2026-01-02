"""
Vector Store for Error Solutions

ChromaDB-based vector store for storing and retrieving similar error solutions.
Used for Few-Shot learning by providing relevant examples to the LLM.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


class ErrorSolutionStore:
    """
    Vector store for error-solution pairs using ChromaDB.
    
    Stores error messages with their solutions and enables semantic
    similarity search to find relevant examples for new errors.
    
    Example:
        store = ErrorSolutionStore()
        
        # Add an error solution
        store.add_error_solution(
            error_type="TypeError",
            error_message="unsupported operand type(s) for +: 'int' and 'str'",
            language="python",
            solution="Use str() to convert the integer before concatenation",
            code_fix="result = str(number) + text"
        )
        
        # Search for similar errors
        results = store.search_similar_errors(
            error_message="cannot add int and string",
            language="python",
            top_k=3
        )
    """
    
    COLLECTION_NAME = "error_solutions"
    
    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the error solution store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Sentence-transformers model for embeddings
        """
        self.persist_directory = persist_directory or Path(__file__).parent.parent.parent / "data" / "chroma_db"
        
        # Initialize ChromaDB client (v0.4+)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use sentence-transformers for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"description": "Error messages with solutions for Few-Shot learning"}
        )
    
    def add_error_solution(
        self,
        error_type: str,
        error_message: str,
        language: str,
        solution: str,
        code_fix: Optional[str] = None,
        traceback: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Add an error-solution pair to the store.
        
        Args:
            error_type: Type of error (e.g., TypeError, SyntaxError)
            error_message: The error message
            language: Programming language
            solution: Natural language explanation of the solution
            code_fix: Optional code snippet showing the fix
            traceback: Optional full traceback
            tags: Optional tags for categorization
            metadata: Optional additional metadata
            
        Returns:
            ID of the added document
        """
        doc_id = str(uuid.uuid4())
        
        # Create document text for embedding
        document_text = self._create_document_text(
            error_type=error_type,
            error_message=error_message,
            language=language,
            traceback=traceback,
        )
        
        # Prepare metadata
        doc_metadata = {
            "error_type": error_type,
            "language": language,
            "solution": solution,
            "created_at": datetime.now().isoformat(),
        }
        
        if code_fix:
            doc_metadata["code_fix"] = code_fix
        if tags:
            doc_metadata["tags"] = ",".join(tags)
        if metadata:
            doc_metadata.update(metadata)
        
        # Add to collection
        self.collection.add(
            ids=[doc_id],
            documents=[document_text],
            metadatas=[doc_metadata],
        )
        
        return doc_id
    
    def search_similar_errors(
        self,
        error_message: str,
        language: Optional[str] = None,
        error_type: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for similar error solutions.
        
        Args:
            error_message: The error message to search for
            language: Optional filter by programming language
            error_type: Optional filter by error type
            top_k: Number of results to return
            
        Returns:
            List of similar error solutions with metadata
        """
        # Build query text
        query_text = error_message
        if error_type:
            query_text = f"{error_type}: {query_text}"
        if language:
            query_text = f"[{language}] {query_text}"
        
        # Build where filter
        where_filter = None
        if language or error_type:
            conditions = []
            if language:
                conditions.append({"language": language})
            if error_type:
                conditions.append({"error_type": error_type})
            
            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}
        
        # Query collection
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter,
        )
        
        # Format results
        solutions = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                solution = {
                    "id": doc_id,
                    "document": results["documents"][0][i] if results["documents"] else None,
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                }
                solutions.append(solution)
        
        return solutions
    
    def delete_error_solution(self, doc_id: str) -> bool:
        """
        Delete an error solution by ID.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False
    
    def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        return {
            "name": self.COLLECTION_NAME,
            "count": self.collection.count(),
            "persist_directory": str(self.persist_directory),
        }
    
    def bulk_add_error_solutions(self, solutions: list[dict[str, Any]]) -> list[str]:
        """
        Add multiple error solutions at once.
        
        Args:
            solutions: List of error solution dictionaries
            
        Returns:
            List of added document IDs
        """
        ids = []
        documents = []
        metadatas = []
        
        for sol in solutions:
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            
            documents.append(self._create_document_text(
                error_type=sol.get("error_type", "Unknown"),
                error_message=sol.get("error_message", ""),
                language=sol.get("language", "unknown"),
                traceback=sol.get("traceback"),
            ))
            
            metadata = {
                "error_type": sol.get("error_type", "Unknown"),
                "language": sol.get("language", "unknown"),
                "solution": sol.get("solution", ""),
                "created_at": datetime.now().isoformat(),
            }
            if sol.get("code_fix"):
                metadata["code_fix"] = sol["code_fix"]
            if sol.get("tags"):
                metadata["tags"] = ",".join(sol["tags"])
            
            metadatas.append(metadata)
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        
        return ids
    
    def _create_document_text(
        self,
        error_type: str,
        error_message: str,
        language: str,
        traceback: Optional[str] = None,
    ) -> str:
        """Create document text for embedding."""
        parts = [
            f"Language: {language}",
            f"Error Type: {error_type}",
            f"Error Message: {error_message}",
        ]
        if traceback:
            # Include first few lines of traceback
            traceback_lines = traceback.strip().split("\n")[:5]
            parts.append(f"Traceback: {' | '.join(traceback_lines)}")
        
        return "\n".join(parts)
