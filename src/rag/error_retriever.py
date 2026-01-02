"""
Error Retriever for Few-Shot Learning

Retrieves similar error solutions from the vector store and formats
them as few-shot examples for LLM prompts.
"""

from dataclasses import dataclass
from typing import Optional

from .vector_store import ErrorSolutionStore
from ..core.schemas import ErrorLog, SupportedLanguage


@dataclass
class FewShotExample:
    """
    A formatted few-shot example for LLM prompts.
    
    Contains the error context and its solution in a format
    ready to be included in prompts.
    """
    error_type: str
    error_message: str
    language: str
    solution: str
    code_fix: Optional[str] = None
    similarity_score: float = 0.0
    
    def to_prompt_format(self) -> str:
        """
        Format as a prompt-ready string.
        
        Returns:
            Formatted string for LLM prompt
        """
        lines = [
            f"**Error ({self.language}):**",
            f"Type: {self.error_type}",
            f"Message: {self.error_message}",
            "",
            f"**Solution:**",
            self.solution,
        ]
        
        if self.code_fix:
            lines.extend([
                "",
                "**Code Fix:**",
                f"```{self.language}",
                self.code_fix,
                "```",
            ])
        
        return "\n".join(lines)


class ErrorRetriever:
    """
    Retrieves and formats error solutions for Few-Shot learning.
    
    Uses the vector store to find similar errors and formats them
    as examples for the LLM to learn from.
    
    Example:
        retriever = ErrorRetriever()
        
        # Get examples for an error
        examples = retriever.get_few_shot_examples(
            error_log=error_log,  # ErrorLog from schemas
            top_k=3
        )
        
        # Format for prompt
        prompt_text = retriever.format_examples_for_prompt(examples)
    """
    
    def __init__(self, store: Optional[ErrorSolutionStore] = None):
        """
        Initialize the error retriever.
        
        Args:
            store: Optional ErrorSolutionStore instance
        """
        self.store = store or ErrorSolutionStore()
    
    def get_few_shot_examples(
        self,
        error_log: ErrorLog,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> list[FewShotExample]:
        """
        Get few-shot examples for an error.
        
        Args:
            error_log: The error to find examples for
            top_k: Maximum number of examples to return
            min_similarity: Minimum similarity score (0-1, lower is more similar)
            
        Returns:
            List of FewShotExample objects
        """
        # Search for similar errors
        results = self.store.search_similar_errors(
            error_message=error_log.message,
            language=error_log.language if isinstance(error_log.language, str) else error_log.language.value,
            error_type=error_log.error_type,
            top_k=top_k,
        )
        
        examples = []
        for result in results:
            # Filter by minimum similarity (distance)
            distance = result.get("distance", 1.0)
            if distance > min_similarity and min_similarity > 0:
                continue
            
            metadata = result.get("metadata", {})
            
            example = FewShotExample(
                error_type=metadata.get("error_type", "Unknown"),
                error_message=result.get("document", "").split("Error Message: ")[-1].split("\n")[0],
                language=metadata.get("language", "unknown"),
                solution=metadata.get("solution", ""),
                code_fix=metadata.get("code_fix"),
                similarity_score=1.0 - distance if distance else 0.0,  # Convert distance to similarity
            )
            examples.append(example)
        
        return examples
    
    def get_few_shot_examples_raw(
        self,
        error_message: str,
        language: str,
        error_type: Optional[str] = None,
        top_k: int = 3,
    ) -> list[FewShotExample]:
        """
        Get few-shot examples from raw error information.
        
        Args:
            error_message: The error message
            language: Programming language
            error_type: Optional error type
            top_k: Maximum number of examples
            
        Returns:
            List of FewShotExample objects
        """
        results = self.store.search_similar_errors(
            error_message=error_message,
            language=language,
            error_type=error_type,
            top_k=top_k,
        )
        
        examples = []
        for result in results:
            metadata = result.get("metadata", {})
            distance = result.get("distance", 1.0)
            
            example = FewShotExample(
                error_type=metadata.get("error_type", "Unknown"),
                error_message=result.get("document", "").split("Error Message: ")[-1].split("\n")[0],
                language=metadata.get("language", "unknown"),
                solution=metadata.get("solution", ""),
                code_fix=metadata.get("code_fix"),
                similarity_score=1.0 - distance if distance else 0.0,
            )
            examples.append(example)
        
        return examples
    
    def format_examples_for_prompt(
        self,
        examples: list[FewShotExample],
        header: str = "Here are some similar error solutions that might help:",
    ) -> str:
        """
        Format examples as a prompt section.
        
        Args:
            examples: List of FewShotExample objects
            header: Header text for the examples section
            
        Returns:
            Formatted string for LLM prompt
        """
        if not examples:
            return ""
        
        parts = [header, ""]
        
        for i, example in enumerate(examples, 1):
            parts.append(f"### Example {i}")
            parts.append(example.to_prompt_format())
            parts.append("")
        
        return "\n".join(parts)
    
    def format_context_for_error(
        self,
        error_log: ErrorLog,
        top_k: int = 3,
    ) -> str:
        """
        Get formatted context for an error.
        
        Convenience method that retrieves examples and formats them.
        
        Args:
            error_log: The error to get context for
            top_k: Number of examples to include
            
        Returns:
            Formatted prompt context string
        """
        examples = self.get_few_shot_examples(error_log, top_k=top_k)
        return self.format_examples_for_prompt(examples)
