"""
Tests for RAG Components

Tests the vector store and error retriever functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from src.core.schemas import ErrorLog, SupportedLanguage
from src.rag.error_retriever import ErrorRetriever, FewShotExample


class TestFewShotExample:
    """Tests for FewShotExample dataclass."""
    
    def test_basic_example(self):
        """Test creating a basic example."""
        example = FewShotExample(
            error_type="TypeError",
            error_message="unsupported operand",
            language="python",
            solution="Use str() to convert",
        )
        
        assert example.error_type == "TypeError"
        assert example.code_fix is None
    
    def test_example_with_code_fix(self):
        """Test example with code fix."""
        example = FewShotExample(
            error_type="KeyError",
            error_message="key not found",
            language="python",
            solution="Use .get() method",
            code_fix="value = dict.get('key', default)",
        )
        
        assert example.code_fix is not None
    
    def test_to_prompt_format(self):
        """Test prompt formatting."""
        example = FewShotExample(
            error_type="ValueError",
            error_message="invalid literal",
            language="python",
            solution="Check input before conversion",
            code_fix="if input.isdigit():\n    number = int(input)",
        )
        
        prompt = example.to_prompt_format()
        
        assert "**Error (python):**" in prompt
        assert "ValueError" in prompt
        assert "**Solution:**" in prompt
        assert "**Code Fix:**" in prompt
        assert "```python" in prompt


class TestErrorRetriever:
    """Tests for ErrorRetriever class."""
    
    @patch('src.rag.error_retriever.ErrorSolutionStore')
    def test_retriever_initialization(self, mock_store_class):
        """Test retriever initialization."""
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        
        retriever = ErrorRetriever()
        
        assert retriever.store is not None
    
    @patch('src.rag.error_retriever.ErrorSolutionStore')
    def test_get_few_shot_examples(self, mock_store_class):
        """Test retrieving few-shot examples."""
        mock_store = MagicMock()
        mock_store.search_similar_errors.return_value = [
            {
                "id": "doc-1",
                "document": "Language: python\nError Type: TypeError\nError Message: test error",
                "distance": 0.2,
                "metadata": {
                    "error_type": "TypeError",
                    "language": "python",
                    "solution": "Fix the type issue",
                    "code_fix": "use str()",
                },
            }
        ]
        mock_store_class.return_value = mock_store
        
        retriever = ErrorRetriever()
        
        error_log = ErrorLog(
            error_id="err-1",
            draft_id="draft-1",
            error_type="TypeError",
            message="type error occurred",
            language=SupportedLanguage.PYTHON,
        )
        
        examples = retriever.get_few_shot_examples(error_log, top_k=3)
        
        assert len(examples) == 1
        assert examples[0].error_type == "TypeError"
        assert examples[0].solution == "Fix the type issue"
    
    @patch('src.rag.error_retriever.ErrorSolutionStore')
    def test_format_examples_for_prompt(self, mock_store_class):
        """Test formatting examples for prompt."""
        retriever = ErrorRetriever(store=MagicMock())
        
        examples = [
            FewShotExample(
                error_type="Error1",
                error_message="msg1",
                language="python",
                solution="sol1",
            ),
            FewShotExample(
                error_type="Error2",
                error_message="msg2",
                language="python",
                solution="sol2",
            ),
        ]
        
        formatted = retriever.format_examples_for_prompt(examples)
        
        assert "Example 1" in formatted
        assert "Example 2" in formatted
        assert "Error1" in formatted
        assert "Error2" in formatted
    
    @patch('src.rag.error_retriever.ErrorSolutionStore')
    def test_format_empty_examples(self, mock_store_class):
        """Test formatting with no examples."""
        retriever = ErrorRetriever(store=MagicMock())
        
        formatted = retriever.format_examples_for_prompt([])
        
        assert formatted == ""
    
    @patch('src.rag.error_retriever.ErrorSolutionStore')
    def test_format_context_for_error(self, mock_store_class):
        """Test convenience method for error context."""
        mock_store = MagicMock()
        mock_store.search_similar_errors.return_value = [
            {
                "id": "doc-1",
                "document": "Error Message: similar error",
                "distance": 0.1,
                "metadata": {
                    "error_type": "TestError",
                    "language": "python",
                    "solution": "Test solution",
                },
            }
        ]
        mock_store_class.return_value = mock_store
        
        retriever = ErrorRetriever()
        
        error_log = ErrorLog(
            error_id="err-1",
            draft_id="draft-1",
            error_type="TestError",
            message="test",
            language=SupportedLanguage.PYTHON,
        )
        
        context = retriever.format_context_for_error(error_log, top_k=1)
        
        assert "similar error" in context or "Test solution" in context


class TestVectorStoreIntegration:
    """Integration tests for vector store (may require dependencies)."""
    
    @pytest.mark.skipif(
        False,  # Enable test as dependencies should be present
        reason="Requires chromadb and sentence-transformers"
    )
    def test_real_vector_store(self):
        """Test real vector store operations."""
        from src.rag.vector_store import ErrorSolutionStore
            # Ignore cleanup errors on Windows to avoid PermissionError
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            store = ErrorSolutionStore(persist_directory=Path(temp_dir))
            
            # Add an error solution
            doc_id = store.add_error_solution(
                error_type="TestError",
                error_message="This is a test error",
                language="python",
                solution="This is the test solution",
            )
            
            assert doc_id is not None
            
            # Search for similar
            results = store.search_similar_errors(
                error_message="test error message",
                language="python",
                top_k=1,
            )
            
            assert len(results) > 0
