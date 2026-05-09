"""Router Agent for task classification and routing."""

import re
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    UNKNOWN = "unknown"


class TaskType(str, Enum):
    """Supported task types."""
    UNIT_TEST = "unit_test"
    UI_TEST = "ui_test"
    EXPLANATION = "explanation"


class RoutingDecision(BaseModel):
    """Result of routing decision."""
    
    language: Language = Field(description="Detected programming language")
    task_type: TaskType = Field(description="Classified task type")
    confidence: float = Field(default=1.0, description="Confidence score (0-1)")
    file_extension: Optional[str] = Field(default=None, description="File extension if available")
    framework_hint: Optional[str] = Field(default=None, description="Suggested test framework")


class RouterAgent:
    """Agent responsible for routing requests to appropriate specialists."""
    
    PYTHON_PATTERNS = [
        r'\bdef\s+\w+\s*\(',
        r'\bclass\s+\w+\s*[:\(]',
        r'\bimport\s+\w+',
        r'\bfrom\s+\w+\s+import',
        r'^\s*@\w+',
        r'\bself\.',
        r'\bNone\b',
        r'\bTrue\b|\bFalse\b',
        r':\s*$',
        r'\bprint\s*\(',
    ]
    
    JS_TS_PATTERNS = [
        r'\bfunction\s+\w+\s*\(',
        r'\bconst\s+\w+\s*=',
        r'\blet\s+\w+\s*=',
        r'\bvar\s+\w+\s*=',
        r'=>',
        r'\bexport\s+',
        r'\bimport\s+.*\s+from\s+',
        r'\bconsole\.log\s*\(',
        r'\{.*\}',
        r'\bnew\s+\w+',
    ]
    
    TS_SPECIFIC_PATTERNS = [
        r':\s*(string|number|boolean|any|void|never)\b',
        r'interface\s+\w+',
        r'type\s+\w+\s*=',
        r'<\w+>',
        r'\bas\s+\w+',
    ]
    
    def __init__(self):
        """Initialize the router agent."""
        pass
    
    def detect_language(self, code: str, file_path: Optional[str] = None) -> Language:
        """Detect the programming language of the code.
        
        Args:
            code: Source code to analyze
            file_path: Optional file path for extension-based detection
            
        Returns:
            Detected language
        """
        if file_path:
            ext = file_path.lower().split('.')[-1] if '.' in file_path else None
            if ext == 'py':
                return Language.PYTHON
            elif ext == 'ts' or ext == 'tsx':
                return Language.TYPESCRIPT
            elif ext == 'js' or ext == 'jsx':
                return Language.JAVASCRIPT
        
        python_score = sum(1 for p in self.PYTHON_PATTERNS if re.search(p, code, re.MULTILINE))
        js_score = sum(1 for p in self.JS_TS_PATTERNS if re.search(p, code, re.MULTILINE))
        ts_score = sum(1 for p in self.TS_SPECIFIC_PATTERNS if re.search(p, code, re.MULTILINE))
        
        if ts_score > 0 and js_score > python_score:
            return Language.TYPESCRIPT
        elif js_score > python_score:
            return Language.JAVASCRIPT
        elif python_score > 0:
            return Language.PYTHON
        
        return Language.UNKNOWN
    
    # Keywords that strongly indicate a unit-testing task. When *any* of
    # these appear in the request, we route to UNIT_TEST regardless of
    # whether explanation keywords are also present -- benchmark
    # problem descriptions frequently contain words like ``comment``,
    # ``complexity`` or ``describe`` that used to hijack routing even
    # when the user explicitly asked for ``unit tests``.
    UNIT_TEST_KEYWORDS = (
        'unit test', 'unit-test', 'generate test', 'generate a test',
        'generate tests', 'write test', 'write a test', 'write tests',
        'pytest', 'jest', 'mocha', 'coverage',
        'test case', 'test function', 'test suite',
    )

    # UI tests are always highest priority because they're the most
    # specific/unambiguous.
    UI_TEST_KEYWORDS = (
        'ui test', 'e2e test', 'end-to-end', 'end to end',
        'playwright', 'selenium', 'browser automation',
        'web test', 'browser test', 'acceptance test',
    )

    # Explanation keywords; only used when the request is *not*
    # clearly a test generation job. Kept narrow on purpose -- words
    # like ``comment`` or ``complexity`` are too common in benchmark
    # descriptions to treat as signal on their own.
    EXPLAIN_KEYWORDS = (
        'explain', 'what does', 'how does', 'walkthrough',
        'describe in plain english', 'summarise', 'summarize',
        'big-o analysis', 'time complexity analysis', 'complexity',
    )

    def classify_task(self, user_request: str) -> TaskType:
        """Classify the task type based on user request.

        Order of precedence:

        1. UI test keywords (most specific) -> UI_TEST
        2. Explicit unit-test keywords -> UNIT_TEST
        3. Explanation keywords -> EXPLANATION
        4. Default -> UNIT_TEST

        We deliberately check unit-test keywords *before* explanation
        keywords because benchmark problem descriptions often mention
        "comment", "complexity", or "describe" while the actual user
        request at the top of the string clearly says "Generate
        comprehensive unit tests". The old ordering misrouted such
        cases as explanations and skipped the sandbox gate entirely.
        """
        request_lower = user_request.lower()

        for keyword in self.UI_TEST_KEYWORDS:
            if keyword in request_lower:
                return TaskType.UI_TEST

        for keyword in self.UNIT_TEST_KEYWORDS:
            if keyword in request_lower:
                return TaskType.UNIT_TEST

        for keyword in self.EXPLAIN_KEYWORDS:
            if keyword in request_lower:
                return TaskType.EXPLANATION

        return TaskType.UNIT_TEST
    
    def get_framework_hint(self, language: Language, task_type: TaskType) -> str:
        """Get suggested framework for the task.
        
        Args:
            language: Detected language
            task_type: Task type
            
        Returns:
            Framework name suggestion
        """
        if task_type == TaskType.UI_TEST:
            return "playwright"
        
        if task_type == TaskType.UNIT_TEST:
            framework_map = {
                Language.PYTHON: "pytest",
                Language.JAVASCRIPT: "jest",
                Language.TYPESCRIPT: "jest",
            }
            return framework_map.get(language, "pytest")
        
        return "none"
    
    def route(
        self,
        code: str,
        user_request: str,
        file_path: Optional[str] = None,
    ) -> RoutingDecision:
        """Route a request to the appropriate specialist.
        
        Args:
            code: Source code to process
            user_request: User's request
            file_path: Optional file path
            
        Returns:
            Routing decision with language, task type, and framework hint
        """
        language = self.detect_language(code, file_path)
        task_type = self.classify_task(user_request)
        framework = self.get_framework_hint(language, task_type)
        
        ext = None
        if file_path and '.' in file_path:
            ext = file_path.split('.')[-1]
        
        confidence = 1.0 if language != Language.UNKNOWN else 0.5
        
        return RoutingDecision(
            language=language,
            task_type=task_type,
            confidence=confidence,
            file_extension=ext,
            framework_hint=framework,
        )
