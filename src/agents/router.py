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
    
    def classify_task(self, user_request: str) -> TaskType:
        """Classify the task type based on user request.
        
        Args:
            user_request: User's request string
            
        Returns:
            Classified task type
        """
        request_lower = user_request.lower()
        
        ui_keywords = [
            'ui test', 'e2e test', 'end-to-end', 'end to end',
            'playwright', 'selenium', 'browser', 'click', 'navigate',
            'web test', 'integration test', 'acceptance test',
        ]
        
        explain_keywords = [
            'explain', 'what does', 'how does', 'understand',
            'complexity', 'big-o', 'analysis', 'describe',
            'documentation', 'comment', 'walkthrough',
        ]
        
        unit_keywords = [
            'unit test', 'test', 'pytest', 'jest', 'coverage',
            'assert', 'mock', 'test case', 'test function',
        ]
        
        for keyword in ui_keywords:
            if keyword in request_lower:
                return TaskType.UI_TEST
        
        for keyword in explain_keywords:
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
