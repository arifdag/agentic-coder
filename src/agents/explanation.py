"""Code Explanation Agent for structured code understanding with complexity analysis."""

import re
import json
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel


class WalkthroughBlock(BaseModel):
    """A single block-level explanation."""

    block: str = Field(description="Line range, e.g. 'lines 5-12'")
    explanation: str = Field(description="What this block does and why")


class ComplexityAnalysis(BaseModel):
    """Big-O time and space complexity with assumptions."""

    time: str = Field(default="O(?)", description="Time complexity, e.g. 'O(n log n)'")
    space: str = Field(default="O(?)", description="Space complexity, e.g. 'O(n)'")
    assumptions: str = Field(default="", description="What n represents")


class CodeExplanation(BaseModel):
    """Structured explanation of a code artifact."""

    summary: str = Field(description="High-level intent in 1-2 sentences")
    walkthrough: List[WalkthroughBlock] = Field(
        default_factory=list, description="Block-level explanations"
    )
    complexity: ComplexityAnalysis = Field(
        default_factory=ComplexityAnalysis, description="Big-O analysis"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="External packages used"
    )
    potential_issues: List[str] = Field(
        default_factory=list, description="Risks, missing validation, etc."
    )

    def to_markdown(self) -> str:
        """Render the explanation as readable Markdown."""
        parts: List[str] = []

        parts.append(f"## Summary\n\n{self.summary}")

        if self.walkthrough:
            parts.append("## Walkthrough\n")
            for block in self.walkthrough:
                parts.append(f"**{block.block}** -- {block.explanation}")

        parts.append("## Complexity\n")
        parts.append(f"| Metric | Value |")
        parts.append(f"|--------|-------|")
        parts.append(f"| Time   | {self.complexity.time} |")
        parts.append(f"| Space  | {self.complexity.space} |")
        if self.complexity.assumptions:
            parts.append(f"| Assumptions | {self.complexity.assumptions} |")

        if self.dependencies:
            parts.append("## Dependencies\n")
            for dep in self.dependencies:
                parts.append(f"- {dep}")

        if self.potential_issues:
            parts.append("## Potential Issues\n")
            for issue in self.potential_issues:
                parts.append(f"- {issue}")

        return "\n\n".join(parts)


class ExplanationRepairContext(BaseModel):
    """Context for repairing a failed explanation."""

    previous_explanation: str = Field(description="Previous explanation as JSON")
    error_type: str = Field(description="Type of error")
    error_message: str = Field(description="Error message from verification")
    judge_feedback: Optional[str] = Field(
        default=None, description="Rubric failures from explanation judge"
    )
    complexity_feedback: Optional[str] = Field(
        default=None, description="AST mismatch details from complexity validator"
    )


SYSTEM_PROMPT = """You are an expert software engineer who produces clear, structured code explanations.

You MUST respond with a single JSON object matching this exact schema:

{
  "summary": "High-level intent in 1-2 sentences",
  "walkthrough": [
    {"block": "lines X-Y", "explanation": "What this block does"}
  ],
  "complexity": {
    "time": "O(...)",
    "space": "O(...)",
    "assumptions": "What n represents"
  },
  "dependencies": ["package1", "package2"],
  "potential_issues": ["Issue 1", "Issue 2"]
}

Guidelines:
1. The summary should capture the code's purpose concisely.
2. The walkthrough must cover every function and class. Reference line numbers.
3. Complexity analysis must specify what variable(s) drive the Big-O.
4. List only external (non-stdlib) dependencies.
5. Potential issues should flag missing input validation, unbounded recursion,
   thread-safety problems, or other real concerns.
6. Output ONLY valid JSON. No markdown, no extra text."""

GENERATION_TEMPLATE = """Analyze the following Python code and produce a structured explanation.

```python
{code}
```

{context_section}

Produce the JSON explanation now:"""

REPAIR_TEMPLATE = """Your previous explanation failed verification. Fix the issues below.

Previous explanation:
```json
{previous_explanation}
```

Errors:
- Type: {error_type}
- Message: {error_message}
{judge_section}
{complexity_section}

Produce a corrected JSON explanation. Output ONLY valid JSON."""


class ExplanationAgent:
    """Agent for producing structured code explanations."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _parse_explanation(self, response: str) -> CodeExplanation:
        """Extract a CodeExplanation from the LLM response."""
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return CodeExplanation(**data)
            except (json.JSONDecodeError, Exception):
                pass

        return CodeExplanation(
            summary=response.strip()[:500],
            walkthrough=[],
            complexity=ComplexityAnalysis(),
            dependencies=[],
            potential_issues=["Failed to parse structured explanation from LLM response"],
        )

    def generate(
        self,
        code: str,
        file_path: Optional[str] = None,
    ) -> CodeExplanation:
        context_section = ""
        if file_path:
            context_section = f"File: {file_path}"

        prompt = GENERATION_TEMPLATE.format(
            code=code,
            context_section=context_section,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        return self._parse_explanation(response_text)

    def repair(self, context: ExplanationRepairContext) -> CodeExplanation:
        judge_section = ""
        if context.judge_feedback:
            judge_section = f"\nJudge feedback:\n{context.judge_feedback}"

        complexity_section = ""
        if context.complexity_feedback:
            complexity_section = f"\nComplexity validation:\n{context.complexity_feedback}"

        prompt = REPAIR_TEMPLATE.format(
            previous_explanation=context.previous_explanation,
            error_type=context.error_type,
            error_message=context.error_message,
            judge_section=judge_section,
            complexity_section=complexity_section,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        return self._parse_explanation(response_text)
