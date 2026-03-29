"""LLM-as-Judge for verifying code explanations against a rubric."""

import re
import json
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from .models import GateResult, Finding, Severity


RUBRIC_SYSTEM_PROMPT = """You are a senior code reviewer evaluating the quality of a code explanation.

Evaluate the explanation against these criteria and output a JSON array:

1. "factual_consistency" - Does the explanation accurately describe what the code does?
2. "completeness" - Are all functions and classes covered in the walkthrough?
3. "complexity_plausibility" - Is the Big-O claim reasonable for the code structure?
4. "clarity" - Is the explanation clear and well-structured?

For each criterion output:
{
  "criterion": "<name>",
  "passed": true/false,
  "reasoning": "Brief explanation"
}

Output ONLY a JSON array of four objects. No other text."""

RUBRIC_PROMPT_TEMPLATE = """## Source code
```python
{code}
```

## Explanation to evaluate
```json
{explanation_json}
```

Evaluate the explanation against the four criteria now:"""


class ExplanationJudge:
    """LLM-based judge for verifying code explanation quality."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _parse_rubric(self, response_text: str) -> List[dict]:
        """Parse the judge response into rubric results."""
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if not json_match:
            return []
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return []

    def verify(self, code: str, explanation_json: str) -> GateResult:
        """Verify an explanation against the rubric.

        Returns a GateResult with findings for each failed criterion.
        """
        prompt = RUBRIC_PROMPT_TEMPLATE.format(
            code=code,
            explanation_json=explanation_json,
        )

        messages = [
            SystemMessage(content=RUBRIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        results = self._parse_rubric(response_text)

        findings: List[Finding] = []
        all_passed = True

        for item in results:
            if not isinstance(item, dict):
                continue
            passed = item.get("passed", True)
            criterion = item.get("criterion", "unknown")
            reasoning = item.get("reasoning", "")

            if not passed:
                all_passed = False
                findings.append(Finding(
                    severity=Severity.ERROR,
                    code=f"rubric_{criterion}",
                    message=f"[{criterion}] {reasoning}",
                    suggestion=f"Fix the {criterion} issue in the explanation",
                ))

        return GateResult(
            gate_name="explanation_judge",
            passed=all_passed,
            findings=findings,
            details=response_text[:1000],
        )

    def format_feedback(self, gate_result: GateResult) -> str:
        """Format judge findings as structured feedback for repair."""
        if gate_result.passed:
            return ""
        lines = ["Rubric failures:"]
        for f in gate_result.findings:
            lines.append(f"  - {f.message}")
            if f.suggestion:
                lines.append(f"    Fix: {f.suggestion}")
        return "\n".join(lines)
