"""Tests for Phase 5: Multi-Language (JS/TS) Support."""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.agents.jest_test import JestTestAgent
from src.agents.unit_test import GeneratedTest, RepairContext
from src.agents.router import RouterAgent, TaskType, Language
from src.verification.js_sandbox import JsSandboxExecutor
from src.verification.dependency import (
    DependencyValidator,
    extract_js_imports,
    NODE_BUILTINS,
    KNOWN_JS_TEST_PACKAGES,
)
from src.verification.sast import SastAnalyzer
from src.config import JsSandboxConfig


# ── JestTestAgent tests ───────────────────────────────────────────────


class TestJestTestAgent:
    def _mock_llm(self, response_text: str) -> MagicMock:
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = response_text
        llm.invoke.return_value = mock_response
        return llm

    def test_generate_returns_generated_test(self):
        code = """const { add } = require('./source_module');
describe('add', () => {
  test('adds two numbers', () => {
    expect(add(1, 2)).toBe(3);
  });
  it('handles negatives', () => {
    expect(add(-1, 1)).toBe(0);
  });
});"""
        llm = self._mock_llm(code)
        agent = JestTestAgent(llm)
        result = agent.generate("function add(a,b){return a+b;}", file_path="add.js")

        assert isinstance(result, GeneratedTest)
        assert "describe" in result.test_code
        assert len(result.test_functions) == 2
        assert "adds two numbers" in result.test_functions

    def test_extract_test_functions_from_it_and_test(self):
        code = """
test('should add', () => {});
it('handles null', () => {});
test("works with strings", () => {});
"""
        agent = JestTestAgent(MagicMock())
        names = agent._extract_test_functions(code)
        assert "should add" in names
        assert "handles null" in names
        assert "works with strings" in names

    def test_extract_code_from_markdown_fence(self):
        response = """Here is the test:
```javascript
const { add } = require('./source_module');
test('basic', () => { expect(add(1,2)).toBe(3); });
```
"""
        agent = JestTestAgent(MagicMock())
        code = agent._extract_code_from_response(response)
        assert "require('./source_module')" in code
        assert "```" not in code

    def test_extract_imports(self):
        code = """const { add } = require('./source_module');
import { foo } from './bar';
const path = require('path');
"""
        agent = JestTestAgent(MagicMock())
        imports = agent._extract_imports(code)
        assert any("require('./source_module')" in i for i in imports)
        assert any("require('path')" in i for i in imports)

    def test_repair_invokes_llm(self):
        llm = self._mock_llm("test('fixed', () => { expect(true).toBe(true); });")
        agent = JestTestAgent(llm)
        ctx = RepairContext(
            previous_code="broken code",
            error_type="syntax_error",
            error_message="Unexpected token",
        )
        result = agent.repair(ctx)
        assert isinstance(result, GeneratedTest)
        llm.invoke.assert_called_once()


# ── JsSandboxExecutor tests ───────────────────────────────────────────


class TestJsSandboxExecutor:
    def test_parse_jest_output_passing(self):
        stdout = """
PASS ./source_module.test.js
  add
    ✓ adds two numbers (2 ms)
    ✓ handles negatives (1 ms)

Tests:  2 passed, 2 total
"""
        executor = JsSandboxExecutor()
        parsed = executor._parse_jest_output(stdout, "")
        assert parsed["tests_passed"] == 2
        assert parsed["tests_failed"] == 0
        assert parsed["tests_run"] == 2
        assert parsed["error_type"] is None

    def test_parse_jest_output_failing(self):
        stdout = """
FAIL ./source_module.test.js
  add
    ✓ adds two numbers (2 ms)
    ✕ handles negatives (3 ms)

Tests:  1 failed, 1 passed, 2 total
"""
        executor = JsSandboxExecutor()
        parsed = executor._parse_jest_output(stdout, "")
        assert parsed["tests_passed"] == 1
        assert parsed["tests_failed"] == 1
        assert parsed["tests_run"] == 2

    def test_parse_jest_output_syntax_error(self):
        stderr = "SyntaxError: Unexpected token )"
        executor = JsSandboxExecutor()
        parsed = executor._parse_jest_output("", stderr)
        assert parsed["error_type"] == "syntax_error"
        assert "Unexpected token" in parsed["error_message"]

    def test_parse_jest_output_module_not_found(self):
        stderr = "Cannot find module './nonexistent'"
        executor = JsSandboxExecutor()
        parsed = executor._parse_jest_output("", stderr)
        assert parsed["error_type"] == "import_error"
        assert "nonexistent" in parsed["error_message"]

    def test_fix_imports_require(self):
        code = "const { add } = require('./myModule');\ntest('x', () => {});"
        executor = JsSandboxExecutor()
        fixed = executor._fix_imports(code)
        assert "require('./source_module')" in fixed
        assert "require('./myModule')" not in fixed

    def test_fix_imports_esm(self):
        code = "import { add } from './myModule';\ntest('x', () => {});"
        executor = JsSandboxExecutor()
        fixed = executor._fix_imports(code)
        assert "'./source_module'" in fixed
        assert "'./myModule'" not in fixed

    def test_config_defaults(self):
        executor = JsSandboxExecutor()
        assert executor.image_name == "llm-agent-node"
        assert executor.timeout == 60
        assert executor.network_disabled is True

    def test_config_from_object(self):
        cfg = JsSandboxConfig(image_name="custom-node", timeout=90)
        executor = JsSandboxExecutor(cfg)
        assert executor.image_name == "custom-node"
        assert executor.timeout == 90


# ── JS Dependency Validation tests ────────────────────────────────────


class TestJsDependencyValidation:
    def test_extract_require_imports(self):
        code = """
const express = require('express');
const { Router } = require('express');
const path = require('path');
const myMod = require('./local');
"""
        imports = extract_js_imports(code)
        assert "express" in imports
        assert "path" in imports
        assert "./local" not in imports and "local" not in imports

    def test_extract_esm_imports(self):
        code = """
import React from 'react';
import { useState } from 'react';
import axios from 'axios';
import './styles.css';
import { helper } from '../utils';
"""
        imports = extract_js_imports(code)
        assert "react" in imports
        assert "axios" in imports
        assert "styles.css" not in imports and "./styles.css" not in imports

    def test_extract_scoped_packages(self):
        code = "const test = require('@testing-library/react');"
        imports = extract_js_imports(code)
        assert "@testing-library/react" in imports

    def test_node_builtins_recognized(self):
        validator = DependencyValidator()
        assert validator._is_known_safe_js("fs")
        assert validator._is_known_safe_js("path")
        assert validator._is_known_safe_js("http")
        assert validator._is_known_safe_js("crypto")

    def test_test_packages_recognized(self):
        validator = DependencyValidator()
        assert validator._is_known_safe_js("jest")
        assert validator._is_known_safe_js("mocha")

    def test_validate_js_with_known_packages(self):
        code = """
const path = require('path');
const jest = require('jest');
const mod = require('./source_module');
"""
        validator = DependencyValidator()
        result = validator.validate(code, language="javascript")
        assert result.passed

    def test_validate_js_phantom_package(self):
        code = "const fake = require('definitely-not-a-real-npm-package-xyz123');"
        validator = DependencyValidator(pypi_timeout=5)
        result = validator.validate(code, language="javascript")
        assert not result.passed
        assert any("PHANTOM-PKG" in f.code for f in result.findings)

    def test_validate_defaults_to_python(self):
        code = "import os\nimport sys\n"
        validator = DependencyValidator()
        result = validator.validate(code)
        assert result.passed


# ── SAST JS extension tests ──────────────────────────────────────────


class TestSastJsSupport:
    def test_filename_for_javascript(self):
        analyzer = SastAnalyzer()
        assert analyzer._filename_for_language("javascript") == "generated_code.js"
        assert analyzer._filename_for_language("typescript") == "generated_code.ts"
        assert analyzer._filename_for_language("python") == "generated_code.py"
        assert analyzer._filename_for_language(None) == "generated_code.py"

    def test_is_python_detection(self):
        analyzer = SastAnalyzer()
        assert analyzer._is_python("generated_code.py") is True
        assert analyzer._is_python("generated_code.js") is False
        assert analyzer._is_python("generated_code.ts") is False


# ── Router JS/TS detection tests ─────────────────────────────────────


class TestRouterJsDetection:
    def test_detect_js_from_extension(self):
        router = RouterAgent()
        assert router.detect_language("", file_path="app.js") == Language.JAVASCRIPT
        assert router.detect_language("", file_path="app.jsx") == Language.JAVASCRIPT

    def test_detect_ts_from_extension(self):
        router = RouterAgent()
        assert router.detect_language("", file_path="app.ts") == Language.TYPESCRIPT
        assert router.detect_language("", file_path="app.tsx") == Language.TYPESCRIPT

    def test_detect_python_from_extension(self):
        router = RouterAgent()
        assert router.detect_language("", file_path="app.py") == Language.PYTHON

    def test_detect_js_from_code_patterns(self):
        router = RouterAgent()
        code = "const x = 5;\nfunction foo() { return x; }\nconsole.log(foo());"
        lang = router.detect_language(code)
        assert lang in (Language.JAVASCRIPT, Language.TYPESCRIPT)

    def test_framework_hint_jest_for_js(self):
        router = RouterAgent()
        assert router.get_framework_hint(Language.JAVASCRIPT, TaskType.UNIT_TEST) == "jest"
        assert router.get_framework_hint(Language.TYPESCRIPT, TaskType.UNIT_TEST) == "jest"

    def test_framework_hint_pytest_for_python(self):
        router = RouterAgent()
        assert router.get_framework_hint(Language.PYTHON, TaskType.UNIT_TEST) == "pytest"


# ── JsSandboxConfig tests ────────────────────────────────────────────


class TestJsSandboxConfig:
    def test_defaults(self):
        cfg = JsSandboxConfig()
        assert cfg.enabled is True
        assert cfg.image_name == "llm-agent-node"
        assert cfg.timeout == 60
        assert cfg.memory_limit == "512m"
        assert cfg.network_disabled is True

    def test_from_env(self):
        with patch.dict("os.environ", {
            "JS_SANDBOX_ENABLED": "false",
            "JS_SANDBOX_IMAGE": "my-node",
            "JS_SANDBOX_TIMEOUT": "120",
        }):
            cfg = JsSandboxConfig.from_env()
            assert cfg.enabled is False
            assert cfg.image_name == "my-node"
            assert cfg.timeout == 120
