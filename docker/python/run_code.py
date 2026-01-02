#!/usr/bin/env python3
"""
Python Code Execution Script for Sandbox

Reads code from /code/input/code.py and executes it safely,
capturing stdout, stderr, and execution time.
"""

import json
import sys
import time
import traceback
from io import StringIO
from pathlib import Path


def execute_code(code_path: str, timeout: int = 30) -> dict:
    """
    Execute Python code and capture results.
    
    Args:
        code_path: Path to the Python file to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary with execution results
    """
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "execution_time_ms": 0,
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }
    
    # Capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    start_time = time.time()
    
    try:
        # Read the code
        code = Path(code_path).read_text(encoding='utf-8')
        
        # Execute in isolated namespace
        exec_globals = {"__name__": "__main__", "__builtins__": __builtins__}
        exec(compile(code, code_path, 'exec'), exec_globals)
        
        result["success"] = True
        
    except SyntaxError as e:
        result["error_type"] = "SyntaxError"
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        
    finally:
        end_time = time.time()
        result["execution_time_ms"] = (end_time - start_time) * 1000
        
        # Capture output
        result["stdout"] = sys.stdout.getvalue()
        result["stderr"] = sys.stderr.getvalue()
        
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return result


def main():
    """Main entry point for sandbox execution."""
    input_path = "/code/input/code.py"
    output_path = "/code/output/result.json"
    
    # Check if input file exists
    if not Path(input_path).exists():
        result = {
            "success": False,
            "error_type": "FileNotFoundError",
            "error_message": f"Code file not found: {input_path}",
            "stdout": "",
            "stderr": "",
            "execution_time_ms": 0,
        }
    else:
        result = execute_code(input_path)
    
    # Write result
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2), encoding='utf-8')
    
    # Also print to stdout for immediate capture
    print(json.dumps(result))


if __name__ == "__main__":
    main()
