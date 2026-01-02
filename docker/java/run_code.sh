#!/bin/bash
# Java Code Execution Script for Sandbox
# Compiles and runs Java code, capturing output and errors

set -e

INPUT_DIR="/code/input"
OUTPUT_DIR="/code/output"
RESULT_FILE="$OUTPUT_DIR/result.json"

# Initialize result
init_result() {
    echo '{
        "success": false,
        "stdout": "",
        "stderr": "",
        "execution_time_ms": 0,
        "error_type": null,
        "error_message": null,
        "compile_output": ""
    }'
}

# Find the main class (class with public static void main)
find_main_class() {
    local java_file=$1
    grep -l "public static void main" "$java_file" | head -1 | xargs basename | sed 's/.java$//'
}

# Main execution
main() {
    cd "$INPUT_DIR"
    
    # Find Java files
    JAVA_FILES=$(find . -name "*.java" 2>/dev/null)
    
    if [ -z "$JAVA_FILES" ]; then
        cat > "$RESULT_FILE" << EOF
{
    "success": false,
    "error_type": "FileNotFoundError",
    "error_message": "No Java files found in input directory",
    "stdout": "",
    "stderr": "",
    "execution_time_ms": 0
}
EOF
        cat "$RESULT_FILE"
        exit 0
    fi
    
    # Compile
    START_TIME=$(date +%s%N)
    COMPILE_OUTPUT=$(javac $JAVA_FILES 2>&1) || COMPILE_FAILED=1
    
    if [ "$COMPILE_FAILED" = "1" ]; then
        END_TIME=$(date +%s%N)
        EXEC_TIME=$(( (END_TIME - START_TIME) / 1000000 ))
        
        # Escape JSON
        COMPILE_OUTPUT_ESCAPED=$(echo "$COMPILE_OUTPUT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
        
        cat > "$RESULT_FILE" << EOF
{
    "success": false,
    "error_type": "CompilationError",
    "error_message": "Java compilation failed",
    "stdout": "",
    "stderr": $COMPILE_OUTPUT_ESCAPED,
    "execution_time_ms": $EXEC_TIME
}
EOF
        cat "$RESULT_FILE"
        exit 0
    fi
    
    # Find and run main class
    MAIN_CLASS=$(find_main_class "$(echo $JAVA_FILES | tr ' ' '\n' | head -1)")
    
    if [ -z "$MAIN_CLASS" ]; then
        MAIN_CLASS="Main"
    fi
    
    # Run
    STDOUT_FILE=$(mktemp)
    STDERR_FILE=$(mktemp)
    
    java "$MAIN_CLASS" > "$STDOUT_FILE" 2> "$STDERR_FILE" || RUN_FAILED=1
    
    END_TIME=$(date +%s%N)
    EXEC_TIME=$(( (END_TIME - START_TIME) / 1000000 ))
    
    STDOUT_ESCAPED=$(cat "$STDOUT_FILE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
    STDERR_ESCAPED=$(cat "$STDERR_FILE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
    
    if [ "$RUN_FAILED" = "1" ]; then
        cat > "$RESULT_FILE" << EOF
{
    "success": false,
    "error_type": "RuntimeError",
    "error_message": "Java execution failed",
    "stdout": $STDOUT_ESCAPED,
    "stderr": $STDERR_ESCAPED,
    "execution_time_ms": $EXEC_TIME
}
EOF
    else
        cat > "$RESULT_FILE" << EOF
{
    "success": true,
    "error_type": null,
    "error_message": null,
    "stdout": $STDOUT_ESCAPED,
    "stderr": $STDERR_ESCAPED,
    "execution_time_ms": $EXEC_TIME
}
EOF
    fi
    
    cat "$RESULT_FILE"
    rm -f "$STDOUT_FILE" "$STDERR_FILE"
}

main
