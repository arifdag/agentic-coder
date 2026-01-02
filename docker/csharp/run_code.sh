#!/bin/sh
# C# Code Execution Script for Sandbox
# Compiles and runs C# code using dotnet, capturing output and errors

INPUT_DIR="/code/input"
OUTPUT_DIR="/code/output"
RESULT_FILE="$OUTPUT_DIR/result.json"
PROJECT_DIR="/tmp/csharp_project"

# Cleanup and create project directory
rm -rf "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR"

# Find C# files
CS_FILES=$(find "$INPUT_DIR" -name "*.cs" 2>/dev/null)

if [ -z "$CS_FILES" ]; then
    cat > "$RESULT_FILE" << EOF
{
    "success": false,
    "error_type": "FileNotFoundError",
    "error_message": "No C# files found in input directory",
    "stdout": "",
    "stderr": "",
    "execution_time_ms": 0
}
EOF
    cat "$RESULT_FILE"
    exit 0
fi

# Create a minimal console project
cd "$PROJECT_DIR"
dotnet new console --force -o . > /dev/null 2>&1

# Remove default Program.cs and copy input files
rm -f Program.cs
cp $CS_FILES .

# Start timing
START_TIME=$(date +%s%N)

# Build
BUILD_OUTPUT=$(dotnet build -c Release 2>&1)
BUILD_EXIT=$?

if [ $BUILD_EXIT -ne 0 ]; then
    END_TIME=$(date +%s%N)
    EXEC_TIME=$(( (END_TIME - START_TIME) / 1000000 ))
    
    BUILD_ESCAPED=$(printf '%s' "$BUILD_OUTPUT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
    
    cat > "$RESULT_FILE" << EOF
{
    "success": false,
    "error_type": "CompilationError",
    "error_message": "C# compilation failed",
    "stdout": "",
    "stderr": $BUILD_ESCAPED,
    "execution_time_ms": $EXEC_TIME
}
EOF
    cat "$RESULT_FILE"
    exit 0
fi

# Run
STDOUT_FILE=$(mktemp)
STDERR_FILE=$(mktemp)

dotnet run --no-build -c Release > "$STDOUT_FILE" 2> "$STDERR_FILE"
RUN_EXIT=$?

END_TIME=$(date +%s%N)
EXEC_TIME=$(( (END_TIME - START_TIME) / 1000000 ))

STDOUT_ESCAPED=$(cat "$STDOUT_FILE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
STDERR_ESCAPED=$(cat "$STDERR_FILE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

if [ $RUN_EXIT -ne 0 ]; then
    cat > "$RESULT_FILE" << EOF
{
    "success": false,
    "error_type": "RuntimeError",
    "error_message": "C# execution failed",
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
