# Python Sandbox for Code Execution
# Secure, isolated environment for running untrusted Python code

FROM python:3.11-slim

# Create non-root user for security
RUN groupadd -r sandbox && useradd -r -g sandbox -d /home/sandbox -m sandbox

# Install common Python packages that might be needed
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    requests \
    pytest \
    && rm -rf /root/.cache/pip

# Set up working directory
WORKDIR /code

# Create directories with proper permissions
RUN mkdir -p /code/input /code/output \
    && chown -R sandbox:sandbox /code

# Copy execution script
COPY --chown=sandbox:sandbox run_code.py /usr/local/bin/run_code.py
RUN chmod +x /usr/local/bin/run_code.py

# Switch to non-root user
USER sandbox

# Set resource limits via environment (enforced by Docker runtime)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "/usr/local/bin/run_code.py"]
