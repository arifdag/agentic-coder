# LLM Agent Platform - Sandbox Environment
# Secure, isolated container for running generated tests

FROM python:3.11-slim

# Create non-root user for security
RUN groupadd -r sandbox && useradd -r -g sandbox -d /home/sandbox -s /sbin/nologin sandbox

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python testing dependencies
RUN pip install --no-cache-dir \
    pytest>=8.0.0 \
    pytest-timeout>=2.2.0 \
    coverage>=7.0.0 \
    pytest-cov>=4.0.0

# Create working directory
WORKDIR /workspace

# Set ownership
RUN chown -R sandbox:sandbox /workspace

# Switch to non-root user
USER sandbox

# Default command: run pytest with coverage
CMD ["pytest", "--tb=short", "-v"]
