# Java Sandbox for Code Execution
# Secure, isolated environment for compiling and running untrusted Java code

FROM eclipse-temurin:17-jdk-jammy

# Create non-root user for security
RUN groupadd -r sandbox && useradd -r -g sandbox -d /home/sandbox -m sandbox

# Set up working directory
WORKDIR /code

# Create directories with proper permissions
RUN mkdir -p /code/input /code/output \
    && chown -R sandbox:sandbox /code

# Copy execution script
COPY --chown=sandbox:sandbox run_code.sh /usr/local/bin/run_code.sh
RUN chmod +x /usr/local/bin/run_code.sh

# Switch to non-root user
USER sandbox

# Environment
ENV JAVA_TOOL_OPTIONS="-Xmx256m -Xms64m"

# Default command
CMD ["/usr/local/bin/run_code.sh"]
