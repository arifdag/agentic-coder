# C# Sandbox for Code Execution
# Secure, isolated environment for compiling and running untrusted C# code

FROM mcr.microsoft.com/dotnet/sdk:8.0-alpine

# Create non-root user for security
RUN addgroup -S sandbox && adduser -S sandbox -G sandbox -h /home/sandbox

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
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1
ENV DOTNET_NOLOGO=1

# Default command
CMD ["/usr/local/bin/run_code.sh"]
