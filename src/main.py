"""CLI entry point for the LLM Agent Platform."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .config import Config, LLMConfig
from .graph.pipeline import run_pipeline
from .utils.logging import ConsoleLogger

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """LLM Agent Platform - Multi-agent system for test generation."""
    pass


@cli.command()
@click.option(
    "--file", "-f",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to source file (Python for unit tests, HTML for UI tests)",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for generated tests (default: stdout)",
)
@click.option(
    "--max-retries", "-r",
    type=int,
    default=None,
    help="Maximum repair iterations (default: 3 for unit, 5 for UI)",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--provider",
    type=click.Choice(["groq", "openrouter"]),
    default=None,
    help="LLM provider to use (default: from env)",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model name to use (default: from env)",
)
@click.option(
    "--task-type", "-t",
    type=click.Choice(["unit_test", "ui_test"]),
    default=None,
    help="Test type (default: auto-detect from request)",
)
@click.option(
    "--url", "-u",
    type=str,
    default=None,
    help="Target URL for UI tests (live web application)",
)
@click.option(
    "--description", "-d",
    type=str,
    default=None,
    help="Natural-language test description (used for UI tests)",
)
def generate(
    file: Path,
    output: Path,
    max_retries: int,
    verbose: bool,
    provider: str,
    model: str,
    task_type: str,
    url: str,
    description: str,
):
    """Generate tests for source code or web applications."""
    logger = ConsoleLogger(verbose=verbose)

    is_ui = task_type == "ui_test"

    if is_ui and not url and not file:
        console.print("[red]UI tests require --url or --file (HTML).[/red]")
        sys.exit(1)
    if not is_ui and not file:
        console.print("[red]Unit tests require --file.[/red]")
        sys.exit(1)

    source_code = ""
    html_content = None
    file_path_str = None

    if file:
        try:
            source_code = file.read_text(encoding="utf-8")
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            sys.exit(1)
        file_path_str = str(file)
        logger.start(f"Generating tests for {file}")
        logger.info(f"Read {len(source_code)} characters from {file}")

        if is_ui and file.suffix in (".html", ".htm"):
            html_content = source_code
            source_code = ""
    elif url:
        logger.start(f"Generating UI tests for {url}")

    try:
        config = Config.load(provider=provider)
        if model:
            config.llm.model = model
        config.pipeline.verbose = verbose
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("\n[yellow]Please set up your API key:[/yellow]")
        console.print("  1. Copy .env.example to .env")
        console.print("  2. Add your Groq API key (get one free at https://console.groq.com/)")
        sys.exit(1)

    logger.step("Configuration loaded", f"Provider: {config.llm.provider}, Model: {config.llm.model}")

    if is_ui:
        user_request = description or "Generate Playwright E2E tests"
        if "ui" not in user_request.lower() and "playwright" not in user_request.lower():
            user_request = f"Generate Playwright E2E tests: {user_request}"
        default_retries = config.ui_test.retry_budget
    else:
        user_request = "Generate comprehensive unit tests"
        default_retries = 3

    retries = max_retries if max_retries is not None else default_retries

    try:
        logger.step("Running pipeline...")

        result = run_pipeline(
            code=source_code,
            user_request=user_request,
            file_path=file_path_str,
            max_retries=retries,
            config=config,
            target_url=url,
            html_content=html_content,
            description=description,
        )

    except Exception as e:
        console.print(f"[red]Pipeline error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    success = result["status"] == "success"
    iterations = result["retry_count"] + 1
    tests_generated = result.get("generated_tests")

    logger.end(
        success=success,
        summary=f"Iterations: {iterations}, Tests: {len(result.get('test_functions', []))}",
    )

    if tests_generated:
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(tests_generated, encoding="utf-8")
            console.print(f"\n[green]Tests written to {output}[/green]")
        else:
            console.print("\n[bold]Generated Tests:[/bold]\n")
            syntax = Syntax(tests_generated, "python", theme="monokai", line_numbers=True)
            console.print(syntax)
    else:
        console.print("[red]No tests were generated[/red]")

    if verbose:
        _print_summary(result)

    if not success:
        console.print(f"\n[yellow]Warning: Tests may not be fully verified[/yellow]")
        if result.get("error_message"):
            console.print(f"[yellow]Last error: {result['error_message']}[/yellow]")
        if verbose and result.get("verification_result"):
            vr = result["verification_result"]
            if vr.get("stdout"):
                console.print("\n[bold]Sandbox stdout:[/bold]")
                console.print(vr["stdout"][:2000])
            if vr.get("stderr"):
                console.print("\n[bold]Sandbox stderr:[/bold]")
                console.print(vr["stderr"][:1000])
        sys.exit(1)


def _print_summary(result: dict):
    """Print execution summary table."""
    table = Table(title="Execution Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Status", result.get("status", "unknown"))
    table.add_row("Total Iterations", str(result.get("retry_count", 0) + 1))
    table.add_row("Language", result.get("routing_decision", {}).get("language", "unknown"))
    table.add_row("Task Type", result.get("task_type", "unknown"))
    table.add_row("Tests Generated", str(len(result.get("test_functions", []))))
    
    report = result.get("verification_report") or {}
    if report:
        table.add_row("Verification Passed", str(report.get("overall_passed", False)))
        if report.get("coverage") is not None:
            table.add_row("Coverage", f"{report['coverage']}%")
        if report.get("coverage_gaps"):
            table.add_row("Uncovered Lines", report["coverage_gaps"])
        table.add_row("Summary", report.get("summary", ""))

        for gate in report.get("gates", []):
            status = "PASS" if gate.get("passed") else "FAIL"
            n = len(gate.get("findings", []))
            table.add_row(f"  Gate: {gate['gate_name']}", f"{status} ({n} finding(s))")

    console.print("\n")
    console.print(table)


@cli.command()
def check():
    """Check environment and configuration."""
    console.print("[bold]Checking environment...[/bold]\n")
    
    checks = []
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        checks.append(("dotenv", True, ""))
    except ImportError:
        checks.append(("dotenv", False, "python-dotenv not installed"))
    
    import os
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        checks.append(("GROQ_API_KEY", True, f"Set ({len(groq_key)} chars)"))
    else:
        checks.append(("GROQ_API_KEY", False, "Not set"))
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        checks.append(("OPENROUTER_API_KEY", True, f"Set ({len(openrouter_key)} chars)"))
    else:
        checks.append(("OPENROUTER_API_KEY", False, "Not set (optional)"))
    
    try:
        import docker
        client = docker.from_env()
        client.ping()
        checks.append(("Docker", True, "Running"))
    except Exception as e:
        checks.append(("Docker", False, f"Not available: {e}"))
    
    try:
        import langgraph
        checks.append(("langgraph", True, f"v{langgraph.__version__ if hasattr(langgraph, '__version__') else 'installed'}"))
    except ImportError:
        checks.append(("langgraph", False, "Not installed"))
    
    try:
        import langchain_groq
        checks.append(("langchain-groq", True, "Installed"))
    except ImportError:
        checks.append(("langchain-groq", False, "Not installed"))
    
    table = Table(title="Environment Check")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")
    
    for name, ok, details in checks:
        status = "[green]OK[/green]" if ok else "[red]MISSING[/red]"
        table.add_row(name, status, details)
    
    console.print(table)
    
    all_ok = all(ok for _, ok, _ in checks if "optional" not in _.lower())
    if not all_ok:
        console.print("\n[yellow]Some required components are missing.[/yellow]")
        console.print("Run 'pip install -r requirements.txt' to install dependencies.")
        console.print("Copy '.env.example' to '.env' and add your API key.")


@cli.command()
@click.option(
    "--limit", "-n",
    type=int,
    default=5,
    help="Number of logs to show",
)
def logs(limit: int):
    """View recent audit logs."""
    from .utils.logging import AuditLogger
    
    logger = AuditLogger()
    log_files = logger.list_logs()[:limit]
    
    if not log_files:
        console.print("[yellow]No audit logs found.[/yellow]")
        return
    
    table = Table(title=f"Recent Audit Logs (showing {min(limit, len(log_files))})")
    table.add_column("File", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Iterations")
    table.add_column("Created")
    
    for log_file in log_files:
        data = logger.load(log_file)
        table.add_row(
            log_file.name,
            data.get("source", "unknown")[:30],
            str(data.get("total_iterations", 0)),
            data.get("created_at", "")[:19],
        )
    
    console.print(table)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
