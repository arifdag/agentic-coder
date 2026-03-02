"""Audit logging for the LLM Agent Platform."""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any
from pydantic import BaseModel


class AuditLogger:
    """Logger for audit trail of pipeline executions."""
    
    def __init__(self, log_dir: str = "audit_logs"):
        """Initialize the audit logger.
        
        Args:
            log_dir: Directory to store audit logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_filename(self, source_identifier: str) -> str:
        """Generate a unique filename for the audit log.
        
        Args:
            source_identifier: Identifier for the source (e.g., file path)
            
        Returns:
            Unique filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = source_identifier.replace("/", "_").replace("\\", "_").replace(".", "_")
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
        return f"{timestamp}_{safe_name}.json"
    
    def save(
        self,
        audit_entries: List[dict],
        source_identifier: str,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Save audit log to file.
        
        Args:
            audit_entries: List of audit entries from pipeline
            source_identifier: Identifier for the source
            metadata: Optional additional metadata
            
        Returns:
            Path to saved log file
        """
        filename = self._generate_filename(source_identifier)
        filepath = self.log_dir / filename
        
        log_data = {
            "source": source_identifier,
            "created_at": datetime.now().isoformat(),
            "total_iterations": len(audit_entries),
            "metadata": metadata or {},
            "entries": audit_entries,
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, default=str)
        
        return filepath
    
    def load(self, filepath: Path) -> dict:
        """Load audit log from file.
        
        Args:
            filepath: Path to log file
            
        Returns:
            Loaded log data
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_logs(self) -> List[Path]:
        """List all audit log files.
        
        Returns:
            List of log file paths
        """
        return sorted(self.log_dir.glob("*.json"), reverse=True)
    
    def get_latest(self, n: int = 5) -> List[dict]:
        """Get the latest n audit logs.
        
        Args:
            n: Number of logs to retrieve
            
        Returns:
            List of log data dictionaries
        """
        logs = self.list_logs()[:n]
        return [self.load(log) for log in logs]


class ConsoleLogger:
    """Console logger for verbose output during pipeline execution."""
    
    def __init__(self, verbose: bool = False):
        """Initialize console logger.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self._start_time = None
    
    def start(self, message: str):
        """Log start of operation.
        
        Args:
            message: Start message
        """
        self._start_time = datetime.now()
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[START] {message}")
            print(f"{'='*60}")
    
    def step(self, step_name: str, details: Optional[str] = None):
        """Log a pipeline step.
        
        Args:
            step_name: Name of the step
            details: Optional details
        """
        if self.verbose:
            print(f"\n[STEP] {step_name}")
            if details:
                print(f"       {details}")
    
    def success(self, message: str):
        """Log success message.
        
        Args:
            message: Success message
        """
        if self.verbose:
            print(f"\n[SUCCESS] {message}")
    
    def error(self, message: str, details: Optional[str] = None):
        """Log error message.
        
        Args:
            message: Error message
            details: Optional error details
        """
        if self.verbose:
            print(f"\n[ERROR] {message}")
            if details:
                print(f"        {details}")
    
    def info(self, message: str):
        """Log info message.
        
        Args:
            message: Info message
        """
        if self.verbose:
            print(f"[INFO] {message}")
    
    def end(self, success: bool, summary: Optional[str] = None):
        """Log end of operation.
        
        Args:
            success: Whether operation succeeded
            summary: Optional summary
        """
        if self.verbose:
            elapsed = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
            status = "COMPLETED" if success else "FAILED"
            print(f"\n{'='*60}")
            print(f"[{status}] Elapsed time: {elapsed:.2f}s")
            if summary:
                print(f"Summary: {summary}")
            print(f"{'='*60}\n")
