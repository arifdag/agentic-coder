"""Print or execute recommended research evaluation command presets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCommand:
    benchmark: str
    max_cases: int
    quality: str
    output_dir: str

    def argv(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "src.main",
            "evaluate",
            "--benchmark",
            self.benchmark,
            "--max-cases",
            str(self.max_cases),
            "--quality",
            self.quality,
            "--output-dir",
            self.output_dir,
        ]


PRESETS: dict[str, list[EvalCommand]] = {
    "smoke": [
        EvalCommand("ult", 1, "fast", "eval_results_smoke"),
        EvalCommand("quixbugs", 1, "fast", "eval_results_smoke"),
    ],
    "paper-lite": [
        EvalCommand("testgeneval_lite", 20, "fast", "eval_results_paper_lite"),
        EvalCommand("projecttest", 20, "fast", "eval_results_paper_lite"),
        EvalCommand("ult", 50, "fast", "eval_results_paper_lite"),
        EvalCommand("quixbugs", 20, "full", "eval_results_paper_lite"),
        EvalCommand("security", 20, "fast", "eval_results_paper_lite"),
        EvalCommand("dep_hallucination", 20, "fast", "eval_results_paper_lite"),
    ],
    "paper-full": [
        EvalCommand("testgeneval", 1210, "fast", "eval_results_paper_full"),
        EvalCommand("projecttest", 60, "fast", "eval_results_paper_full"),
        EvalCommand("ult", 200, "fast", "eval_results_paper_full"),
        EvalCommand("quixbugs", 40, "full", "eval_results_paper_full"),
        EvalCommand("security", 50, "fast", "eval_results_paper_full"),
        EvalCommand("dep_hallucination", 50, "fast", "eval_results_paper_full"),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run commands. Without this flag, commands are printed only.",
    )
    args = parser.parse_args()

    commands = PRESETS[args.preset]
    for cmd in commands:
        argv = cmd.argv()
        print(" ".join(argv))
        if args.execute:
            completed = subprocess.run(argv, check=False)
            if completed.returncode != 0:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
