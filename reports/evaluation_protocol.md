# Evaluation Protocol For LLM-Backed Test Generation

## Measurement Goals

Case-pass remains the system acceptance metric: a case passes only when the active gates and sandbox execution accept the generated tests. It is not enough by itself, so the evaluation also reports metrics that measure generated-test quality.

Primary quality metrics:

- Mutation score: killed mutants divided by executable mutants.
- Target line and branch coverage: how much of the requested target is actually exercised.
- Coverage gain: coverage added over the benchmark baseline when the dataset provides one.
- Bug detection rate: generated tests pass on fixed code and fail on buggy code.

Supporting metrics:

- Test-pass and syntax/build pass for executability.
- Relevance pass, target-reference rate, and gaming rate for anti-gaming analysis.
- Assertion presence and assertions per test for oracle strength.
- Reliability and flakiness from repeated sandbox runs.
- SAST catch rate and dependency hallucination detection for verification-gate quality.
- Average iterations, runtime, and cost for efficiency.

## Benchmarks

Main results should use:

- TestGenEvalLite/TestGenEval as the primary external real-world Python test-generation benchmark.
- ProjectTest for project-level unit test generation across practical codebases.
- ULT or the current internal benchmark for continuity with existing results.

Quality analysis should use:

- QuixBugs for fixed-vs-buggy bug detection in Python.
- A bounded mutation subset using `--quality full`.
- Relevance and gaming analysis on all benchmark outputs.

Safety and reliability analysis should use:

- CWEval and the custom security suite for SAST behavior.
- The dependency hallucination benchmark for phantom package detection.
- Repeated-run reliability on a selected passing subset.

## Comparisons And Baselines

Report the following systems on the same case subsets where possible:

- Pynguin baseline for Python, when installed and runnable.
- Bare LLM with no repair.
- GDR without gates.
- Full GDR with SAST, dependency, judge, sandbox, and retries.
- Full GDR plus relevance gate.

The main thesis table should include case-pass, test-pass, target branch coverage, coverage gain, mutation score, mutation coverage, bug detection rate, relevance pass, gaming rate, assertion presence, average iterations, runtime, and cost when audit logs are available.

## Ablations

Use controlled ablations to isolate which system parts improve acceptance and quality:

- Retry budget: `k=0,1,3,5`.
- SAST gate on/off.
- Dependency gate on/off.
- Semantic judge on/off.
- Relevance gate on/off.
- Prompt version old/new, if old prompt results are preserved.

Do not compare contaminated runs directly. Report provider errors separately and use clean pass rate for headline claims when rate limits or provider outages occur.

## Reporting Rule

The framing should be:

> Case-pass measures whether the pipeline accepted a generated test suite. Test quality is measured separately using target coverage, coverage gain, mutation score, and bug detection rate. This separates system reliability from the generated tests' ability to verify target behavior and catch faults.
