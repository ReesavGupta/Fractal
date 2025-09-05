# Product Requirements Document (PRD) — Coding Agent

Document purpose
- This PRD describes the architecture, components, data contracts, flows, and non‑functional requirements for a production coding agent (generation → diff application → execution → improve → self‑heal). The document is intentionally technical and omits code, focusing on interfaces, data shapes, algorithms, and operational constraints.

1. Overview
- Problem statement: Developers want a tool that can generate, modify, and iteratively improve code with minimal manual editing. The agent must accept a developer prompt, generate code or patches, apply them to a project workspace, execute and test the result, and optionally iterate on failures.
- High‑level flow: User prompt -> Agent orchestrator -> LLM(s) -> Chat parsing -> Files/diffs -> Apply -> Execute -> Feedback loop (improve/self‑heal/human review).

2. Goals & Success Metrics
- Goals
  - Reliable diffs: >90% of model diffs apply cleanly after salvage heuristics.
  - Safe execution: No default network egress from sandbox, resource limits enforced.
  - Developer productivity: Mean time from prompt to runnable project < 10 minutes for small projects.
- Metrics
  - Patch apply success rate
  - Self‑heal success rate
  - Time to first successful run
  - Human review frequency per 100 runs

3. System Components & Interfaces

3.1 AI Adapter (Interface)
- Responsibility: Abstract provider details (HTTP SDK, streaming) and provide a uniform API for sending messages and receiving responses, including chunked/streaming responses.
- Inputs: messages: List[Message], options: {model, temperature, stream, max_tokens}
- Outputs: Response object: {id, choices:[{role, content, delta?, finish_reason}], usage: {prompt_tokens, completion_tokens, total_tokens}}
- Errors: RateLimitError, ProviderError, TimeoutError
- Features: retries with exponential backoff, provider failover, token accounting, deterministic seeding for test harnesses.

3.2 Prompt & Template Manager
- Responsibility: Store and render versioned prompt templates. Provide safety checks (no environment secrets in templates), parameter validation, and templated sections for system/meta/user messages.
- Data shapes: Template {id, version, role, content, placeholders, required_meta}

3.3 FilesDict (Canonical file set)
- Responsibility: In‑memory canonical model of a project's file tree and metadata. FilesDict is authoritative during a pipeline run until persisted.
- Schema (example JSON):
  - files: [{path: str, content: str, hash: str, mode: str, language: str, source: enum(generated,human,external), metadata: {executable: bool}}]
  - workspace_root: str
  - version: uuid
- Operations: read(path), write(path, content, metadata), apply_patch(patch), diff_with_disk(), serialize()/deserialize().

3.4 Diff & Hunk Model
- Responsibility: Represent incremental edits from LLMs as structured hunks and provide deterministic application semantics.
- Hunk schema:
  - hunk_id: uuid
  - file_path: str
  - range_start: int
  - range_end: int
  - patch_text: str (unified diff or replacement block)
  - context: optional lines
  - confidence: optional float
- Application semantics: atomic per hunk with precondition checks on context or hash. On failure, return detailed diagnostics.

3.5 Chat→Files Parser
- Responsibility: Convert raw LLM outputs into a set of hunks/files, with robust heuristics for malformed output.
- Steps:
  1. Identify file blocks and inline diffs (regex and heuristic rules).
  2. Normalize line endings and language fences.
  3. Create FilesDict or Hunk objects.
  4. Validate for duplication or conflicting writes.
- Failure modes: invalid syntax, missing file path; actions: salvage heuristics or request clarification.

3.6 Diff Engine (Apply/Validate)
- Responsibility: Apply hunks to FilesDict with transaction semantics.
- Algorithm outline:
  - Group hunks by file.
  - For each file, create a working copy.
  - For each hunk: check preconditions (hash, context), attempt apply using fuzzy alignment when exact match fails.
  - On partial failure: attempt salvage by expanding context or asking the model for a repair.
  - Commit if all hunks succeed; otherwise rollback and surface diagnostics.

3.7 Execution Environment (ExecutionEnv)
- Responsibility: Safely run entrypoints (scripts, tests, binaries) and collect structured execution artifacts.
- Interface:
  - prepare_workspace(files_dict): ensures files on disk in a sandbox
  - install_dependencies(spec): optional, with timeouts and network policy
  - run(command, timeout, capture_output=True): returns ExecutionResult
  - ExecutionResult: {exit_code, stdout, stderr, duration_ms, artifacts: [paths], resource_usage: {cpu_ms, mem_bytes}}
- Isolation requirements: namespace isolation (containers/VMs), cgroups or job objects, blocked network egress by default, ephemeral storage.

3.8 Agent Orchestrator
- Responsibility: Implement top‑level flows (generate→apply→execute→improve) and maintain run state, memory, telemetry.
- State model: Session {id, files_dict, chat_history, run_attempts, artifacts}
- Modes: generate, improve, clarify, self_heal, dry_run

3.9 Memory & Persistence
- Responsibility: Store chat histories, artifacts, token accounting, and versioned archives.
- Data store options: simple file archives (for local) or object storage + metadata DB for larger deployments.

3.10 Telemetry & Feedback
- Responsibility: Capture structured metrics and optional human reviews. Telemetry must be opt‑in by default and privacy conscious.

4. Flows & Sequence Diagrams (textual)

4.1 Simple generate + apply + run
1. User invokes CLI with prompt and mode=generate.
2. Agent renders gen_code prompt and sends to AI Adapter.
3. AI returns response; Chat→Files Parser converts to hunks/files.
4. Diff Engine applies hunks to working FilesDict; if apply succeeds, persist to disk.
5. ExecutionEnv.prepare_workspace writes files to sandbox and run entrypoint.
6. ExecutionResult returned; if success -> store artifacts and return success; else -> if mode includes self_heal, continue to 4.2; else -> present diagnostics.

4.2 Self‑heal / Improve loop
1. On failure, ExecutionResult is analyzed by error classifier to produce error summary.
2. Agent composes an improve prompt including failing files, stacktrace, tests, and asks LLM for fixes.
3. Repeat Chat→Files → Diff Engine → apply → execute until success or max attempts.
4. If unable to resolve, open human review flow (editor with suggested hunks and diagnostics).

5. Non‑functional requirements
- Performance: typical LLM round trip < 10s (depends on model); diff apply operations should be < 1s for small files.
- Scalability: system must support concurrent runs for multiple users; per‑run sandbox isolation is required.
- Security: default deny network, redact secrets in logs, require explicit opt‑in for telemetry or remote execution.
- Observability: structured logs, trace IDs for runs, and token usage metrics.

6. Data Models (detailed)
- Message: {role, content, metadata}
- FilesDict (see 3.3)
- Hunk (see 3.4)
- ExecutionResult (see 3.7)

7. Error handling & retries
- LLM errors: retry with backoff and jitter; if repeated, failover to alternate provider or return to user.
- Patch apply errors: escalate to salvage heuristics, attempt fuzzy alignment, or require clarification.
- Execution errors: classify and map to repair templates; for dependency errors, optionally attempt dependency installation in offline cache.

8. Testing strategy
- Unit tests: parser, diff engine, FilesDict ops, error classifier.
- Integration tests: end‑to‑end flows with a mock LLM and sample projects.
- Fuzz tests: adversarial model outputs and malformed diffs.
- Security tests: attempt to break sandbox via crafted code and ensure isolation.

9. Operational considerations
- Deployment: CLI for local use; server mode for multi‑user deployments behind auth.
- Backups & retention: store artifacts for a configurable retention period; provide export for human review.
- Secrets: never persist API keys in run artifacts; require environment variables and ephemeral credentials.

10. Roadmap & phased rollout
- Phase 0: internal MVP (local only, single LLM, disk runner)
- Phase 1: safety hardening (sandboxing, telemetry opt‑in)
- Phase 2: provider diversification and failover
- Phase 3: scale (multi‑tenant service, RBAC, remote sandboxes)

11. Appendices
- A. Example model prompt fragments (roles & deterministic markers)
- B. Patch application diagnostics (schema for error messages)
- C. Telemetry schema (events and fields)

---

End of PRD.
