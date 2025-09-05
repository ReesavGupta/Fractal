## Project: Coding Agent — Detailed Task List

Purpose
- A comprehensive, phase‑based task list to implement a production‑grade coding agent similar to Devseeker: generation, apply diffs, execute, improve, and self‑heal. Tasks include design, implementation, testing, security, and ops.

Assumptions
- Primary language: Python (but language‑agnostic design). 
- LLM provider(s) behind an AI adapter (OpenAI, Anthropic, local LLMs).
- Local execution sandbox for running generated code (disk/container).
- Git is available for optional file history and patches.

Checklist (requirements extracted)
- Create `tasklist.md` in `Fractal/Fractal/docs` (Done)
- Create `prd.md` in `Fractal/Fractal/docs` (Done)
- Produce deep, phase‑based tasks covering design, implementation, testing, security, and deployment (Done)

High‑level phases (each phase contains detailed tasks and acceptance criteria)

Phase A — Discovery & Architecture (1–2 weeks)
1. Define product goals and success metrics
   - Task: Create one‑page goals doc with measurable KPIs (time‑to‑first‑working‑project, patch apply success rate, execution safety incidents). 
   - Deliverable: `goals.md` and KPI dashboard plan.
   - Acceptance: KPIs reviewed by stakeholders.

2. Identify core abstractions and public interfaces
   - Task: Draft interfaces for AI adapter, Memory, ExecutionEnv, FilesDict, Agent, Step pipeline, Diff engine, FileStore, and Telemetry. 
   - Deliverable: Interface contract doc (methods, inputs/outputs, errors). 
   - Acceptance: Interfaces approved by lead engineer.

3. Security & threat model
   - Task: Create threat model focusing on untrusted code execution, LLM prompt injection, secrets exfiltration.
   - Deliverable: Security doc with mitigations (resource limits, network egress controls, secret redaction). 
   - Acceptance: Security signoff (devops/security).

Phase B — Core Data Models & Contracts (1 week)
1. Files representation (FilesDict)
   - Task: Specify schema for `FilesDict`: file path, metadata (executable, language), contents, version/hash, source (generated/human). 
   - Deliverable: `FilesDict.md` with JSON schema and invariants.
   - Acceptance: Conforms with diff/hunk APIs.

2. Diff & Hunk model
   - Task: Define canonical diff/hunk representation (hunk id, file path, start/end lines, patch text, metadata, context). 
   - Deliverable: `diff_schema.md` with examples.
   - Acceptance: Parser/serializer contracts are clear.

3. Chat → Files contract
   - Task: Define how model responses are parsed into file updates and metadata (language tags, format tokens, dedup rules). 
   - Deliverable: `chat_to_files_contract.md` with examples of valid model outputs.

Phase C — LLM Adapter & Prompting (1–2 weeks)
1. AI adapter interface implementation
   - Task: Implement an adapter standardizing send/receive, retries, rate‑limit handling, token accounting, and streaming. 
   - Deliverable: Adapter design doc listing supported providers and fallback/resilience strategies.
   - Acceptance: Adapter can swap providers via configuration.

2. Prompt engineering library
   - Task: Build modular prompt templates, prompt interpolation, deterministic tokens for functions like gen_code, improve, clarify. 
   - Deliverable: Template set + tests ensuring deterministic sections are present.
   - Acceptance: Prompt safety checks (no secret leakage), versioned templates.

Phase D — Diff Parsing & Application (2–3 weeks)
1. Robust diff parser
   - Task: Build parser for common model output formats: unified diff blocks, fenced code, file blocks. Must handle malformed diffs gracefully.
   - Subtasks: implement tolerant parsing, hunk extraction, recovery heuristics for missing context.
   - Deliverable: Parser with unit tests and fuzz tests.
   - Acceptance: Parser handles 95% of real model samples from dataset.

2. Apply & validate hunks
   - Task: Implement atomic application of hunks to `FilesDict` with validation; provide rollback on failure; salvage partial hunks when possible.
   - Deliverable: apply_hunks API with deterministic outcomes and detailed failure reasons.
   - Acceptance: Patch apply success rate and clear failure messages.

3. Conflict resolution & salvage
   - Task: Heuristics for mismatched contexts, auto‑alignment, and human fallbacks (open editor with suggested hunks).
   - Deliverable: salvage strategy doc and UI interaction spec.

Phase E — Execution Environment & Self‑heal (2–3 weeks)
1. Disk execution environment (v1)
   - Task: Implement local disk runner that: prepares workspace, installs dependencies (optional, with sandboxing), runs tests/entrypoint, captures stdout/stderr, exit codes, tracebacks.
   - Deliverable: ExecutionEnv API, runner logs, reproducible artifacts (core dumps/traceback). 
   - Acceptance: Runner executes sample entrypoints and returns structured results.

2. Self‑heal loop & error classification
   - Task: Define and implement error classifiers (syntax, runtime, dependency, incorrect I/O). Map classes to automated repair strategies.
   - Deliverable: classifier + rulebook mapping error classes -> repair prompt template and retry strategy.
   - Acceptance: Self‑heal reduces human fixes by X% in testbed.

3. Safety controls
   - Task: Enforce timeouts, CPU/memory cgroups or container limits; prevent external network access by default.
   - Deliverable: security policy and enforcement mechanism documented.

Phase F — Agent & Steps Pipeline (2–4 weeks)
1. Steps orchestration engine
   - Task: Implement pipeline runner that composes step functions: gen_code, gen_entrypoint, execute_entrypoint, improve_loop, apply_diffs. Steps declare inputs/outputs and errors.
   - Deliverable: step runner with transactional behavior and retry semantics.
   - Acceptance: The pipeline can be executed end‑to‑end in a sandbox.

2. Agent orchestration (CliAgent equivalent)
   - Task: Build agent that sequences modal behaviors (generate, improve, clarify, self_heal) and manages state (memory, file store, logs). 
   - Deliverable: Agent design doc and behavior tests for each mode.

Phase G — CLI & UX (1–2 weeks)
1. CLI design & argument schema
   - Task: Design CLI flags, environment variables, config files, interactive vs noninteractive modes, file selector flow.
   - Deliverable: CLI UX spec and CLI help text.
   - Acceptance: CLI usability review.

2. Interactive file selector
   - Task: Implement file selection UI: tree view, filters, ranges, and TOML file for selection persistence.
   - Deliverable: file selector UX and integration tests.

Phase H — Persistence, Memory & Telemetry (1–2 weeks)
1. DiskMemory design
   - Task: Design chat/session persistence, versioned archives, token usage logs, and query APIs.
   - Deliverable: data retention and purge policy.

2. Telemetry & human reviews
   - Task: Collect structured human feedback, schema for review payloads, and secure optional upload pipeline.
   - Deliverable: Telemetry schema and sender module (opt-in by default).

Phase I — Testing, CI, and Quality (ongoing)
1. Unit & integration tests
   - Task: Tests for parser, apply_hunks, FilesDict, ExecutionEnv, and agent flows. Include deterministic test vectors for LLM outputs (golden responses). 
   - Deliverable: test matrix and coverage targets (e.g., 80% core modules).

2. Fuzzing & adversarial model outputs
   - Task: Create fuzz corpus for malformed diffs and adversarial prompt outputs to stress parser and apply logic.
   - Deliverable: fuzz harness and CI integration.

3. End‑to‑end scenarios
   - Task: E2E tests running the pipeline on small example projects (succeeds, fails with self‑heal, requires manual review). 
   - Deliverable: e2e job in CI with ephemeral runners and sandboxing.

Phase J — Packaging, Deployment & Ops (1–2 weeks)
1. Packaging & distribution
   - Task: Package as a CLI tool (pip wheel) and optionally Docker image for consistent runtime. 
   - Deliverable: release artifacts and install instructions.

2. CI / CD
   - Task: Setup CI pipelines (lint, tests, security checks). Create release pipeline for packaging and GitHub releases.

Phase K — Docs, Examples & Community (1–2 weeks)
1. Tutorial projects & templates
   - Task: Add sample projects (simple app, web app, data script) and step‑by‑step walkthroughs.

2. API & architecture docs
   - Task: Publish interface docs: FilesDict, diff formats, AI adapter contract.

Backlog & optional advanced tasks
- Multi‑LLM orchestration and voting
- Distributed execution (remote sandbox pool)
- Plugin architecture for language-specific generators
- GUI for file selection and diff review

Risk register (short)
- Untrusted code execution risk — mitigate via strict sandboxing and least privilege.
- LLM output correctness — mitigate via multi‑step verification, tests, and human review.
- Parser brittleness — mitigate with robust heuristics and fallback human workflows.

Estimated timeline (single team of 2–4 engineers)
- MVP (core pipeline, parser, disk runner, simple agent, CLI): 6–10 weeks
- Production (safety, telemetry, tests, packaging): 12–20 weeks

Acceptance criteria (MVP)
1. The system can generate a set of files from a prompt and apply them to a working workspace.
2. The system can run an entrypoint and capture structured errors and logs.
3. The system can parse and apply >90% of diffs produced by a reference LLM sample corpus.

---
## Task board template (epics -> stories -> tasks)
... (see above for pattern)
