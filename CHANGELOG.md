# Changelog

All notable changes to Paper Agent are documented here.

## [Unreleased]

### Added

- Independent `legacy`/`structured` report generation, defaulting to structured output rendered locally without duplicating the entire report in model output. Preserve complete input notes and strict references; allow at most one bounded format/length recovery. Final three paired fixed-note trials produced 3/3 valid reports versus 0/3 legacy, at median 9.27 versus 22.81 seconds; this is report availability, not semantic accuracy or end-to-end speed. See `docs/综合报告修复与对照实验_20261116.md`.
- Download/fulltext budget controls with default (45/180/180 seconds), slow-network (90/240/600 seconds) and explicit custom choices. Edits become effective for new tasks after saving; existing task snapshots retain their settings. Keep rank-ordered batch replacement as the production strategy.
- Independent source, candidate-ranking, PDF-parser and single-paper-analysis controls, preserving compatible CLI profiles and recorded baselines. Custom timeout persistence and preset switching are covered through the actual desktop bridge.
- Optional `grounded` profile: retain the entire supplied source, insert paragraph locators and request evidence before the answer in one call, with exact quote checks and unchanged acquisition scope. Engineering checks pass; live quality validation is pending because initial generation requests failed with provider HTTP 500. Keep existing profiles and defaults.
- Paper-disjoint, preregistered quality comparison with all QASPER annotation types, frozen prompt/code/input hashes, source-bound quote checks and a stop after three consecutive provider/network failures. Service failures remain visible and do not count as evidence of method quality.
- An optional `evidence_fast` comparison profile using BM25 top-40 passage retrieval followed by the existing Cross-Encoder, parent recovery and source-quote validation. Preserve the whole-paper and hybrid profiles; a separate 98-paper holdout, isolated latency benchmark and 72 live QA calls record both gains and regressions.
- An optional evidence comparison profile with local hybrid passage retrieval, parent-paragraph recovery, bounded context and exact source quote locations. QASPER held-out measurements and paired live answers retain raw inputs, model usage, negative latency results and source-scope limits.
- Evidence output validation and original-passage recovery for valid body locators missing from the model's separate citation list, with regression and real MCP transport coverage.

- Baseline, source-only, reranking-only, layout-only and combined comparison profiles in desktop settings and CLI. Capture secret-free settings/input fingerprints per task; changes apply to new tasks, and old results retain their own profile.
- A comparison entry point that preserves frozen inputs, raw measurements and per-parser outputs in a fresh bundle, with explicit historical replay versus free local re-execution labels.
- CPU layout-aware whole-PDF extraction using the already pinned PyMuPDF4LLM 1.28.2, preserving page provenance and whole-paper single-call analysis; explicit legacy selection and audited exception fallback remain available.
- An open-source mechanism comparison and a reproducible local PDF parser comparison. Three public PDFs, two alternating-order pairs each, showed 56.6%, 15.4% and 21.9% lower median extraction time; this is not an end-to-end or LLM accuracy benchmark.
- A local Cross-Encoder candidate-ranking experiment based on the existing Sentence Transformers dependency, with fixed revision, bounded candidate pool, input-window audit and baseline-preserving failures.
- Optional free Crossref journal/conference metadata search inside the existing MCP pipeline, with independent public-pool pacing, bounded retries/timeouts and provider provenance.
- Source health, returned counts and deduplicated exclusive-candidate counts in execution records; additive source failure does not block core-source results.
- A public metadata validation script that uses no saved keys or LLM calls and retains observed records for reproducible comparisons.
- Disabled-by-default reference-value reranking with normalized citation impact, publication age and MMR-style redundancy control; preserve baseline Top-k anchors and all component scores. Small candidate-stage comparisons did not demonstrate an adoption-worthy gain.
- Disabled-by-default long-paper section analysis with complete extracted-text coverage, shared context, bounded original-context backreading and a shared request concurrency limit.
- A reviewed live-comparison script with request reservations, persistent budget ledger, isolated strategy artifacts and overwrite protection. Scheduling fixtures are not speed or quality evidence.

### Experiment decision

- Cascade retrieval reduced isolated warm-model new-document preparation time by 52.8% against hybrid retrieval. New 98-paper evidence recall was 89.11% versus 88.30%, with an inconclusive paired interval. Correction: the old QA scorer omitted Unanswerable references; rescoring unchanged outputs with the official protocol gives cascade 57.81%, hybrid 60.72%, wholetext 54.02%. The earlier claim that both retrieval methods scored below wholetext is withdrawn. Repeated cached hybrid retrieval was faster. Keep this as an opt-in comparison, not a universal speed or quality claim. See `docs/级联证据检索与版本缓存实测_20261107.md`.
- A conservative Cross-Encoder that only reorders the original Selector top 20 raised candidate nDCG@5 from 0.719 to 0.807 and 0.749 to 0.916 on two fixed 20-query sets, at about 1.5 seconds warm CPU inference. Candidate membership is preserved. This is an opt-in strategy; final LLM selection/report gains have not been measured. The earlier 40-candidate attempt and its recall tradeoff remain recorded.
- Three fixed public extracted texts were analyzed with DeepSeek Flash using 15 successful requests: section analysis was 29–51% slower and used 1.75 times the aggregate provider-reported tokens. One pair per paper is a pilot, not an end-to-end benchmark. Manual text checks also found a new parameter conflation and processing details in a section report; retain direct analysis by default and stop further paid repetitions.

### Fixed

- Exclude arXiv source-gate queueing from the per-address download timer while retaining the per-paper and whole-stage limits. Preserve timeout reasons and partial-download byte diagnostics.
- Preserve every QASPER answer annotation, including Unanswerable, in a shared official-protocol scorer. Rescore historical saved outputs into new artifacts without replacing raw answers or legacy measurements; verify normalization and annotation precedence against the official evaluator.
- Isolate explicit arXiv versions in verified PDF caches and standalone Markdown output; reject known version mismatches across redirects. Cached author-keyword extraction uses the same identity and verifies versioned manifests/hashes. Detach known stale revision-derived paths and keywords from returned copies while preserving historical files and unversioned compatibility.
- Keep converted Markdown inside each run so baseline and layout experiments on the same paper cannot overwrite earlier artifacts; reject continuing a new-format staged run with changed controls.
- Use paragraph boundaries when section-analysis inputs have no Markdown headings; retain complete character coverage and keep the experimental strategy disabled by default.
- Keep settings paths within the resizable panel; retain the full-path tooltip and add keyboard-accessible expansion.
- Continue ordered replacements when a PDF has no readable converted body; retain partial bodies, original selections, artifacts and errors.
- Bound replacement attempts (automatic default: twice Top-k) and the whole fulltext stage (default: 180 seconds), with per-paper checkpoints and explicit timeout records.
- Evaluate current paper-report bodies without requiring obsolete summary fields; keep legacy records compatible.

## [1.0.0] - 2026-11-02

### Added

- A visible “并行优化” switch beside the paper count. Off submits sequential execution; on submits bounded parallel execution. The performance panel displays the task's actual execution mode.
- One real sequential/parallel comparison through MainAgent and the official MCP client/server, with a fixed question/query, identical ranked selector input and PDF hashes, isolated directories and disabled caches. Wall time fell from 257.77 s to 141.08 s in this pair (45.27%). See `docs/并行优化真实实验_20261102.md` and its measurement JSON.

### Changed

- Rank all deduplicated candidates with the existing semantic/RRF pipeline, then submit the first 20 complete titles/abstracts to the selection Agent; submit all when fewer than 20 exist. Retain later candidates for ordered PDF replacement.
- Preserve author-provided keywords when explicitly available; keep the raised complete-abstract input budget and bounded batch selection.
- Expose research notes, source Markdown and PDF as peer reading views. Task metrics show semantic Top-k cosine scores, Agent scores and runtime separately; execution details stay under “执行记录”.
- Apply conservative duplicate-finding normalization without merging different numerical, conditional or negative claims.
- Align Python, Electron, renderer and lockfile versions at 1.0.0; API remains v3.

### Validation and actual limits

- Local checks: Python 423 passed / 3 skipped; Ruff, TypeScript and production build passed; frontend 17/17; Electron 18/18; source desktop E2E 18 groups; Mock HTTP desktop E2E 10 groups.
- Both real runs produced three PDFs, three Markdown files, three research notes and a partial report. Validated chunks changed from 16/21 to 14/21; fulltext-scope notes changed from one to zero. The additional quality gate did not pass. Publication follows the user's time-reduction condition, without a claim of equal or improved research quality.
- This is one paired observation, not a repeated statistical benchmark. No installer or macOS build was produced for this release; the GitHub release contains source and experiment records.

## [0.3.0] - 2026-10-28

### Added

- Research briefs with evidence-backed findings, key papers and a research next step, with execution details and additional limitations available on demand.
- In-app PDF.js reading through the existing allowlisted PDF-byte API, including page navigation, zoom, text search and focused reading.

### Fixed

- Prepared full-text records now take precedence over the original selection when associating papers, summaries and PDF/Markdown artifacts; replacement papers remain available in the brief.
- Source-bound chunk evidence and bounded paper synthesis retain verified material and preserve incomplete delivery instead of promoting it to full success.
- Targeted title queries and source handling preserve the real multi-source, Sentence-BERT and LLM-selection workflow.

### Changed

- Python, desktop, renderer and lockfile metadata now use `0.3.0`; API remains v3.
- Future development uses the permanent `dev` branch with pull requests to `main`, without version-named development branches.
- Primary reading surfaces prioritize research content and fold technical statuses, scores, repairs and replacement traces into execution records.

### Validation and known limitations

- PDF page rendering, page turns, search and focused reading were observed with real papers before this metadata-only version update.
- The latest independent research task retrieved 43 unique candidates but produced only one partial-fulltext summary and an invalid-as-complete report. Full-text availability, duplicate findings and finding-to-evidence matching remain unresolved.
- This version update adds no model calls, controlled comparisons, new test-suite runs, packaging, release tags or deployment. See `docs/v0.3.0-交付说明.md` for checks and scope.

## [0.2.2] - 2026-10-24

### Fixed

- Removed the native Electron application menu; common edit, reload/devtools and Quit shortcuts are preserved through a scoped window shortcut handler.
- Research output and user-facing errors now use Simplified Chinese. JSON keys, IDs, enum values, citations and source titles remain unchanged.
- Dock tabs now measure the actual panel width, adapt tab width, scroll horizontally when needed and expose an overflow menu with full titles; split groups, resize and fullscreen re-measure correctly.
- Paper tabs store their owning job ID. Switching conversation or restarting no longer makes a previously opened paper look missing; replacements are addressable through `paperById`.
- Activity records are grouped into expandable cards with redacted details, and empty event/placeholder sections are removed.
- Composer grows from 72px to 190px, shows a character count and code-point limit, allows drafting while a job runs, and displays a live recent-progress stream.
- PDF acquisition uses a 45-second per-candidate budget and a 180-second per-paper total budget, with a 5-second connect timeout and at most eight candidates. Slow candidates fall through to the next candidate instead of exhausting the whole paper.
- Removed default report/activity/settings placeholder tabs; the default Dock contains only the paper tab. v1 layouts migrate by dropping the old placeholder tabs.

### Added

- `frontend/src/app/utils/labels.ts` for Chinese status/stage/tool/kind/error labels.
- `frontend/src/app/utils/activity.ts` and expandable activity cards.
- Dock tab measurement helper and overflow menu; frontend tests for labels/tab metrics/replacement lookup/activity grouping.
- E2E coverage for menu removal, overflow tabs, activity cards, cross-conversation paper tabs, composer growth and the historical Sidebar/Dock regressions.

### Changed

- Version `0.2.2`; API remains v3.
- Prompt language: summary/report/selector prose is requested in Simplified Chinese while evidence quotes and reference titles stay verbatim.
- PDF error messages are Chinese but stable machine codes remain in `ErrorInfo.code`.

### Verification

- 307 Python tests passed, 3 skipped; Ruff passed; TypeScript/build passed; Electron 8/8; frontend 8/8; source desktop E2E 16/16. See `README.md` and `docs/运行与验收.md` for the latest live/packaging evidence.

## [0.2.1] - 2026-10-24

### Fixed

- Sidebar collapse/expand no longer clips the expand control: the rail now has one state source, a 58px layout, and a dedicated footer control that stays inside the rail.
- Closing the right Dock no longer unmounts panel state; unsaved settings, report scroll and paper tabs survive close/reopen.
- Electron now forwards backend error messages (for example API-key and active-job conflicts) instead of showing only HTTP status codes.
- SSE fatal status handling and indefinite recovery polling prevent jobs from being silently stranded.
- Job snapshots merge events by `seq`, avoiding loss of events that arrive while a GET snapshot is in flight.
- The composer no longer steals the empty welcome surface by auto-scrolling to the bottom.
- Replacement papers are listed once, and chat jobs keep access to the research job's artifacts.
- Paper detail now separates full text from the single-paper summary and surfaces scores, ranking reason, PDF/Markdown status and truncation.
- Agent tool errors are returned to the model as recoverable tool feedback instead of immediately aborting the run.
- Explicit English queries bypass the query planner; duplicate stage errors are de-duplicated; staged MainAgent runs now persist total timing.

### Added

- Light and dark semantic token sets; improved focus, contrast, typography and Markdown readability.
- Motion tokens and reduced-motion handling that keeps spinners visible.
- `scripts/stress_run.py` for labeled 40/40/30/8 profiles through the real MCP path.
- Frontend layout-store tests (`npm run test:frontend`) and E2E regressions for sidebar collapse/expand/rapid toggles/persistence and Dock state.

### Changed

- Version: `0.2.1`; API remains v3.
- Top-k validation is 1–15 and concurrency 1–8 consistently across renderer, preload and backend.
- Research input limit is 12,000 characters across renderer, preload and backend.
- Python runtime version is read from a single `__version__` source.

### Removed

- Stale PySide6/Qt source remnants and regenerable Python caches; user research data, PDFs, reports and `.env` were preserved. See `docs/运行与验收.md#清理执行记录`.

### Verification

- Python: 304 passed, 3 skipped; Ruff passed.
- Desktop: typecheck/build passed; Electron tests 8/8; frontend layout tests 2/2; source E2E 12/12; unpacked v0.2.1 E2E 12/12.
- Live small run: `partial`, 8 physical model requests, 41,501 tokens, report body 7,942 chars.
- Live 40/40/30/8 run: arXiv 40 + OpenAlex 40, 59 unique, 30 downstream, Top-8 readable, 8 summaries (5 partial, 3 failed), 56 physical model requests and 347,389 tokens. A first report regeneration hit the old input budget; after adding an independent report budget and compact evidence projection, the successful retry produced a 2,417-char overview, 13,875-char report and 5 real references. One intermediate attempt was rejected for invalid citations and preserved as `invalid_model_output`.
- Packaging: unpacked app generated with local Electron distribution. GitHub TLS was unavailable, so `winCodeSign` could not be downloaded; executable metadata editing/signing was disabled and installer targets were not rebuilt.
- Demo/mock results are not live quality or performance evidence. No paired ≥5-run benchmark was executed, so no speedup is claimed.

See `README.md` and `docs/运行与验收.md` for the full evidence boundary.
