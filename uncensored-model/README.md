# Abyss — Phase 1

A personal, text-only React + FastAPI chat application. SQLite stores conversations on this computer. Ollama runs one selected local model at a time. No accounts, telemetry, external fonts, CDNs, cloud APIs, attachments, search, or tools are used by the application.

## macOS setup

Requires Python 3.12+, Node.js 22 with npm, and Ollama installed. From the `uncensored-model` folder:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
(cd frontend && npm ci --no-audit --no-fund && npm run build)
```

If you migrated a Windows `.venv` with the project, move that environment aside before creating the macOS one.

Quit the Ollama menu bar app if it is running, then start the local-only server in a terminal:

```bash
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1 OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=4096 ollama serve
```

In another terminal, run `./start.sh` from the project folder and open **http://127.0.0.1:8000**. Abyss discovers installed models through Ollama and performs no model downloads. Node is only needed to build the frontend. Chat history is stored in `data/chat.sqlite3`.

## Windows setup

Requires Python 3.12+, Node.js 22 LTS, and Ollama installed. Dependency installation and the initial model download require internet access; normal chat does not.

In PowerShell, from this folder:

```powershell
.\setup.ps1
ollama pull hf.co/richardyoung/Qwen3-8B-Abliterated-GGUF:Q4_K_M
ollama pull hf.co/mlabonne/gemma-3-4b-it-abliterated-v2-GGUF:Q4_K_M
ollama pull hf.co/tutuchen2000/Qwen3.5-9B-abliterated-GGUF:Q4_K_M
```

If PowerShell blocks local scripts, invoke them with `powershell -ExecutionPolicy Bypass -File .\setup.ps1` (or `start.ps1`). This applies to that process only.

For a runtime with cloud features disabled, quit Ollama from its tray icon and run this in a terminal:

```powershell
.\start-ollama.ps1
```

Then in another terminal:

```powershell
.\start.ps1
```

Open **http://127.0.0.1:8000**. Create a conversation and send a message. Enter sends; Shift+Enter inserts a newline. Rename and Delete apply to the selected conversation. Delete asks for confirmation. Close and reopen the browser or restart the backend: history remains. Ctrl+C stops the backend.

The launch command performs no installation or network downloads. The React build is served by FastAPI; no Node process is needed at runtime. After editing React files, run `npm.cmd run build` inside `frontend`, then restart the backend.

## Selecting and comparing models

Qwen3.5 9B Abliterated is a new candidate, not a declared winner. Its Q4_K_M build is `tutuchen2000/Qwen3.5-9B-abliterated-GGUF`. The previous two models remain available. Quality and acceptance require your own assessment. The app remains text-only. Use an up-to-date Ollama runtime with Qwen3.5 support.

The rename to Abyss preserves your browser model preference by migrating the old storage key once. Conversation storage remains at the same `data/chat.sqlite3` path with no schema changes.

Use **Active text model** in the chat header. Your choice is remembered in this browser and applies to the next message or retry, including in existing conversations. Switching does not load a model until a request is sent. New browsers prefer the configured default when it is installed, then another installed model if available. Edit `default_model` in `models.json` to change that preference.

Click **Compare models** to send one prompt to every installed local model shown in the selector, sequentially. Each gets fresh context with the same shared system instruction and its own configured generation settings. The panel streams each response and shows total time and time to first token, including model preparation/loading. Score Acceptance, Prompt understanding, Reasoning quality, Hallucination / factual reliability, and Response speed from 1 (poor) to 10 (excellent). Higher factual reliability means fewer hallucinations. Leave any category unrated if you cannot assess it. These are your assessments, not automatic factual verification; Abyss does not calculate a winner or change your selected model based on ratings. Use a known-answer prompt when possible. Lower temperature and an honesty instruction do not guarantee accurate answers or eliminate refusals.

Comparison prompts, responses, and ratings stay in page memory, including while returning to chat, and are cleared by reloading. They are not written to SQLite or external services. Existing conversation records and database columns are unchanged. Historical messages are labeled “Assistant”; model provenance is not retroactively assigned or stored. Use separate, clearly named conversations if you want a lasting model comparison record.

All registered models use temperature `0.4`, top_p `0.9`, context `4096`, and maximum output `1024`. Edit their separate `options` in **models.json**, then restart the backend. Add another entry with a unique `id`, friendly `displayName`, exact installed Ollama `ollamaModel`, `no_think` flag, and generation `options` to customize an installed model. Abyss also discovers local models without a registry entry and uses the exact tag as the label with the default generation settings. Copy an existing entry as a template. Configuration is validated at startup. `no_think` is the legacy Qwen3 prompt switch and stays false for Gemma and Qwen3.5. Qwen3.5 uses native `think: false` instead; it does not support the old /no_think soft switch. The shared `system_instruction` encourages direct answers, factual restraint, explicit uncertainty, and honesty about absent tools/files/web access.

Download weights explicitly with Ollama before testing; the app never downloads models. Click **Refresh models** after installing one. Uninstalled entries are labeled and excluded from comparisons. The model dropdown never sends an arbitrary user-supplied runtime URL.

The app uses a single shared generation lock for normal chat and comparisons. Before generation it unloads other Ollama models and verifies they have been released. It sets `keep_alive: 0` so the selected model is released after its response. This favors low VRAM usage and consistent cold-run comparisons over warm-model speed. Keep using `start-ollama.ps1` (one model, one parallel request), and avoid running another inference client against that same Ollama process during comparisons.

## Privacy and storage

- Backend binds to `127.0.0.1:8000`; runtime launcher binds Ollama to `127.0.0.1:11434` and sets `OLLAMA_NO_CLOUD=1`.
- The model adapter has a fixed local URL and accepts only registry model IDs. It checks installed local weight metadata before sending prompts, rejects remote-model metadata, ignores HTTP proxy environment variables, follows no redirects, and never pulls models or falls back to a cloud service.
- Frontend fetches only same-origin `/api` endpoints. Content Security Policy blocks external resources and connections. Cross-origin/cross-site requests and unexpected Host headers are rejected. Model text is rendered as escaped plain text, never executable HTML.
- Application data is in `data/chat.sqlite3` (plus SQLite WAL/SHM files while running). This is ordinary unencrypted local storage. Stop the backend before copying the `data` folder for backup. Deletion removes records, not a forensic secure erase.
- No prompt logging or analytics. Ollama is a separate installed program; its Windows app may check for updates. Use the provided standalone runtime launcher and disable cloud features; the application itself makes no external requests. An OS firewall can enforce machine-wide offline operation.
- There is deliberately no authentication. Run only on your trusted personal machine and keep the localhost binding.

## Hardware defaults and limits

Qwen3 8B Q4_K_M weights are approximately 5 GB; Gemma Q4_K_M weights approximately 2.5 GB; the new Qwen3.5 9B Q4_K_M candidate approximately 5.6 GB. A 6 GB GPU may offload some layers to system RAM; actual speed depends on free VRAM and other applications. The app requests a 4096-token context and up to 1024 generated tokens, with Qwen thinking disabled. No GPU layer count is forced: Ollama automatically places layers across GPU and CPU/system RAM as needed. Keep other large GPU workloads closed on the 6 GB laptop. The Ollama launcher sets one loaded model and one parallel request. Use `ollama ps` to inspect GPU/CPU placement.

Messages are limited to 2500 UTF-8 bytes. A conservative 2800-byte recent-turn budget leaves space for output and template tokens. Older turns remain visible and saved but may be outside the model's context. This intentionally favors predictable memory use over long-context chat. Generation options live in `models.json`; context selection lives in `backend/runtime.py`.

Only one response generates at a time. Deltas are committed as they arrive. A closed connection or runtime error preserves the partial answer; restart marks unfinished streams interrupted. Retry replaces the incomplete assistant response without duplicating the user message. A backend process should run with **one worker**, as configured by `main.py`.

## Code layout and Phase 2 boundary

- `frontend/src/main.jsx`: conversation UI and incremental NDJSON stream reader.
- `frontend/src/Comparison.jsx`: transient comparison results and manual ratings.
- `models.json` and `backend/models.py`: model registry, settings, and validation.
- `backend/app.py`: validated API, response lifecycle, local request policy, static hosting.
- `backend/storage.py`: SQLite schema and persistence; schema version starts at 1.
- `backend/runtime.py`: isolated model adapter and context selection.

Phase 2 can add an attachments table keyed to message IDs, attachment UI, and a separate vision adapter alongside the text adapter. The existing conversation/message IDs and storage API can stay intact. No attachment handling or vision model loading is implemented now.

The previous Transformers entry point was preserved as `legacy_transformers_demo.py`; it is not used by this application and its dependencies are no longer required.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Push-Location frontend
npm.cmd run build
Pop-Location
```

Tests use a fake runtime and temporary SQLite files to verify streaming events, persistence across app instances, rename/delete and cascade behavior, failure/retry, recovery, validation, local request protection, generation concurrency, context trimming, and Ollama NDJSON parsing. They require no model download.

With the app and model running, `.\.venv\Scripts\python.exe tests/smoke_local.py` verifies real incremental generation, stored output, rename, and deletion using a temporary test conversation. With Ollama running, `.\.venv\Scripts\python.exe -m tests.smoke_models` runs the real comparison API using a separate temporary database and monitors the number of loaded models. Automated tests cover model selection, preserved history, comparison isolation, per-model options, unloading, and remote-model rejection as well as the original chat behavior.

The supplied GGUF template opens a thinking block even with `think: false`; the adapter also uses Qwen's documented `/no_think` system switch and removes an empty leading closing tag. See [Qwen's thinking controls](https://github.com/QwenLM/Qwen3/blob/main/docs/source/inference/transformers.md).

Manual acceptance: start Ollama and the app, create a conversation, send a short message, observe streamed output, rename the conversation, stop/restart the backend, verify history, then delete the test conversation.

References: [Qwen model and Ollama tag](https://huggingface.co/richardyoung/Qwen3-8B-Abliterated-GGUF), [Gemma v2 GGUF model](https://huggingface.co/mlabonne/gemma-3-4b-it-abliterated-v2-GGUF), [Ollama chat API](https://docs.ollama.com/api/chat), [Ollama configuration and cloud disable setting](https://docs.ollama.com/faq).

Qwen3.5 references: [candidate GGUF and quantizations](https://huggingface.co/tutuchen2000/Qwen3.5-9B-abliterated-GGUF), [official base model and thinking controls](https://huggingface.co/Qwen/Qwen3.5-9B).

### Verification of the Abyss update

The installed Qwen3.5 entry uses the exact Windows Ollama name `hf.co/tutuchen2000/Qwen3.5-9B-abliterated-GGUF:Q4_K_M`. Detection compares `ollamaModel` only to each `name` returned by the local `/api/tags` endpoint; generation uses that same exact string. The live local runtime reports this model installed. Seventeen automated tests pass, including absent-to-installed-to-removed refresh behavior and rejection of friendly-name matches. No database migration was made. Real generation with this replacement build has not been benchmarked.

The model list refreshes at startup, via **Refresh Models**, when returning to the browser tab, and every 15 seconds while visible. Each check reads Ollama again with browser caching disabled. After installing a model outside Abyss, return to the app or press Refresh Models; a backend restart is only required for registry/code changes, not subsequent installs.

