# pixelgen web app

A local web app to try the trained models: type a short description, pick one or both
models and get 16×16 pixel-art images side by side.

- **Backend**: FastAPI (`app/backend`) — loads the trained models and serves a small JSON API.
- **Web UI**: plain HTML, CSS and ES modules (`app/web`) — **no npm, no build step**; the API
  serves it, so one command runs everything.
- **Optional Angular UI** (`app/frontend`): the same interface as an Angular app, for anyone
  with Node and npm.

Everything runs on your machine; nothing is hosted.

## Run it

All commands are run from the **project root** (the MLflow database and the dataset paths
stored in each run are relative to it).

### 1. Install

```bash
pip install -e ".[dev,app]"
```

### 2. Choose the models

`app/backend/models.yaml` lists the MLflow runs to serve (a run needs its `checkpoint.pt`
artifact). Edit the `run_id`s to serve other runs.

### 3. Start

```bash
make api
# or: python -m uvicorn --app-dir app/backend pixelgen_api.main:app --port 8000
```

Open **http://localhost:8000**. The API loads the models at startup, so the first start takes
a few seconds. Interactive API docs: http://localhost:8000/docs.

### 4. Test

```bash
make app-test
```

This runs the backend tests (fake models, so no torch or checkpoints needed) and the unit
tests of the web UI logic with Node's built-in test runner (Node 20+; no packages to install).

### Optional: the Angular UI

```bash
cd app/frontend && npm install
npm start                       # http://localhost:4200, proxies /api to :8000
npm run build                   # then serve it from the API:
PIXELGEN_STATIC_DIR=app/frontend/dist/pixelgen-app/browser make api
```

Angular 20 needs Node 20.19+ (or 22.12+).

## Architecture

```mermaid
flowchart LR
    subgraph Browser["Web UI (app/web)"]
        C["ui/*<br/>prompt, settings, results"] --> S["core/generator.js<br/>state + actions"]
        S --> A["core/api.js<br/>fetch"]
    end
    A -- "JSON /api/*" --> R

    subgraph API["FastAPI (pixelgen_api)"]
        R["routes.py<br/>thin HTTP layer"] --> G["GenerationService<br/>validate, seed, time"]
        G --> V["PromptValidator"]
        G --> M["ModelRegistry<br/>ModelEntry.make_generator"]
        ST["StaticFiles"] -. "serves" .-> Browser
    end

    M --> L["pixelgen.evaluation.loading.load_run<br/>(MLflow checkpoints)"]
    L --> P["pixelgen models<br/>decoder-only / encoder-decoder"]
```

Layers and responsibilities:

| Layer | File | Responsibility |
|---|---|---|
| HTTP | `routes.py`, `main.py` | Parse requests, map errors to one JSON envelope, serve the web UI. No business logic. |
| Use case | `services/generation.py` | Validate the prompt, seed the sampler, run each model, time it. Depends only on abstractions. |
| Domain rules | `services/prompt.py` | Normalise a prompt and check it against the vocabulary. Pure Python. |
| Catalogue | `services/registry.py` | Describe the served models; the only place that imports torch / mlflow (lazily). |
| Contract | `schemas.py`, `app/web/js/core/api.js` | The JSON shapes shared by both sides. |
| UI state | `app/web/js/core/generator.js` | Single source of truth of the screen. Pure logic, tested with Node without a browser. |
| UI views | `app/web/js/ui/*` | Read the state and update the DOM; call intent-named actions. No HTTP, no storage. |

Design decisions worth knowing:

- **Dependency injection for testability.** `create_app(settings, service)` accepts a ready-made
  service; tests pass one built with fake models, so the HTTP layer is tested without torch.
  In the UI, `createActions({store, api, storage})` takes its collaborators the same way.
- **Models are hidden behind `generate_fn(captions) -> images`.** The same abstraction the
  evaluation scripts use; adding a model means adding an entry to `models.yaml`.
- **Validation on both sides.** The UI mirrors the vocabulary check for instant feedback
  (unknown words are highlighted with suggestions); the server re-validates every request.
- **Reproducibility.** Every response carries the seed that was used. All selected models are
  seeded identically, so comparisons are made under the same sampling conditions.
- **One error envelope.** `{"error": {"code", "message", "details"}}` for domain errors,
  validation errors and unexpected failures; the UI switches on `code`, never on text.
- **Blocking inference stays off the event loop.** `POST /api/generate` is a plain `def`, so
  FastAPI runs it in a worker thread; a lock serialises sampling because PyTorch's random
  state is global.
- **Images travel as palette indices** (`pixels[row][col]`) plus the palette, not as PNGs:
  the payload is tiny and the canvas draws crisp, resolution-independent pixels.
- **No innerHTML.** The UI builds DOM nodes with text nodes only, so prompts and server
  messages can never inject markup.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness probe. |
| GET | `/api/models` | Served models (id, label, kind, parameter count, description). |
| GET | `/api/vocabulary` | Vocabulary, word groups, max length, palette and example prompts. |
| POST | `/api/generate` | `{prompt, models[], n_variants, temperature, seed?}` → images per model. |

Error codes: `empty_prompt`, `prompt_too_long`, `unknown_words` (with `details.unknown` and
`details.suggestions`), `unknown_model`, `invalid_request`, `internal_error`.

## UX notes

- Live feedback while typing: word count, recognised words, unknown words with one-click fixes.
- Word chips grouped by size / color / shape, and examples taken from the real dataset —
  including **held-out combinations** the models never saw in training.
- Side-by-side comparison with timing, seed copy / reuse, and PNG download per image.
- Loading skeletons, a clear empty state, errors with a retry action, and a disabled Generate
  button that always says why.
- Keyboard: `Ctrl/⌘ + Enter` generates; focus rings, labels and live regions for screen
  readers; respects dark mode and reduced motion; works down to phone width.
- The last prompt and settings are remembered in the browser (`localStorage`, optional).
- The stylesheet uses native CSS nesting, so it needs a recent browser (Chrome 120+,
  Safari 17.2+, Firefox 117+).

## Ideas for later

- A tab with the cross-attention maps (`pixelgen.evaluation.attention_maps`) for the
  encoder-decoder.
- Show whether each generated image is a *valid render* of the prompt (`valid_renders`).
- Docker image, if it is ever deployed.
