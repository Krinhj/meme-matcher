# Repository Guidelines

## Project Structure & Module Organization
- `app.py`: live split-screen viewer; main entry point (`python app.py`).
- `docs/PRD.md`, `docs/PHASES.md`: product requirements and phased roadmap.
- `memes/`: tracked sample assets for indexing/matching experiments.
- `.venv/` (local), `requirements.txt`: Python virtualenv and locked dependency list.
- Future work: add `index_memes.py`, `memes_index.json`, and any model artifacts beside `app.py`.

## Build, Test, and Development Commands
- Create env & install deps:
  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Run live viewer (default 1920×1080 window): `python app.py`.
- Override resolution when debugging layouts: `python app.py --window-width 1280 --window-height 720`.

## Coding Style & Naming Conventions
- Python 3.13+. Stick to PEP 8, 4-space indentation, descriptive snake_case functions (`build_placeholder`, `run_viewer`).
- Keep modules ASCII-only unless a dependency requires Unicode; short module-level docstrings preferred.
- Configuration constants live near the top of each module (`WINDOW_TITLE`, `DEFAULT_WINDOW_WIDTH`).
- Document CLI arguments via `argparse` help strings; prefer explicit defaults.

## Testing Guidelines
- No automated tests yet; plan to add unit tests for the meme indexer and matching helpers (pytest recommended).
- When adding tests, mirror source tree (e.g., `tests/test_indexer.py`). Use descriptive names (`test_cosine_similarity_weights`).
- Smoke-test the viewer manually before merging (verify webcam feed, placeholder message, close-button behavior).

## Commit & Pull Request Guidelines
- Use concise, present-tense commits (e.g., `Add split-screen viewer placeholder scaling`, `Document phase completion`).
- PRs should describe the user-facing behavior, note environment impacts (dependencies, new CLI flags), and link related tickets/content posts.
- Include screenshots or short clips when UI behavior changes (e.g., new overlay, layout adjustments).

## Process & Phase Tracking
- After finishing a feature, confirm with the requester that the result meets expectations **before** marking the phase complete.
- Once approved, update `docs/PHASES.md` with the new status so the roadmap stays in sync with reality.

## Security & Configuration Tips
- Keep memes and captured frames local—no network calls are expected.
- Regenerate `requirements.txt` after any dependency change (`pip freeze > requirements.txt`).
- If MediaPipe support is required, pin Python to 3.12 in a new env; document the change in README + PRD risks.
