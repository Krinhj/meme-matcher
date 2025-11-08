# Meme Matcher - Phased Plan

A rough implementation roadmap that balances fast demo value with learning milestones. Each phase is shippable; stop whenever you have enough for content.

## Phase 0: Environment & Dataset Prep

**Status:** Completed – `.venv` created, dependencies installed, and ~14 starter memes collected.

- Create/activate a virtualenv; install OpenCV, NumPy, FER (or chosen emotion model), and utility libs.
- Curate a starter `memes/` folder (local only) with at least 20 diverse expressions.
- Capture a few reference selfies to test matching accuracy.

## Phase 1: Meme Indexing Script

**Status:** Completed — `index_memes.py` extracts FER emotions, optional gesture inference, manual tags, and structured JSON metadata.

- Implement `index_memes.py` to iterate through `memes/`, detect faces/emotions, and store vectors in `memes_index.json`.
- Add optional manual tags (JSON metadata) for memes with unclear expressions.
- Validate the JSON schema and document expected fields.

## Phase 2: Live Viewer Skeleton

**Status:** Completed – `app.py` now renders the split-screen webcam + placeholder view with clean close behavior.

- Build `app.py` to open the webcam, show the live feed, and reserve space for the meme pane (placeholder message for now).
- Ensure the window handles resize/close events gracefully and maintains ~30 FPS.

## Phase 3: Emotion Matching Loop

**Status:** Pending.

- Plug emotion detection into the live feed, convert outputs into the same vector space as `memes_index.json`.
- Compute cosine similarity to find the top match; display the meme image in the second pane.
- Add the "No meme matched yet" guard for low-confidence frames.

## Phase 4: Gesture + Scoring Tweaks (Optional)

**Status:** Pending.

- Layer in MediaPipe or heuristic gesture tags and incorporate them into the weighted score (0.7 emotion / 0.25 gesture / 0.05 CLIP).
- Expose a simple config for adjusting weights or thresholds without editing code.

## Phase 5: Polishing for Content

**Status:** Pending.

- Add overlay text showing emotion confidence or the meme name for storytelling.
- Capture a short screen recording or OBS scene for LinkedIn/IG.
- Write a mini devlog or README snippet summarizing what you learned.

## Stretch Ideas

**Status:** Backlog / nice-to-have.

- Integrate CLIP embeddings for memes that lack obvious facial cues.
- Add keyboard shortcuts (space to pause, s to save comparison frames).
- Experiment with multi-person detection or group memes.
