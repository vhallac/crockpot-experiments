# Experiment design notes

Use this directory for experiment groups that are still primarily specifications, plans, or early prototypes. When an experiment grows substantial reusable Python code, promote the implementation to a top-level package while keeping the original spec here or linking to it.

Every experiment directory should include:

- `README.md` — what is measured, expected signal, execution path, result policy;
- `spec.md` — pre-registration or detailed design when available;
- small curated artifacts only, not raw generated outputs.

Generated outputs belong under repository-level `outputs/`, which is ignored by git.
