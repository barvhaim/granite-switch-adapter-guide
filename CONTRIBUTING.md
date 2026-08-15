# Contributing

Contributions that improve correctness, reproducibility, or clarity are welcome.

## Development setup

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

The default test suite is CPU-only and does not download model weights.

## Documentation rules

- Distinguish commands verified by CI from GPU commands that readers must run themselves.
- Cite upstream behavior with a stable source URL or pinned revision.
- Do not add benchmark claims without a committed evaluation artifact and reproduction command.
- Keep sample data synthetic and free of personal or confidential information.

## Pull requests

Explain the user-facing change, list the exact verification commands you ran, and call out any GPU path that was not exercised.
