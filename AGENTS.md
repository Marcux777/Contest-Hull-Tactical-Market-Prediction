# Repository Guidelines

## Project Structure & Module Organization
- `notebooks/`: primary work happens here; `Hull Tactical.ipynb` is mirrored as `Hull Tactical.py` via jupytext (`ipynb,py:percent`). Edit either and run `jupytext --sync notebooks/Hull\ Tactical.ipynb` to keep them aligned.
- `data/raw/`: original Kaggle files (`train.csv`, `test.csv`, `hull-tactical-market-prediction.zip`). Do not commit new data.
- `data/processed/`: intermediate artifacts you generate; keep large files out of git.
- `models/`, `reports/`: optional outputs (artifacts/figures). Avoid committing heavy binaries.
- `scripts/`, `src/`: currently empty; prefer adding reusable code here instead of bloating notebooks.
- **Notebook workflow**: sempre edite em `notebooks/Hull Tactical.py` e depois rode `jupytext --sync notebooks/Hull\ Tactical.ipynb` para manter o `.ipynb` atualizado.

## Build, Test, and Development Commands
- Create/activate a virtual env (Python 3.10+ recommended), then install deps used in notebooks:
  - `pip install pandas numpy seaborn matplotlib lightgbm scikit-learn kaggle jupytext`
- Sync notebook/script: `jupytext --sync notebooks/Hull\ Tactical.ipynb`
- Fetch data via CLI (requires credentials): `kaggle competitions download -c hull-tactical-market-prediction -p data/raw`

## Coding Style & Naming Conventions
- Python, 4-space indentation; keep cells/scripts concise and reusable.
- Prefer functions/utilities in `src/` or `scripts/`; keep notebook cells for orchestration/EDA.
- Use snake_case for variables/functions; lowercase-kebab for files; mirror notebook/script names when syncing.
- Keep comments brief and purposeful; avoid noisy prints.

## Testing Guidelines
- No formal test suite yet. When adding functions, include lightweight assertions or example runs in notebooks/scripts.
- If you add tests, mirror structure under `tests/` with `pytest` and keep names `test_*.py`; document any new test command in this file.

## Commit & Pull Request Guidelines
- Commits: short imperative subject (e.g., “add kaggle sync helper”), include why when non-obvious. Keep data and secrets out of commits.
- PRs: state goal, key changes, and any data/metric impacts; link issues/tasks when applicable. Include screenshots/plots if UI or modeling results change.

## Security & Configuration Tips
- Set Kaggle creds via env vars `KAGGLE_USERNAME`/`KAGGLE_KEY` (or pre-create `~/.kaggle/kaggle.json` with `chmod 600`); never commit keys.
- Large artifacts: prefer `.gitignore` entries; stage only minimal derived assets needed for review.

## Planning Workflow
- Antes de iniciar mudanças, crie um arquivo `.md` com o plano de modificações (ex.: `PLAN.md`) listando passos concretos.
- Siga esse plano como próximos passos; atualize o arquivo se o escopo mudar e mantenha-o no repositório para consulta.
- Mantenha o `PLAN.md` (ou arquivo de plano equivalente) em formato de checklist com caixas de seleção (`- [ ]`) para permitir marcar o progresso.
