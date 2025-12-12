# Repository Guidelines

## Project Structure & Module Organization
- `notebooks/`: primary work happens here; `01_research.ipynb`/`01_research.py` e `02_submission.ipynb`/`02_submission.py` são espelhados via jupytext (`ipynb,py:percent`). Edite o `.py` e rode `jupytext --sync notebooks/01_research.py` (ou `notebooks/02_submission.py`) para manter o `.ipynb` atualizado.
- `data/raw/`: original Kaggle files (`train.csv`, `test.csv`, `hull-tactical-market-prediction.zip`). Do not commit new data.
- `data/processed/`: intermediate artifacts you generate; keep large files out of git.
- `models/`, `reports/`: optional outputs (artifacts/figures). Avoid committing heavy binaries.
- `scripts/`, `src/`: código reutilizável do projeto (pipelines, features, modelos); prefira adicionar aqui em vez de inchar notebooks.
- **Notebook workflow**: edite `notebooks/01_research.py` (pesquisa) ou `notebooks/02_submission.py` (submissão) e rode `jupytext --sync` no arquivo editado.

## Build, Test, and Development Commands
- Create/activate a virtual env (Python 3.10+ recommended), then install deps used in notebooks:
  - `pip install -r requirements.txt`
- Sync notebook/script: `jupytext --sync notebooks/01_research.py` (ou `notebooks/02_submission.py`)
- Fetch data via CLI (local only): `kaggle competitions download -c hull-tactical-market-prediction -p data/raw`

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
- Nunca commite credenciais Kaggle; se precisar do CLI localmente, use `~/.kaggle/kaggle.json` com `chmod 600`.
- Large artifacts: prefer `.gitignore` entries; stage only minimal derived assets needed for review.

## Regras adicionais (manutenção)
- Trate `src/hull_tactical/` como fonte única de verdade; o notebook deve só orquestrar/EDA leve.
- Notebook deve executar do topo ao fim em kernel limpo; evite células que dependem de execução parcial.
- Prefira imports explícitos no notebook; evite `import *` e exponha API pública via `__all__`/`__init__.py`.
- Funções em `src/` com docstring curta, type hints e sem side‑effects (não baixar dados nem gravar arquivos automaticamente).
- Centralize configuração em dataclasses (`HullConfig`) e passe via parâmetros; evite globais mutáveis.
- Dependências do pacote devem ser leves; libs pesadas/experimentais ficam opcionais e só no notebook ou `scripts/`.
- Artefatos gerados vão para `data/processed/`, `models/` ou `reports/` e entram no `.gitignore` se forem pesados.
- Ao adicionar/refatorar função importante, inclua teste unitário mínimo em `tests/` e rode `pytest`.
- Nunca automatize download de dados dentro de módulos; use Kaggle CLI ou `scripts/` quando solicitado.
- Antes de finalizar mudanças, rode smoke‑check de imports/pipeline mínimo.

## Planning Workflow
- Antes de iniciar mudanças, crie um arquivo `.md` com o plano de modificações (ex.: `PLAN.md`) listando passos concretos.
- Siga esse plano como próximos passos; atualize o arquivo se o escopo mudar e mantenha-o no repositório para consulta.
- Mantenha o `PLAN.md` (ou arquivo de plano equivalente) em formato de checklist com caixas de seleção (`- [ ]`) para permitir marcar o progresso.
