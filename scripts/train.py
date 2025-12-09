import json
from pathlib import Path

from hull_tactical import data, pipeline
from hull_tactical.models import default_config


def main():
    cfg = default_config()
    df_train, df_test = data.load_raw_data(Path("data"))
    # Para treino local, usamos apenas train; test fica para submissão.
    allocations = pipeline.train_pipeline(data_dir=Path("data"))
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Salvamos alocações como proxy de modelo simples
    alloc_path = output_dir / "allocations_train.pkl"
    allocations.to_pickle(alloc_path)
    summary = {"feature_cfg": cfg.feature_cfg, "intentional_cfg": cfg.intentional_cfg, "allocations_path": str(alloc_path)}
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Treino concluído. Alocações salvas em {alloc_path}")


if __name__ == "__main__":
    main()
