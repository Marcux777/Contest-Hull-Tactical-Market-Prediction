from pathlib import Path

from hull_tactical import data, pipeline


def main():
    df_train, df_test = data.load_raw_data(Path("data"))
    # Treina rápido e gera submissão em uma tacada
    allocations = pipeline.train_pipeline(data_dir=Path("data"))
    row_id_col = "row_id" if "row_id" in df_test.columns else df_test.columns[0]
    sub_path = Path("data/submissions/submission.csv")
    pipeline.make_submission_csv(sub_path, allocations, df_test[row_id_col])
    print(f"Submissão gerada em {sub_path}")


if __name__ == "__main__":
    main()
