from src.train_models import (
    load_dataset,
    split_data,
    build_model_configs,
)


def test_load_dataset_returns_rows():
    df = load_dataset()
    assert len(df) > 0


def test_split_data_keeps_all_rows():
    df = load_dataset()
    x_train, x_val, y_train, y_val = split_data(df)

    assert len(x_train) + len(x_val) == len(df)
    assert len(y_train) + len(y_val) == len(df)


def test_model_configs_exist():
    configs = build_model_configs()

    assert "logreg_basic" in configs
    assert "logreg_tuned" in configs
    assert "linear_svm" in configs
    assert "naive_bayes" in configs