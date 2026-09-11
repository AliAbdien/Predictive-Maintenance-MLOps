from src.data import FEATURE_COLUMNS, LEAKAGE_COLUMNS, TARGET_COLUMN, load_dataset


def test_no_leakage_columns_in_features():
    """The failure-mode breakdown columns must never end up in the feature set -
    they encode which failure happened, which is only known after the fact."""
    assert not set(LEAKAGE_COLUMNS) & set(FEATURE_COLUMNS)
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_load_dataset_shapes_and_split():
    X_train, X_test, y_train, y_test = load_dataset()
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert list(X_train.columns) == FEATURE_COLUMNS
    assert len(X_train) + len(X_test) == 10000


def test_stratified_split_preserves_failure_rate():
    X_train, X_test, y_train, y_test = load_dataset()
    # Failure rate should match between splits within a small tolerance -
    # stratify=y in the train_test_split call is what guarantees this.
    assert abs(y_train.mean() - y_test.mean()) < 0.01
