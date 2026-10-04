import numpy as np
import pandas as pd

from autoclaim.ml import frame as fr


def test_feature_list_excludes_reserved_and_requested() -> None:
    df = pd.DataFrame(columns=["a", "b", fr.Y, fr.T, fr.AMOUNT])
    assert fr.feature_list(df) == ["a", "b"]
    assert fr.feature_list(df, exclude=["b"]) == ["a"]


def test_categorical_columns_inferred_from_dtypes() -> None:
    df = pd.DataFrame({"make": ["x", "y"], "age": [30.0, 40.0], "n": [1, 2], "flag": [True, False]})
    assert fr.categorical_columns(df, ["make", "age", "n", "flag"]) == ["make"]


def test_for_catboost_strings_and_floats() -> None:
    df = pd.DataFrame({"make": ["Honda", None], "age": [30, None], "flag": [True, False]})
    out = fr.for_catboost(df, ["make", "age", "flag"], cats=["make"])
    assert out["make"].tolist() == ["Honda", "missing"]
    assert out["age"].dtype == "float64"
    assert np.isnan(out.loc[1, "age"])
    assert out["flag"].tolist() == [1.0, 0.0]


def test_category_vocab_fixes_levels_and_drops_unseen() -> None:
    train = pd.DataFrame({"make": ["b", "a", "b"], "age": [1.0, 2.0, 3.0]})
    vocab = fr.CategoryVocab.fit(train, ["make"])
    assert vocab.levels == {"make": ["a", "b"]}
    test = vocab.transform(pd.DataFrame({"make": ["a", "zzz"], "age": ["1", "x"]}), ["make", "age"])
    assert list(test["make"].cat.categories) == ["a", "b"]
    assert test["make"].isna().tolist() == [False, True]
    assert test["age"].tolist()[0] == 1.0
    assert np.isnan(test["age"].tolist()[1])  # unparseable numbers become NaN, never strings
