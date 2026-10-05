from autoclaim.datasets.sas_labels import parse_proc_format, read_labels

SOURCE = """
PROC FORMAT;
*MAKE;
VALUE V3Z
1  = 'AMC'
12 = 'FORD'
;
value v19n 1='Driven Away' 2 = 'Towed Due To Damage' 7-8 = 'Not Reported' OTHER = 'x';
VALUE $CHARF 'A' = 'char format';
"""


def test_parse_proc_format() -> None:
    labels = parse_proc_format(SOURCE)
    assert labels["V3Z"] == {1.0: "AMC", 12.0: "FORD"}
    assert labels["V19N"] == {1.0: "Driven Away", 2.0: "Towed Due To Damage", 7.0: "Not Reported",
                              8.0: "Not Reported"}  # fmt: skip
    assert "$CHARF" not in labels and "CHARF" not in labels


def test_read_labels_from_folder(tmp_path) -> None:
    (tmp_path / "Format02.sas").write_text(SOURCE, encoding="latin-1")
    assert read_labels(tmp_path)["V3Z"][12.0] == "FORD"
    assert read_labels(tmp_path / "missing_dir_is_empty") == {}
