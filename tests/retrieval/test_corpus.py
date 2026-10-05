import copy

import pytest
import yaml
from retrieval_fakes import POLICY, policy

from autoclaim.retrieval.corpus import PolicyDoc, load_policy


def test_load_and_index(tmp_path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(POLICY), encoding="utf-8")
    doc = load_policy(path)
    assert set(doc.by_id) == {"INS-COLL", "INS-COMP", "DEF-COLL", "EXC-WEAR", "GEN-EXCEPT"}
    assert doc.by_id["DEF-COLL"].document.startswith("Collision. ")


def test_fingerprint_changes_with_text() -> None:
    raw = copy.deepcopy(POLICY)
    raw["clauses"][0]["text"] += " Extra."
    assert policy().fingerprint() == policy().fingerprint()
    assert PolicyDoc.model_validate(raw).fingerprint() != policy().fingerprint()


def _dup(r: dict) -> None:
    r["clauses"].append(copy.deepcopy(r["clauses"][0]))


def _dangling(r: dict) -> None:
    r["clauses"][0]["edges"].append({"type": "see_also", "target": "X"})


def _bad_type(r: dict) -> None:
    r["clauses"][0]["edges"].append({"type": "bogus", "target": "DEF-COLL"})


@pytest.mark.parametrize(
    ("mutate", "match"), [(_dup, "duplicate"), (_dangling, "unknown"), (_bad_type, "type")]
)
def test_invalid_policies_rejected(mutate, match) -> None:
    raw = copy.deepcopy(POLICY)
    mutate(raw)
    with pytest.raises(ValueError, match=match):
        PolicyDoc.model_validate(raw)
