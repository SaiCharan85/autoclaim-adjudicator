"""Simulator fixtures (builders live in tests/fakes.py)."""

import pandas as pd
import pytest

from autoclaim.lines.auto.simulator.world import World, load_world
from fakes import make_incidents


@pytest.fixture(scope="session")
def incidents() -> pd.DataFrame:
    return make_incidents()


@pytest.fixture(scope="session")
def world() -> World:
    return load_world().model_copy(update={"n_claims": 4000})
