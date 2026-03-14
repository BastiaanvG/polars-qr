import runpy
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.py"))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_the_example_runs(example, capsys):
    runpy.run_path(str(example), run_name="__main__")

    assert capsys.readouterr().out
