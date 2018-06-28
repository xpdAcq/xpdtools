import pytest

import tempfile


@pytest.fixture(scope="function")
def fast_tmpdir():
    td = tempfile.TemporaryDirectory()
    yield td.name
    td.cleanup()
