"""Test Developer utilities"""
from xpdtools.dev_utils import _timestampstr


def test_timestampstr():
    assert isinstance(_timestampstr(0), str)
