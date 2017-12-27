"""Test Developer utilities"""
from xpdtools.dev_utils import _timestampstr


def test_timestampstr():
    assert _timestampstr(0) == '19691231-190000'
