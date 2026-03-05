"""
Unit and regression test for the pycomsia package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import pycomsia


def test_pycomsia_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pycomsia" in sys.modules
