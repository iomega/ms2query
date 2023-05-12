import os

import pytest

from ms2query.query_from_sqlite_database import SqliteLibrary


@pytest.fixture(scope="package")
def path_to_general_test_files() -> str:
    return os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        'tests/test_files/general_test_files')


@pytest.fixture(scope="package")
def path_to_test_files():
    return os.path.join(os.path.split(os.path.dirname(__file__))[0], 'tests/test_files')


@pytest.fixture(scope="package")
def sqlite_library(path_to_test_files):
    path_to_library = os.path.join(path_to_test_files, "general_test_files", "100_test_spectra.sqlite")
    return SqliteLibrary(path_to_library)