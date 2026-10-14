"""Partition, never skip, existing tests that allocate their own scratch DB.

Run both phases through full_suite.py. The regular DB is deleted before scratch
fixtures start; the scratch phase uses only the cluster's empty admin database.
"""

import os

SCRATCH_FIXTURES = {"scratch_db_factory", "db048_url", "isolated_database"}


def pytest_collection_modifyitems(config, items):
    phase = os.environ["M0_DATABASE_PHASE"]
    selected, deselected = [], []
    for item in items:
        scratch = bool(SCRATCH_FIXTURES.intersection(item.fixturenames))
        (selected if scratch == (phase == "scratch") else deselected).append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
