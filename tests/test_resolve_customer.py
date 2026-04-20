"""Tests for resolve_customer_id address-tiebreaker logic.

Scenario covered (per Pass 1 spec): when a customer name fuzzy-matches more
than one existing row AND the caller supplies an address, the resolver should
silently pick the single candidate whose stored address is clearly the closest
trigram match — collapsing two "Setton" rows to one without bothering the
operator for disambiguation.

Test data is seeded into a real Postgres transaction via the `db_cursor`
fixture and rolled back at teardown — production data is never touched.
"""

import pytest

from main import resolve_customer_id


SETTON_ADDRESS = "85 Austin Blvd, Commack, NY 11725"
OTHER_ADDRESS = "1 Different Way, Los Angeles, CA 90001"


def _insert_customer(cur, name: str, address: str) -> int:
    cur.execute(
        "INSERT INTO customers (name, address, active) VALUES (%s, %s, true) RETURNING id",
        (name, address),
    )
    return cur.fetchone()["id"]


@pytest.mark.db
def test_address_tiebreaker_collapses_two_name_matches(db_cursor):
    """Two rows share the 'Setton' prefix. When the caller supplies the address
    of one of them, resolve_customer_id picks that row silently."""

    keeper_id = _insert_customer(
        db_cursor, "Setton Farms (TEST)", SETTON_ADDRESS
    )
    decoy_id = _insert_customer(
        db_cursor, "Setton International Foods (TEST DECOY)", OTHER_ADDRESS
    )

    resolved_id, resolved_name = resolve_customer_id(
        db_cursor,
        "Setton (TEST)",
        auto_create=False,
        address=SETTON_ADDRESS,
    )

    assert resolved_id == keeper_id, (
        f"expected tiebreaker to pick id={keeper_id} (matching address), "
        f"got id={resolved_id}"
    )
    assert "Farms" in resolved_name
    assert resolved_id != decoy_id


@pytest.mark.db
def test_no_address_still_raises_409(db_cursor):
    """Baseline: same two rows, no address passed → 409 ambiguous (unchanged
    legacy behavior). Guards the 'additive' promise."""
    from fastapi import HTTPException

    _insert_customer(db_cursor, "Setton Farms (TEST)", SETTON_ADDRESS)
    _insert_customer(db_cursor, "Setton International Foods (TEST DECOY)", OTHER_ADDRESS)

    with pytest.raises(HTTPException) as exc:
        resolve_customer_id(db_cursor, "Setton (TEST)", auto_create=False)

    assert exc.value.status_code == 409
    assert exc.value.detail["error_code"] == "CUSTOMER_AMBIGUOUS"


@pytest.mark.db
def test_address_too_far_falls_back_to_409(db_cursor):
    """If both candidates have addresses but neither is close to the supplied
    one, the tiebreaker does not fire — operator must still disambiguate."""
    from fastapi import HTTPException

    _insert_customer(db_cursor, "Setton Farms (TEST)", "999 Nowhere Rd, Boise, ID")
    _insert_customer(
        db_cursor, "Setton International Foods (TEST DECOY)", "2 Other St, Miami, FL"
    )

    with pytest.raises(HTTPException) as exc:
        resolve_customer_id(
            db_cursor,
            "Setton (TEST)",
            auto_create=False,
            address=SETTON_ADDRESS,
        )

    assert exc.value.status_code == 409
    assert exc.value.detail["error_code"] == "CUSTOMER_AMBIGUOUS"


@pytest.mark.db
def test_single_name_match_unchanged_by_address(db_cursor):
    """If the name already resolves to exactly one row, the address arg is a
    no-op. Guards against accidentally rejecting a unique match on weak
    address similarity."""
    keeper_id = _insert_customer(db_cursor, "UniqueCorpXYZ (TEST)", "nowhere")

    resolved_id, _ = resolve_customer_id(
        db_cursor,
        "UniqueCorpXYZ (TEST)",
        auto_create=False,
        address=SETTON_ADDRESS,  # deliberately mismatched
    )
    assert resolved_id == keeper_id
