"""Immutable A12 receipt projection edge cases independent of catalog/database state."""
from datetime import datetime, timezone

import pytest

import kosher_tiers


def lot(lot_id, tier='extra_kosher', actor=9, at='2026-10-13T14:00:00+00:00'):
    return {'id': lot_id, 'kosher_tier': tier, 'kosher_attested_by_actor_id': actor,
            'kosher_attested_at': at}


@pytest.mark.parametrize('action', ['adjust', 'ship_order', 'ship_standalone', 'ship_bulk'])
def test_lot_receipts_use_lot_proof_instead_of_regular_transaction_default(action):
    receipt = {'action': action, 'transactions': [{'kosher_tier': 'regular'}], 'lots': [lot(11)]}
    fields = kosher_tiers.summary_fields(receipt)
    assert fields['kosher_tier'] == 'extra_kosher'
    assert fields['kosher_attested_by_actor_id'] == 9
    assert fields['kosher_attested_at'] == '2026-10-13T14:00:00+00:00'
    assert fields['transaction_kosher_downgrade'] is False


def test_frozen_ship_proofs_ignore_later_lot_metadata():
    proof = {'lot_id': 11, 'kosher_tier': 'extra_kosher', 'kosher_attested_by_actor_id': 9,
             'kosher_attested_at': '2026-10-13T14:00:00+00:00'}
    receipt = {'action': 'ship_order', 'response': {'kosher_lots': [proof]},
               'lots': [lot(11, tier='regular', actor=10)]}
    assert kosher_tiers.summary_lots(receipt) == [proof]
    fields = kosher_tiers.summary_fields(receipt)
    assert (fields['kosher_tier'], fields['kosher_attested_by_actor_id']) == ('extra_kosher', 9)


def test_multiple_ship_attesters_are_preserved_without_claiming_one_owner():
    receipt = {'action': 'ship_order', 'lots': [lot(12, actor=10), lot(11, actor=9)]}
    fields = kosher_tiers.summary_fields(receipt)
    assert fields['kosher_tier'] == 'extra_kosher'
    assert fields['kosher_attested_by_actor_id'] is None and fields['kosher_attested_at'] is None
    assert [evidence['lot_id'] for evidence in kosher_tiers.summary_lots(receipt)] == [11, 12]
    assert [evidence['kosher_attested_by_actor_id'] for evidence in kosher_tiers.summary_lots(receipt)] == [9, 10]


def test_mixed_ship_tiers_do_not_claim_the_entire_shipment_is_extra():
    receipt = {'action': 'ship_order', 'lots': [lot(11), lot(12, tier='regular', actor=None, at=None)]}
    fields = kosher_tiers.summary_fields(receipt)
    assert fields['kosher_tier'] == 'regular' and fields['kosher_attested_by_actor_id'] is None
    assert {evidence['kosher_tier'] for evidence in kosher_tiers.summary_lots(receipt)} == {'regular', 'extra_kosher'}


def test_historical_receive_uses_saved_extra_evidence_and_label():
    receipt = {'action': 'receive', 'response': {'kosher_tier': 'extra_kosher',
               'kosher_label': 'Stored approval label', 'kosher_attested_by_actor_id': 9,
               'kosher_attested_at': '2026-10-13T14:00:00+00:00'},
               'draft': {'kosher_tier': 'regular', 'kosher_label': 'Regular'},
               'transactions': [{'kosher_tier': 'regular'}]}
    fields = kosher_tiers.summary_fields(receipt)
    assert fields['kosher_tier'] == 'extra_kosher' and fields['kosher_label'] == 'Stored approval label'
    assert fields['kosher_attested_by_actor_id'] == 9


def test_regular_pack_downgrade_retains_regular_output_and_explicit_null_attester():
    receipt = {'action': 'pack', 'response': {'kosher_tier': 'regular', 'transaction_kosher_downgrade': True,
               'kosher_attested_by_actor_id': None, 'kosher_attested_at': None},
               'transactions': [{'kosher_tier': 'regular', 'kosher_attested_by_actor_id': 9}]}
    fields = kosher_tiers.summary_fields(receipt)
    assert set(fields) == set(kosher_tiers.SUMMARY_FIELDS)
    assert fields['kosher_tier'] == 'regular' and fields['transaction_kosher_downgrade'] is True
    assert fields['kosher_attested_by_actor_id'] is None and fields['kosher_attested_at'] is None


def test_snapshot_freezes_distinct_ship_lot_proofs_before_serialization():
    at = datetime(2026, 10, 13, 14, tzinfo=timezone.utc)
    proofs = [{'lot_id': 11, 'kosher_tier': 'extra_kosher', 'kosher_attested_by_actor_id': 9, 'kosher_attested_at': at},
              {'lot_id': 12, 'kosher_tier': 'extra_kosher', 'kosher_attested_by_actor_id': 10, 'kosher_attested_at': at}]

    class Cursor:
        def execute(self, sql, params):
            assert 'FROM lots' in sql and params == ([11, 12],)

        def fetchall(self):
            return proofs

    response = {'shipment_id': 3}
    kosher_tiers.snapshot_response(Cursor(), response, 'ship_order', lot_ids=[12, 11, 12])
    assert response['kosher_lots'] == proofs
    assert response['kosher_tier'] == 'extra_kosher'
    assert response['kosher_attested_by_actor_id'] is None
