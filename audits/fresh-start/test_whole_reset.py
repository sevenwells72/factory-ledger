"""Offline O-3 whole-reset proof tests: several signed area groups posted through the real
executor against the fake ticket API, then proved as a union. No live access; temp files removed."""
import contextlib
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import apply_reset as app
import fresh_start_common as f
import fresh_start_v3 as v
import whole_reset_proof as w
from test_fresh_start_v3 import FakeTicketBackend

MOVE_COLUMNS = ['move_id', 'product_id', 'SKU', 'lot_code', 'quantity', 'unit', 'from_area', 'from_location', 'to_area',
                'to_location', 'moved_at', 'tag_id', 'fl_transaction_id', 'notes']
# product -> (book lb, area/qty observations). 1: A only, −600 (owner-held >500 lb); 2: B only;
# 3: one lot counted in B AND C (multi-area, handled once by the B+C group); 4: D, counted at book (no change).
PRODUCTS = {1: (1000, [('A', '400')]), 2: (100, [('B', '90')]), 3: (200, [('B', '120'), ('C', '75')]), 4: (50, [('D', '50')])}
OFFSET = dict(A=0, B=10, C=20, D=30)


class WholeResetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='.FAKE-whole-', dir=f.OUT)
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        for name, val in [('KEY_FILE', self.out / 'key'), ('JOURNAL_ROOT', self.out / 'state'), ('LOCK', self.out / 'lock'), ('RESULTS', self.out / 'results')]:
            c = patch.object(app, name, val); c.start(); self.addCleanup(c.stop)
        app.KEY_FILE.write_text('FAKE_SECRET_MUST_NOT_LEAK'); app.KEY_FILE.chmod(0o600)
        for target in ('socket.create_connection', 'socket.socket.connect', 'subprocess.run', 'fresh_start_common.snapshot',
                       'apply_reset.LiveBackend.request', 'apply_reset.LiveBackend.actor'):
            c = patch(target, side_effect=AssertionError('LIVE ACCESS FORBIDDEN')); c.start(); self.addCleanup(c.stop)
        c = patch('fresh_start_v3.snapshot', side_effect=lambda target='production': self.b.snapshot()); c.start(); self.addCleanup(c.stop)
        self.t = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(hours=3)
        products = [dict(id=pid, odoo_code=f'1100{pid}', name=f'TEST ingredient {pid}', type='ingredient', active=True, is_service=False, uom='lb')
                    for pid in PRODUCTS]
        self.s = dict(snapshot_at=self.t.isoformat(), products=products, catalog=deepcopy(products), lots=[], lines=[], corrections=[],
                      transaction_line_counts={}, actors=[dict(id=7, name='FAKE Owner', role='owner', active=True)],
                      correction_tickets=[], shortages=[], held_exceptions=[])
        for pid, (book, _) in PRODUCTS.items():
            self.s['lots'].append(dict(id=pid, product_id=pid, lot_code=f'LOT-{pid}', status='active', merged_into_lot_id=None))
            when = (self.t - timedelta(days=1)).isoformat()
            self.s['lines'].append(dict(line_id=pid, transaction_id=pid, product_id=pid, lot_id=pid, quantity_lb=f.D(book), occurred_at=when,
                                        transaction_created_at=when, line_created_at=when, transaction_created_at_source='database',
                                        line_created_at_source='database', type='receive', effective_status='posted', adjust_reason=None,
                                        operator_id='TEST', notes='', line_correction_at=None, transaction_correction_at=None))
            self.s['transaction_line_counts'][str(pid)] = 1
        self.b = FakeTicketBackend(self.s)
        self.count = []
        for pid, (_, seen) in PRODUCTS.items():
            for n, (area, qty) in enumerate(seen):
                at = self.t + timedelta(minutes=OFFSET[area])
                r = {k: '' for k in v.COUNT_COLUMNS}
                r.update(sheet_id='S' + area, area=area, count_date=at.astimezone(f.PLANT).date().isoformat(), counter='TEST',
                         start_time=(self.t + timedelta(minutes=OFFSET[area] - 5)).isoformat(), end_time=(self.t + timedelta(minutes=OFFSET[area] + 5)).isoformat(),
                         row_id=f'{area}{pid}', product_id=str(pid), SKU=f'1100{pid}', name=f'TEST ingredient {pid}', row_kind='COUNT',
                         lot_code=f'LOT-{pid}', quantity=qty, unit='lb', location=area + ' shelf', counted_time=at.isoformat())
                self.count.append(r)
        self.coverage = [dict(product_id=str(pid), SKU=f'1100{pid}', name=f'TEST ingredient {pid}', all_locations_searched='yes',
                              completed_at=(self.t + timedelta(hours=1)).isoformat(), owner_initials='TEST', notes='') for pid in PRODUCTS]
        self.review = {}
        a = v.analyze(self.b.snapshot(), dict(count=self.count, coverage=self.coverage, moves=[], review={}))
        self.review['multi_area_reviews'] = {fp: dict(owner='TEST', no_unlogged_moves=True) for fp in a['review_template']['multi_area_reviews']}
        self.inputs = self.write_inputs(self.out / 'inputs')
        self.groups = {}

    # ── helpers ──
    def write_inputs(self, folder, count=None):
        folder.mkdir(parents=True, exist_ok=True)
        for name, rows, cols in (('count', count or self.count, v.COUNT_COLUMNS), ('coverage', self.coverage, list(self.coverage[0])), ('moves', [], MOVE_COLUMNS)):
            with (folder / (name + '.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=cols); writer.writeheader(); writer.writerows(rows)
        (folder / 'review.json').write_text(json.dumps(self.review))
        return {k: folder / (k + ('.json' if k == 'review' else '.csv')) for k in ('count', 'coverage', 'moves', 'review')}

    def sign(self, name, areas, ids, held='N', signature='FAKE Owner'):
        folder = self.out / name
        folder.mkdir()
        data = v.load_inputs(*(self.inputs[k] for k in ('count', 'coverage', 'moves', 'review')))
        a = v.analyze(self.b.snapshot(), data, (), areas)
        preview = folder / 'reset-preview-v3.json'
        preview.write_text(json.dumps(a, default=str))
        approval = folder / 'approval.md'
        approval.write_text('\n'.join([f'Preview file: {preview.name}', f'SHA-256: {app.digest(preview.read_bytes())}',
                                       'Approved product IDs: ' + ','.join(map(str, ids)), 'Owner name: FAKE Owner', 'Named actor: FAKE Owner',
                                       f'Date: {datetime.now(f.PLANT).date()}', f'Signature: {signature}', 'Opening entry classification: N',
                                       f'Owner approves held entries: {held}', 'Extra-kosher lots: none']) + '\n')
        self.groups[name] = dict(name=name, areas=areas, approved_preview=str(preview.relative_to(self.out)), approval=str(approval.relative_to(self.out)))
        return preview, approval

    def apply(self, name):
        g = self.groups[name]
        preview, approval = self.out / g['approved_preview'], self.out / g['approval']
        plan = app.load_plan(preview, approval, self.inputs['count'])
        phrase = f'APPLY OPENING BALANCE {plan["cutoff"].astimezone(f.PLANT).date()} {plan["sha"]}'
        g['journal'] = str((app.JOURNAL_ROOT / plan['sha'] / 'journal.jsonl').relative_to(self.out))
        with contextlib.redirect_stdout(io.StringIO()):
            return app.run(preview, approval, self.inputs['count'], True, self.b, lambda _: phrase)

    def cutover(self, held_mode=None):
        """G1 (A) → G2 (B) → G3 (B+C, the multi-area lot) → G4 (D, no change)."""
        self.sign('G1', ['A'], [1], held='Y')
        if held_mode:
            self.b.mode = held_mode
            with self.assertRaises(RuntimeError):
                self.apply('G1')
        else:
            self.assertIn(self.apply('G1'), (0, 2))
        self.sign('G2', ['B'], [2]); self.assertIn(self.apply('G2'), (0, 2))
        self.sign('G3', ['B', 'C'], [3]); self.assertIn(self.apply('G3'), (0, 2))   # 2 = other scope still open
        self.sign('G4', ['D'], [4]); self.groups['G4']['no_change'] = True

    def prove(self, groups=None, final=None, **extra):
        manifest = dict(version=1, target='production', expected_reset_products=len(self.s['products']),
                        final_inputs={k: str(p.relative_to(self.out)) for k, p in (final or self.inputs).items()},
                        groups=groups if groups is not None else list(self.groups.values()), **extra)
        path = self.out / 'manifest.json'
        path.write_text(json.dumps(manifest))
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = w.run(path, self.out / 'proof')
        self.printed = printed.getvalue()
        self.proof = json.loads((self.out / 'proof/whole-reset-proof.json').read_text())
        return code

    def kinds(self):
        return {g['kind'] for g in self.proof['gaps']}

    # ── cases ──
    def test_complete_reset_reconciles_with_multi_area_lot_handled_once(self):
        self.cutover()
        self.assertEqual(self.prove(), 0, self.printed)
        self.assertEqual(self.proof['verdict'], 'FULL RESET SCOPE RECONCILED')
        self.assertEqual(self.proof['gaps'], [])
        self.assertEqual(len(self.proof['proved_line_ids']), 3)
        multi = self.proof['multi_area_lots']
        self.assertEqual([(m['product_id'], m['areas'], m['handled_by']) for m in multi], [(3, ['B', 'C'], 'G3')])
        self.assertIn('G2', multi[0]['held_in'])
        self.assertEqual([g['posts'] for g in self.proof['groups']][3], [])
        self.assertIn('FULL RESET SCOPE RECONCILED', (self.out / 'proof/whole-reset-proof.md').read_text())

    def test_multi_area_new_extra_kosher_lot_written_lower_case(self):
        p = dict(id=283, odoo_code='9283', name='TEST extra batch', type='batch', active=True, is_service=False, uom='lb', kosher_tier='extra_kosher')
        self.s['products'].append(p); self.s['catalog'].append(dict(p))
        for area, qty in (('B', '180'), ('C', '120')):
            r = dict(next(x for x in self.count if x['area'] == area), row_id=area + '283', product_id='283', SKU='9283', name=p['name'], lot_code='ek-1', quantity=qty)
            self.count.append(r)
        self.coverage.append(dict(self.coverage[0], product_id='283', SKU='9283', name=p['name']))
        a = v.analyze(self.b.snapshot(), dict(count=self.count, coverage=self.coverage, moves=[], review={}))
        self.review['multi_area_reviews'] = {fp: dict(owner='TEST', no_unlogged_moves=True) for fp in a['review_template']['multi_area_reviews']}
        self.inputs = self.write_inputs(self.out / 'inputs')
        self.sign('G1', ['A'], [1], held='Y'); self.apply('G1')
        self.sign('G2', ['B'], [2]); self.apply('G2')
        preview, approval = self.sign('G3', ['B', 'C'], [3, 283], held='Y')
        approval.write_text(approval.read_text().replace('Extra-kosher lots: none', 'Extra-kosher lots: 283:EK-1'))
        self.apply('G3')
        self.sign('G4', ['D'], [4]); self.groups['G4']['no_change'] = True
        self.assertEqual(self.prove(), 0, self.printed)
        multi = {m['product_id']: m for m in self.proof['multi_area_lots']}
        self.assertEqual((multi[283]['lot_code'], multi[283]['areas'], multi[283]['handled_by']), ('EK-1', ['B', 'C'], 'G3'))
        self.assertTrue(any(x.startswith('FND-') for x in self.proof['groups'][2]['posts']))

    def test_verify_reset_entry_point(self):
        self.cutover()
        self.prove()
        with contextlib.redirect_stdout(io.StringIO()), patch('sys.argv', ['verify_reset.py', '--whole-reset', str(self.out / 'manifest.json'),
                                                                           '--output-dir', str(self.out / 'proof2')]):
            import verify_reset
            self.assertEqual(verify_reset.main(), 0)

    def test_missing_area_group(self):
        self.cutover()
        self.assertEqual(self.prove([g for n, g in self.groups.items() if n != 'G4']), 2)
        self.assertIn('MISSING_PRODUCT', self.kinds())
        self.assertEqual({g['product_id'] for g in self.proof['gaps'] if g['kind'] == 'MISSING_PRODUCT'}, {4})

    def test_missing_posted_group_is_also_an_unauthorized_post(self):
        self.cutover()
        self.assertEqual(self.prove([g for n, g in self.groups.items() if n != 'G2']), 2)
        self.assertTrue({'MISSING_PRODUCT', 'POST_OUTSIDE_APPROVED_PLAN'} <= self.kinds(), self.kinds())

    def test_overlap_two_groups_approve_one_product(self):
        self.cutover()
        self.sign('G5', ['D'], [4]); self.groups['G5']['no_change'] = True
        self.assertEqual(self.prove(), 2)
        self.assertEqual(self.kinds(), {'OVERLAP'})
        self.assertIn('G4, G5', self.proof['gaps'][0]['detail'])

    def test_extra_post_outside_any_signed_plan(self):
        self.cutover()
        status, draft = self.b.call('k', '/adjust/prepare', dict(lot_id=2, delta_lb=5, reason_code='physical_count', note='owner recount',
                                                                 occurred_at=datetime.now(timezone.utc).isoformat()))
        self.b.call('k', '/tickets/' + draft['ticket'] + '/commit', dict(payload_hash=draft['payload_hash'], acknowledged_warnings=[]))
        # The later post moves FL and the "after count" basis together, so balances still match:
        # only the union proof sees that no signed plan authorized it.
        self.assertEqual(self.prove(), 2)
        self.assertEqual(self.kinds(), {'POST_OUTSIDE_APPROVED_PLAN'})
        self.assertEqual(self.proof['final_verification_exit'], 0)

    def test_unsigned_group(self):
        self.cutover()
        approval = self.out / self.groups['G2']['approval']
        approval.write_text(approval.read_text().replace('Signature: FAKE Owner', 'Signature: someone else'))
        self.assertEqual(self.prove(), 2)
        self.assertTrue({'UNSIGNED_OR_INVALID_GROUP', 'MISSING_PRODUCT', 'POST_OUTSIDE_APPROVED_PLAN'} <= self.kinds(), self.kinds())
        self.assertIn('Owner signature mismatch', self.printed)

    def test_no_change_group_is_proved_without_a_journal(self):
        self.cutover()
        self.assertEqual(self.prove(), 0)
        g4 = next(g for g in self.proof['groups'] if g['name'] == 'G4')
        self.assertEqual((g4['no_change'], g4['journal'], g4['posts']), (True, None, []))

    def test_no_change_declared_for_a_group_with_changes(self):
        self.cutover()
        groups = deepcopy(list(self.groups.values()))
        groups[0].pop('journal'); groups[0]['no_change'] = True
        self.assertEqual(self.prove(groups), 2)
        self.assertTrue({'NO_CHANGE_GROUP_HAS_CHANGES', 'POST_OUTSIDE_APPROVED_PLAN'} <= self.kinds(), self.kinds())

    def test_concatenated_journal_is_refused(self):
        self.cutover()
        both = self.out / 'state/concatenated.jsonl'
        both.write_bytes((self.out / self.groups['G1']['journal']).read_bytes() + (self.out / self.groups['G2']['journal']).read_bytes())
        groups = deepcopy(list(self.groups.values()))
        groups[1]['journal'] = str(both.relative_to(self.out))
        self.assertEqual(self.prove(groups), 2)
        self.assertIn('JOURNAL_PROOF_FAILED', self.kinds())
        self.assertIn('Journal chain mismatch', self.printed)

    def test_same_journal_listed_twice(self):
        self.cutover()
        groups = deepcopy(list(self.groups.values()))
        groups[1]['journal'] = groups[0]['journal']
        self.assertEqual(self.prove(groups), 2)
        self.assertTrue({'DUPLICATE_GROUP_FILE', 'JOURNAL_PROOF_FAILED'} <= self.kinds(), self.kinds())

    def test_held_owner_entry_not_released(self):
        self.cutover(held_mode='approve_refused')
        self.assertEqual(self.prove(), 2)
        self.assertTrue({'HELD_NOT_RELEASED', 'UNEXPLAINED_DIFFERENCE'} <= self.kinds(), self.kinds())

    def test_observation_found_after_its_group_was_signed(self):
        self.cutover()
        late = dict(self.count[0], sheet_id='SE', area='E', row_id='E1', quantity='30', location='E shelf')
        final = self.write_inputs(self.out / 'final', self.count + [late])
        self.assertEqual(self.prove(final=final), 2)
        self.assertIn('MISSING_OBSERVATION', self.kinds())

    def test_count_row_edited_after_signing(self):
        self.cutover()
        edited = deepcopy(self.count)
        next(r for r in edited if r['row_id'] == 'B2')['quantity'] = '91'
        final = self.write_inputs(self.out / 'final', edited)
        self.assertEqual(self.prove(final=final), 2)
        self.assertTrue({'CHANGED_OBSERVATION', 'UNEXPLAINED_DIFFERENCE'} <= self.kinds(), self.kinds())

    def test_review_answer_changed_after_signing(self):
        self.cutover()
        fp = next(iter(self.review['multi_area_reviews']))
        self.review['multi_area_reviews'][fp] = dict(owner='SOMEONE ELSE', no_unlogged_moves=True)
        final = self.write_inputs(self.out / 'final')
        self.assertEqual(self.prove(final=final), 2)
        self.assertEqual(self.kinds(), {'REVIEW_CHANGED'})

    def test_signed_input_file_changed_after_approval(self):
        self.cutover()
        with self.inputs['moves'].open('a') as stream:
            stream.write('\n')
        self.assertEqual(self.prove(), 2)
        self.assertIn('UNSIGNED_OR_INVALID_GROUP', self.kinds())
        self.assertIn('signed input file changed', self.printed)

    def test_scope_count_must_match_manifest(self):
        self.cutover()
        path = self.out / 'manifest.json'
        self.prove()
        manifest = json.loads(path.read_text()); manifest['expected_reset_products'] = 178
        path.write_text(json.dumps(manifest))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(w.run(path, self.out / 'proof3'), 2)
        self.assertIn('SCOPE_CHANGED', {g['kind'] for g in json.loads((self.out / 'proof3/whole-reset-proof.json').read_text())['gaps']})

    def test_manifest_validation(self):
        self.cutover()
        g = deepcopy(self.groups['G4'])
        g['journal'] = self.groups['G1']['journal']
        with self.assertRaisesRegex(ValueError, 'give a journal, or no_change'):
            self.prove([g])
        with self.assertRaisesRegex(ValueError, 'only for a staging rehearsal'):
            self.prove(rehearsal_scope_product_ids=[1])
        with self.assertRaisesRegex(ValueError, 'no area groups'):
            self.prove([])

    def test_signed_plan_rules_match_the_executor(self):
        """The read-only approval check accepts and refuses what apply_reset.load_v3_plan does."""
        preview, approval = self.sign('G1', ['A'], [1], held='Y')
        app.load_plan(preview, approval, self.inputs['count'])
        w.signed_plan(preview, approval, 'production')
        for old, new in (('held entries: Y', 'held entries: N'), ('Signature: FAKE Owner', 'Signature: X'),
                         ('Named actor: FAKE Owner', 'Named actor: master'), ('Approved product IDs: 1', 'Approved product IDs: 2')):
            approval.write_text(approval.read_text().replace(old, new))
            with self.assertRaises(RuntimeError):
                app.load_plan(preview, approval, self.inputs['count'])
            with self.assertRaises(ValueError):
                w.signed_plan(preview, approval, 'production')
            approval.write_text(approval.read_text().replace(new, old))

    def test_proof_never_imports_the_write_executor(self):
        source = Path(w.__file__).read_text()
        self.assertNotIn('import apply_reset', source)
        self.assertNotIn('urlopen', source)


if __name__ == '__main__':
    unittest.main()
