"""Standalone safety tests; no application imports or database/network access.
Run: python3 -m unittest discover -s tests -p test_granola_writedown_0917.py
"""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('granola', Path(__file__).parents[1] / 'scripts/granola_writedown_0917.py')
g = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g)


def posted_prefix(plan, n):
    fresh = copy.deepcopy(plan['snapshot'])
    events = []
    for i,r in enumerate(plan['adjustments'][:n]):
        fresh['totals'][str(r['product_id'])] += g.D(r['change'])
        for l in fresh['lots']:
            if l['lot_id'] == r['lot_id']:
                l['balance'] = g.D(r['after'])
        events.append(dict(transaction_id=9000+i, product_id=r['product_id'], lot_id=r['lot_id'],
                           quantity_lb=g.D(r['change']), adjust_reason=r['reason'],
                           business_date='2026-09-17', operator_id='legacy-shared-key'))
    fresh['lots'] = [l for l in fresh['lots'] if l['balance'] != 0]
    fresh['existing_adjustments'] = events
    return fresh


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.plan = g.load_plan()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        journal_patch = patch.object(g, 'JOURNAL', Path(tmp.name) / 'journal.json')
        journal_patch.start()
        self.addCleanup(journal_patch.stop)

    def test_every_resume_prefix_and_split_lot(self):
        for n in range(25):
            self.assertEqual(g.reconcile(self.plan, posted_prefix(self.plan,n))[0], n)

    def test_differences_explain_expected_actual(self):
        f=posted_prefix(self.plan,0);f['lots'][-1]['balance'] += g.D('0.01')
        with self.assertRaisesRegex(RuntimeError, r'lot 1455: expected 269.04 lb; actual 269.05 lb'):
            g.reconcile(self.plan,f)

    def test_duplicate_foreign_or_nonprefix_posting(self):
        f=posted_prefix(self.plan,1)
        f['existing_adjustments'].append(copy.deepcopy(f['existing_adjustments'][0]))
        with self.assertRaises(RuntimeError):g.reconcile(self.plan,f)
        f=posted_prefix(self.plan,1);f['existing_adjustments'][0]['adjust_reason']='unapproved'
        with self.assertRaises(RuntimeError):g.reconcile(self.plan,f)
        f=posted_prefix(self.plan,2);f['existing_adjustments'].pop(0)
        with self.assertRaises(RuntimeError):g.reconcile(self.plan,f)

    def test_uncertain_request_never_retried_absent_event(self):
        j=dict(plan_sha256=g.fingerprint(), attempts=[dict(sequence=1,status='in_flight')])
        with self.assertRaisesRegex(RuntimeError, 'Do not retry'):
            g.reconcile(self.plan,posted_prefix(self.plan,0),j)
        self.assertEqual(g.reconcile(self.plan,posted_prefix(self.plan,1),j)[0],1)

    def test_protected_product_and_suffix(self):
        self.assertNotIn(112,{r['product_id'] for r in self.plan['adjustments']})
        self.assertTrue(all(r['reason'].endswith(g.SUFFIX) for r in self.plan['adjustments']))
        self.plan['adjustments'][0]['product_id']=112
        with self.assertRaises(RuntimeError):g.validate(self.plan,self.plan['snapshot'])

    def test_flag_refused_before_query_or_credentials(self):
        with patch.object(g,'query',side_effect=AssertionError('query forbidden')), patch.object(g,'get_shared_key',side_effect=AssertionError('key forbidden')):
            with self.assertRaisesRegex(RuntimeError,'allow-shared-key'):g.apply(self.plan,False)

    def test_default_dry_run_no_api(self):
        with patch.object(g,'query',return_value=posted_prefix(self.plan,0)), patch.object(g,'api',side_effect=AssertionError('HTTP forbidden')), patch.object(g,'get_shared_key',side_effect=AssertionError('key forbidden')), patch('sys.argv',['script']), contextlib.redirect_stdout(io.StringIO()):
            g.main()

    def test_full_execution_and_second_run_no_http(self):
        count=0; commits=[]
        def fake_api(key,path,body=None):
            nonlocal count
            if path=='/auth/whoami':return dict(key_kind='legacy_ledger',actor=None)
            # Locate exact request, including the distinct reason on a split lot.
            r=next(r for r in self.plan['adjustments'] if g.payload(r,body['mode'])==body)
            f=posted_prefix(self.plan,count)
            lb=next(l['balance'] for l in f['lots'] if l['lot_id']==r['lot_id'])
            if body['mode']=='commit':
                self.assertEqual(r,self.plan['adjustments'][count])
                self.assertEqual(lb,g.D(r['before']))
                count+=1;commits.append(r['lot_id'])
            return dict(mode=body['mode'],success=True,transaction_id=8999+count,
                        product_id=r['product_id'],lot_code=r['lot_code'],
                        current_quantity_lb=lb,adjustment_lb=g.D(r['change']),
                        new_balance_lb=lb+g.D(r['change']),reason=r['reason'])
        with tempfile.TemporaryDirectory() as tmp, patch.object(g,'JOURNAL',Path(tmp)/'journal.json'), patch.object(g,'LOCK',Path(tmp)/'lock'), patch.object(g,'query',side_effect=lambda _:posted_prefix(self.plan,count)), patch.object(g,'get_shared_key',return_value='test'), patch.object(g,'api',side_effect=fake_api), contextlib.redirect_stdout(io.StringIO()):
            g.apply(self.plan,True)
            self.assertEqual(count,24)
            with patch.object(g,'api',side_effect=AssertionError('HTTP forbidden')), patch.object(g,'get_shared_key',side_effect=AssertionError('key forbidden')):
                g.apply(self.plan,True)
            self.assertEqual(len(commits),24)

    def test_error_stops_and_rerun_cannot_repeat_unresolved_request(self):
        calls=[]
        def fake_api(key,path,body=None):
            if path=='/auth/whoami':return dict(key_kind='legacy_ledger',actor=None)
            r=next(r for r in self.plan['adjustments'] if g.payload(r,body['mode'])==body)
            if body['mode']=='commit':
                calls.append(body)
                raise RuntimeError('simulated timeout')
            lb=next(l['balance'] for l in self.plan['snapshot']['lots'] if l['lot_id']==r['lot_id'])
            return dict(mode='preview',product_id=r['product_id'],lot_code=r['lot_code'],current_quantity_lb=lb,adjustment_lb=g.D(r['change']),new_balance_lb=lb+g.D(r['change']),reason=r['reason'])
        with tempfile.TemporaryDirectory() as tmp, patch.object(g,'JOURNAL',Path(tmp)/'journal.json'), patch.object(g,'LOCK',Path(tmp)/'lock'), patch.object(g,'query',return_value=posted_prefix(self.plan,0)), patch.object(g,'get_shared_key',return_value='test'), patch.object(g,'api',side_effect=fake_api), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'simulated timeout'):g.apply(self.plan,True)
            with self.assertRaisesRegex(RuntimeError,'Do not retry'):g.apply(self.plan,True)
            self.assertEqual(len(calls),1)


if __name__ == '__main__':
    unittest.main()
