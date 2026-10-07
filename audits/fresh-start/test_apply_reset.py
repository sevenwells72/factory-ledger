"""Offline only: real parser/preview/verifier with a fake ledger and mocked HTTP.
All synthetic files, keys, receipts and reports live in temporary directories
under fresh-start and are deleted, including on failed assertions.
"""
import contextlib
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import apply_reset as app
import fresh_start_common as f
from reset_preview import write_preview

REAL_REQUEST = app.LiveBackend.request


class FakeBackend:
    def __init__(self, s):
        self.s = s
        self.calls = []
        self.mode = None
        self.commits = 0
        self.reads = 0
        self.actor_name = 'FAKE Owner'
        self.drift_read = None
        self.fail_reads = set()

    def actor(self, key):
        return self.actor_name

    def snapshot(self):
        self.reads += 1
        if self.reads in self.fail_reads:
            raise OSError('FAKE read-only failure')
        if self.drift_read == self.reads:
            self.s['lines'][0]['quantity_lb'] += 1
        self.s['snapshot_at'] = datetime.now(timezone.utc).isoformat()
        return deepcopy(self.s)

    def request(self, key, route, body):
        self.calls.append((route, deepcopy(body)))
        if route == '/auth/whoami':
            if self.mode == 'identity_timeout':
                raise TimeoutError('FAKE identity read failure')
            if self.mode == 'shared':
                return dict(key_kind='legacy_ledger', actor=None)
            return dict(key_kind='actor', actor=dict(name=self.actor_name))
        row = next(l for l in self.s['lots'] if l['lot_code'] == body['lot_code'])
        current = sum(x['quantity_lb'] for x in self.s['lines'] if x['lot_id'] == row['id'] and x['effective_status'] == 'posted')
        delta = f.D(str(body['adjustment_lb']))
        result = dict(mode=body['mode'], product_id=row['product_id'], lot_code=row['lot_code'],
                      reason=body['reason'], adjustment_lb=delta, current_quantity_lb=current, new_balance_lb=current+delta)
        if body['mode'] == 'preview':
            if self.mode == 'bad_preview':
                result['product_id'] = 99999
            if self.mode == 'preview_timeout':
                raise TimeoutError('FAKE SECRET MUST NOT LEAK')
            return result
        self.commits += 1
        if self.mode == 'timeout_before':
            raise TimeoutError('FAKE SECRET MUST NOT LEAK')
        tid = 100 + self.commits
        line = dict(line_id=tid, transaction_id=tid, product_id=row['product_id'], lot_id=row['id'],
                    quantity_lb=delta, occurred_at=body['occurred_at'], transaction_created_at=datetime.now(timezone.utc).isoformat(),
                    line_created_at=datetime.now(timezone.utc).isoformat(), transaction_created_at_source='database', line_created_at_source='database',
                    effective_status='posted', type='adjust', adjust_reason=body['reason'], operator_id=self.actor_name,
                    notes='FAKE adjustment', line_correction_at=None, transaction_correction_at=None)
        if self.mode == 'wrong_quantity':
            line['quantity_lb'] += f.D('0.000001')
        if self.mode == 'wrong_actor':
            line['operator_id'] = 'legacy-shared-key'
        self.s['lines'].append(line)
        self.s['transaction_line_counts'][str(tid)] = 2 if self.mode == 'extra_line' else 1
        if self.mode == 'duplicate':
            self.s['lines'].append(dict(line, line_id=tid+1000, transaction_id=tid+1000))
            self.s['transaction_line_counts'][str(tid+1000)] = 1
        if self.mode in ('timeout_after', 'crash_after'):
            self.mode = None
            if self.crash:
                raise KeyboardInterrupt()
            raise TimeoutError('FAKE SECRET MUST NOT LEAK')
        result.update(success=True, transaction_id=tid)
        if self.mode == 'bad_commit':
            result['product_id'] = 99999
        if self.mode == 'wrong_receipt':
            result['transaction_id'] += 999
        return result

    @property
    def crash(self):
        return getattr(self, 'crash_enabled', False)


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.FAKE-apply-test-', dir=f.OUT)
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.count = self.out / 'FAKE-count.csv'
        self.approval = self.out / 'FAKE-approval.md'
        self.key = self.out / 'FAKE-personal-key'
        # Valid key file format; never any real credential.
        self.key.write_text('FAKE_SECRET_MUST_NOT_LEAK')
        self.key.chmod(0o600)
        self.state = self.out / 'state'
        for name, value in [('KEY_FILE', self.key), ('JOURNAL_ROOT', self.state), ('LOCK', self.out / 'lock'), ('RESULTS', self.out / 'results')]:
            ctx = patch.object(app, name, value); ctx.start(); self.addCleanup(ctx.stop)
        # Even an accidental use of a production adapter fails this test suite.
        for target in ('socket.create_connection', 'socket.socket.connect', 'subprocess.run', 'fresh_start_common.snapshot',
                       'apply_reset.LiveBackend.request', 'apply_reset.LiveBackend.snapshot', 'apply_reset.LiveBackend.actor'):
            ctx = patch(target, side_effect=AssertionError('LIVE ACCESS FORBIDDEN IN TESTS')); ctx.start(); self.addCleanup(ctx.stop)
        products = json.loads((f.OUT/'scope-manifest-v2.json').read_text())['products']
        self.cut = datetime.now(timezone.utc) - timedelta(hours=2)
        self.before = (self.cut-timedelta(hours=1)).isoformat()
        self.s = dict(snapshot_at=(self.cut+timedelta(minutes=1)).isoformat(), products=products,
                      catalog=products, lots=[], lines=[], corrections=[], transaction_line_counts={})
        rows = []
        for p in products:
            base = {'product id':p['id'], 'SKU':p['odoo_code'], 'name':p['name'], 'area':f.area(p), 'count unit':f.unit(p)}
            rows.append(dict(base, **{'paper row':f"P{p['id']}-C", 'product checked':f.CHECKED_LABEL}))
            if p['id'] in (107,108):
                lid = p['id']; qty = 10 if lid == 107 else 20; target = 6 if lid == 107 else 25
                code = f'FAKE-{lid}'
                self.s['lots'].append(dict(id=lid,product_id=lid,lot_code=code,status='active',merged_into_lot_id=None))
                self.s['lines'].append(dict(line_id=lid,transaction_id=lid,product_id=lid,lot_id=lid,quantity_lb=f.D(qty),
                    occurred_at=self.before,transaction_created_at=self.before,line_created_at=self.before,
                    transaction_created_at_source='database',line_created_at_source='database',effective_status='posted',
                    type='make',adjust_reason=None,operator_id='FAKE Owner',notes='FAKE',line_correction_at=None,transaction_correction_at=None))
                rows.append(dict(base,**{'paper row':f'P{lid}-001','lot code':code,'counted qty':target}))
        f.write_csv(self.count,f.V2_COLUMNS,rows)
        with contextlib.redirect_stdout(io.StringIO()):
            write_preview(f.analyze(self.s,self.count,self.cut),self.out)
        self.preview = next(self.out.glob('reset-preview-*.csv'))
        self.sign()
        self.journal = self.state / app.digest(self.preview.read_bytes()) / 'journal.jsonl'
        self.backend = FakeBackend(self.s)

    def sign(self, ids='107,108', **overrides):
        fields = {'Preview file':self.preview.name,'SHA-256':app.digest(self.preview.read_bytes()),
                  'Approved product IDs':ids,'Owner name':'FAKE Owner','Named actor':'FAKE Owner',
                  'Date':datetime.now().date().isoformat(),'Signature':'FAKE Owner','Opening entry classification':'N'}
        fields.update(overrides)
        self.approval.write_text('\n'.join(f'{k}: {v}' for k,v in fields.items())+'\n')

    def run_tool(self, apply=True, confirmation=None):
        out = io.StringIO()
        plan = app.load_plan(self.preview,self.approval,self.count)
        phrase = f'APPLY OPENING BALANCE {self.cut.astimezone(f.PLANT).date()} {plan["sha"]}'
        with contextlib.redirect_stdout(out):
            result = app.run(self.preview,self.approval,self.count,apply=apply,backend=self.backend,
                             input_fn=lambda prompt: phrase if confirmation is None else confirmation)
        self.assertNotIn(self.key.read_text() if self.key.exists() else 'FAKE_SECRET_MUST_NOT_LEAK',out.getvalue())
        return result,out.getvalue()

    def test_dry_run_no_requests_no_journal(self):
        code,out=self.run_tool(False)
        self.assertEqual(code,0);self.assertIn('DRY RUN ONLY',out)
        self.assertEqual(self.backend.calls,[]);self.assertFalse(self.journal.exists())

    def test_real_flow_mocked_and_rerun_no_duplicates(self):
        code,out=self.run_tool()
        self.assertEqual(code,0);self.assertIn('2 lots counted, 2 reconciled, 0 unexplained differences',out)
        self.assertEqual(self.backend.commits,2)
        self.assertNotIn('FAKE_SECRET_MUST_NOT_LEAK',self.journal.read_text())
        requests=len(self.backend.calls)
        self.run_tool();self.assertEqual(len(self.backend.calls),requests)
        self.assertEqual(self.backend.commits,2)
        lines=[json.loads(x) for x in self.journal.read_text().splitlines()]
        for i,e in enumerate(lines):
            if e['kind']=='receipt':self.assertEqual(lines[e['intent']]['kind'],'intent');self.assertLess(e['intent'],i)

    def test_hash_mismatch(self):
        self.preview.write_text(self.preview.read_text()+'\n')
        with self.assertRaisesRegex(RuntimeError,'SHA-256'):self.run_tool()
        self.assertEqual(self.backend.calls,[])

    def test_unsigned_approval(self):
        self.sign(Signature='')
        with self.assertRaisesRegex(RuntimeError,'incomplete'):self.run_tool()

    def test_unapproved_product(self):
        self.sign(ids='107')
        with self.assertRaisesRegex(RuntimeError,'not approved'):self.run_tool()

    def test_changed_count_file(self):
        self.count.write_text(self.count.read_text()+'\n')
        with self.assertRaisesRegex(RuntimeError,'Count CSV differs'):self.run_tool()

    def test_wrong_confirmation_no_http(self):
        with self.assertRaisesRegex(RuntimeError,'Confirmation'):self.run_tool(confirmation='yes')
        self.assertEqual(self.backend.calls,[])

    def test_missing_key_no_http(self):
        self.key.unlink()
        with self.assertRaisesRegex(RuntimeError,'key file'):self.run_tool(False)
        self.assertEqual(self.backend.calls,[])

    def test_wrong_registry_actor_no_http(self):
        self.backend.actor_name='legacy-shared-key'
        with self.assertRaisesRegex(RuntimeError,'named actor'):self.run_tool()
        self.assertEqual(self.backend.calls,[])

    def test_master_key_whoami_refused_before_adjust(self):
        self.backend.mode='shared'
        with self.assertRaisesRegex(RuntimeError,'Read-only API check'):self.run_tool()
        self.assertEqual(len(self.backend.calls),1);self.assertEqual(self.backend.commits,0)

    def test_balance_drift_any_lot_no_requests(self):
        self.s['lines'][1]['quantity_lb']+=1
        with self.assertRaisesRegex(RuntimeError,'balance drift'):self.run_tool()
        self.assertEqual(self.backend.calls,[])

    def test_drift_after_previews_before_first_commit(self):
        self.backend.drift_read=3
        with self.assertRaisesRegex(RuntimeError,'balance drift'):self.run_tool()
        self.assertEqual(self.backend.commits,0)

    def test_new_lot_stops(self):
        self.s['lots'].append(dict(id=999,product_id=107,lot_code='FAKE-NEW',status='active',merged_into_lot_id=None))
        with self.assertRaisesRegex(RuntimeError,'new, missing'):self.run_tool()

    def test_lookalike_never_matches(self):
        self.s['lots'][0]['lot_code']='fake-107'
        with self.assertRaisesRegex(RuntimeError,'identity'):self.run_tool()

    def test_exact_duplicate_sku_refused(self):
        sku=next(p['odoo_code'] for p in self.s['products'] if p['id']==107)
        self.s['catalog']=self.s['products']+[dict(id=99999,name='FAKE unrelated name',odoo_code=sku)]
        with self.assertRaisesRegex(RuntimeError,'not unique'):self.run_tool()

    def test_bad_preview_no_commit(self):
        self.backend.mode='bad_preview'
        with self.assertRaisesRegex(RuntimeError,'Read-only API check'):self.run_tool()
        self.assertEqual(self.backend.commits,0)

    def test_preview_timeout_does_not_block_later_run(self):
        self.backend.mode='preview_timeout'
        with self.assertRaises(RuntimeError):self.run_tool()
        self.assertFalse(self.journal.exists());self.assertEqual(self.backend.commits,0)
        self.backend.mode=None
        self.assertEqual(self.run_tool()[0],0);self.assertEqual(self.backend.commits,2)

    def test_timeout_before_commit_not_retried(self):
        self.backend.mode='timeout_before'
        with self.assertRaises(RuntimeError):self.run_tool()
        n=len(self.backend.calls);self.backend.mode=None
        with self.assertRaisesRegex(RuntimeError,'0 matching candidates'):self.run_tool()
        self.assertEqual(len(self.backend.calls),n)

    def test_timeout_after_commit_reconciles_and_resumes(self):
        self.backend.mode='timeout_after'
        with self.assertRaises(RuntimeError):self.run_tool()
        self.assertEqual(self.backend.commits,1)
        self.assertEqual(self.run_tool()[0],0);self.assertEqual(self.backend.commits,2)

    def test_process_crash_after_commit_reconciles_intent_only(self):
        self.backend.mode='crash_after';self.backend.crash_enabled=True
        with self.assertRaises(KeyboardInterrupt):self.run_tool()
        self.assertEqual(self.run_tool()[0],0);self.assertEqual(self.backend.commits,2)

    def test_wrong_actor_stops_after_first_post_and_on_rerun(self):
        self.backend.mode='wrong_actor'
        with self.assertRaisesRegex(RuntimeError,'wrong actor'):self.run_tool()
        self.assertEqual(self.backend.commits,1)
        n=len(self.backend.calls)
        with self.assertRaises(RuntimeError):self.run_tool()
        self.assertEqual(len(self.backend.calls),n)

    def test_duplicate_post_stops_and_never_retries(self):
        self.backend.mode='duplicate'
        with self.assertRaisesRegex(RuntimeError,'2 matching candidates'):self.run_tool()
        self.assertEqual(self.backend.commits,1)
        with self.assertRaises(RuntimeError):self.run_tool()
        self.assertEqual(self.backend.commits,1)

    def test_foreign_receipt_stops(self):
        self.backend.mode='wrong_receipt'
        with self.assertRaisesRegex(RuntimeError,'receipt and ledger'):self.run_tool()
        self.assertEqual(self.backend.commits,1)

    def test_extra_transaction_line_stops(self):
        self.backend.mode='extra_line'
        with self.assertRaisesRegex(RuntimeError,'multiple lines'):self.run_tool()
        self.assertEqual(self.backend.commits,1)

    def test_bad_response_but_exact_post_recovers(self):
        self.backend.mode='bad_commit'
        with self.assertRaises(RuntimeError):self.run_tool()
        self.backend.mode=None
        self.assertEqual(self.run_tool()[0],0);self.assertEqual(self.backend.commits,2)

    def test_torn_journal_stops(self):
        self.journal.parent.mkdir(parents=True)
        self.journal.write_text('{"partial":')
        with self.assertRaisesRegex(RuntimeError,'torn'):self.run_tool()
        self.assertEqual(self.backend.calls,[])

    def test_journal_other_approval_stops(self):
        self.run_tool()
        self.approval.write_text(self.approval.read_text()+'\n')
        with self.assertRaisesRegex(RuntimeError,'another approval'):self.run_tool()
        self.assertEqual(self.backend.commits,2)

    def test_foreign_opening_event_not_adopted_without_intent(self):
        x=dict(self.s['lines'][0],line_id=999,transaction_id=999,type='adjust',adjust_reason=f.opening_balance_reason(self.cut),
               occurred_at=self.cut.isoformat(),transaction_created_at=app.now(),line_created_at=app.now(),quantity_lb=f.D(-4))
        self.s['lines'].append(x)
        with self.assertRaisesRegex(RuntimeError,'balance drift'):self.run_tool()
        self.assertEqual(self.backend.calls,[])

    def test_offsetting_late_entries_still_stop(self):
        for tid,qty in ((500,1),(501,-1)):
            self.s['lines'].append(dict(self.s['lines'][0],line_id=tid,transaction_id=tid,quantity_lb=f.D(qty),
                                       transaction_created_at=app.now(),line_created_at=app.now()))
        with self.assertRaisesRegex(RuntimeError,'New or undated'):self.run_tool()

    def test_corrected_transaction_stops_even_without_balance_change(self):
        self.s['lines'][0]['transaction_correction_at']=app.now()
        with self.assertRaisesRegex(RuntimeError,'corrected after preview'):self.run_tool()

    def reissue(self, confirmations=None):
        with contextlib.redirect_stdout(io.StringIO()):
            write_preview(f.analyze(self.s, self.count, self.cut, confirmations), self.out)
        self.sign()
        self.journal = self.state / app.digest(self.preview.read_bytes()) / 'journal.jsonl'

    def test_finished_cases_and_partial_case_use_approved_pounds(self):
        p = next(p for p in self.s['products'] if p['id']==150)
        self.s['lots'].append(dict(id=150,product_id=150,lot_code='FAKE-150',status='active',merged_into_lot_id=None))
        self.s['lines'].append(dict(self.s['lines'][0],line_id=150,transaction_id=150,product_id=150,lot_id=150,quantity_lb=f.D(10)))
        with self.count.open() as stream: rows=list(csv.DictReader(stream))
        rows.append({'paper row':'P150-001','product id':150,'SKU':p['odoo_code'],'name':p['name'],
                     'area':f.area(p),'count unit':'cases','lot code':'FAKE-150','counted qty':'2','partial cases':'5 of 6'})
        f.write_csv(self.count,f.V2_COLUMNS,rows);self.reissue();self.sign(ids='107,108,150')
        code,out=self.run_tool();self.assertEqual(code,0)
        bodies=[b for _,b in self.backend.calls if b and b['mode']=='commit' and b['lot_code']=='FAKE-150']
        self.assertEqual(bodies[0]['adjustment_lb'],-2.5625)
        self.assertIn('3 lots counted, 3 reconciled',out)

    def test_zero_merged_alias_not_posted_or_held(self):
        self.s['lots'].append(dict(id=900,product_id=107,lot_code='FAKE-OLD-Lot',status='merged',merged_into_lot_id=107))
        self.reissue();self.assertEqual(self.run_tool()[0],0)
        self.assertEqual(self.backend.commits,2)

    def test_approved_late_entry_answers_carried_to_verifier(self):
        entered=(self.cut+timedelta(seconds=10)).isoformat()
        self.s['lines'].append(dict(self.s['lines'][0],line_id=901,transaction_id=901,quantity_lb=f.D(2),
            line_created_at=entered,transaction_created_at=entered))
        a=f.analyze(self.s,self.count,self.cut)
        path=f.write_late_review(a,self.out)
        with path.open() as stream: rows=list(csv.DictReader(stream))
        for row in rows: row['happened after count']='N';row['owner initials']='FAKE'
        f.write_csv(path,f.LATE_COLUMNS,rows);self.reissue(path)
        code,out=self.run_tool();self.assertEqual(code,0)
        self.assertIn('2 lots counted, 2 reconciled',out)

    def test_changed_case_weight_requires_new_approval(self):
        # Change an approved row's source count rather than silently trusting CSV math.
        with self.preview.open() as stream: rows=list(csv.DictReader(stream)); columns=list(rows[0])
        r=next(r for r in rows if r['row_type']=='LOT_REVIEW' and r['product_id']=='107')
        r.update(counted_lb='7',candidate_adjustment_lb='-3',rounded_adjustment_lb='-3',
                 adjustment_lb='-3',expected_current_lb='7',planned_balance_lb='7',rounding_delta_lb='0')
        f.write_csv(self.preview,columns,rows);self.sign()
        with self.assertRaisesRegex(RuntimeError,'Count, case weight'):self.run_tool()
        self.assertEqual(self.backend.calls,[])

    def test_voided_completed_event_never_reposted(self):
        self.run_tool()
        next(x for x in self.s['lines'] if x['type']=='adjust')['effective_status']='voided'
        with self.assertRaisesRegex(RuntimeError,'status'):self.run_tool()
        self.assertEqual(self.backend.commits,2)

    def test_intent_write_failure_prevents_commit(self):
        original=app.Journal.append
        def fail(journal,**event):
            if event.get('kind')=='intent' and event.get('phase')=='commit':
                raise OSError('FAKE disk full')
            return original(journal,**event)
        with patch.object(app.Journal,'append',fail):
            with self.assertRaises(OSError):self.run_tool()
        self.assertEqual(self.backend.commits,0)

    def test_receipt_write_failure_resumes_from_durable_intent(self):
        original=app.Journal.append
        def fail(journal,**event):
            if event.get('kind')=='receipt' and event.get('transaction_id'):
                raise OSError('FAKE disk full')
            return original(journal,**event)
        with patch.object(app.Journal,'append',fail):
            with self.assertRaises(OSError):self.run_tool()
        self.assertEqual(self.backend.commits,1)
        self.assertEqual(self.run_tool()[0],0);self.assertEqual(self.backend.commits,2)

    def test_lock_refuses_concurrent_process(self):
        with patch.object(app.fcntl,'flock',side_effect=BlockingIOError):
            with self.assertRaisesRegex(RuntimeError,'holds the lock'):self.run_tool()
        self.assertEqual(self.backend.calls,[])

    def test_interactive_confirmation_required_by_cli(self):
        args=['apply_reset.py',str(self.preview),'--count-csv',str(self.count),'--apply']
        with patch.object(app.sys,'argv',args),patch.object(app.sys.stdin,'isatty',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'interactive terminal'):app.main()

    def test_private_key_permissions_required(self):
        self.key.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError,'private'):self.run_tool(False)
        self.assertEqual(self.backend.calls,[])

    def test_journal_corruption_refused(self):
        self.run_tool()
        raw=self.journal.read_text().replace('"version":1','"version":2')
        self.journal.write_text(raw)
        with self.assertRaisesRegex(RuntimeError,'chain mismatch'):self.run_tool()
        self.assertEqual(self.backend.commits,2)

    def test_readonly_adapter_uses_authorized_wrapper_and_transactions(self):
        credential=self.out/'FAKE-db-url';credential.write_text('postgresql://FAKE.invalid:5432/db')
        completed=type('Result',(),{'returncode':0,'stdout':'{}'})()
        with patch.object(f,'CREDENTIAL',credential),patch.object(app.subprocess,'run',return_value=completed) as mock:
            app.LiveBackend.read_json(app.APPLY_SNAPSHOT_SQL)
        args,kwargs=mock.call_args
        self.assertEqual(args[0][0],str(f.WRAPPER))
        self.assertIn('BEGIN;',kwargs['input']);self.assertIn('SET TRANSACTION READ ONLY;',kwargs['input'])
        self.assertTrue(kwargs['input'].strip().endswith('ROLLBACK;'))
        self.assertNotIn('PGOPTIONS',kwargs['env'])
        self.assertIn("'catalog'",kwargs['input']);self.assertIn("'transaction_line_counts'",kwargs['input'])
        self.assertNotIn('FAKE_SECRET_MUST_NOT_LEAK',str(mock.call_args))

    def test_readonly_adapter_refuses_supplied_sql(self):
        with self.assertRaisesRegex(RuntimeError,'fixed read-only'):
            app.LiveBackend.read_json('SELECT 1;')

    def test_http_transport_is_pinned_and_never_retries(self):
        # Exercise the actual transport function with a mocked opener only.
        from unittest.mock import Mock
        client=Mock();client.open.side_effect=TimeoutError('FAKE_SECRET_MUST_NOT_LEAK')
        # setUp prevents using request(), so unwrap via the saved class source is
        # unnecessary: use the unpatched function captured below at module load.
        with patch.object(app,'build_opener',return_value=client):
            with self.assertRaisesRegex(RuntimeError,'details suppressed') as error:
                REAL_REQUEST(app.LiveBackend(),'FAKE_SECRET_MUST_NOT_LEAK','/auth/whoami',None)
        self.assertEqual(client.open.call_count,1)
        req=client.open.call_args.args[0]
        self.assertEqual(req.full_url,app.API+'/auth/whoami')
        self.assertEqual(req.get_method(),'GET')
        self.assertNotIn('FAKE_SECRET_MUST_NOT_LEAK',str(error.exception))

    def test_inactive_product_held_in_apply_dry_run_after_preview(self):
        next(p for p in self.s['products'] if p['id']==107)['active']=False
        with self.assertRaisesRegex(RuntimeError,'HELD – inactive in FL, owner decision needed'):
            self.run_tool(False)
        self.assertEqual(self.backend.calls,[])

    def test_inactive_product_cannot_be_signed_into_ready_group(self):
        next(p for p in self.s['products'] if p['id']==107)['active']=False
        self.reissue()
        with self.assertRaisesRegex(RuntimeError,'inactive in FL'):
            self.run_tool(False)
        self.assertEqual(self.backend.calls,[])

    def test_four_decimal_plan_posts_rounded_positive_and_negative(self):
        with self.count.open() as stream:rows=list(csv.DictReader(stream))
        for r in rows:
            if r['lot code']=='FAKE-107':r['counted qty']='6.123456'
            if r['lot code']=='FAKE-108':r['counted qty']='25.123456'
        f.write_csv(self.count,f.V2_COLUMNS,rows);self.reissue()
        plan=app.load_plan(self.preview,self.approval,self.count)
        self.assertEqual([r['change'] for r in plan['rows']], [f.D('-3.8765'), f.D('5.1235')])
        self.assertEqual(self.run_tool()[0],0)
        posts=[b['adjustment_lb'] for _,b in self.backend.calls if b and b['mode']=='commit']
        self.assertEqual(posts,[-3.8765,5.1235])
        self.run_tool();self.assertEqual(self.backend.commits,2)

    def test_half_quantum_residual_does_not_propose_second_adjustment(self):
        self.s['lines'][0]['quantity_lb']=f.D('10.00005');self.reissue()
        self.assertEqual(self.run_tool()[0],0)
        self.assertEqual(self.run_tool()[0],0);self.assertEqual(self.backend.commits,2)

    def test_readback_requires_exact_rounded_quantity_not_tolerance(self):
        self.backend.mode='wrong_quantity'
        with self.assertRaisesRegex(RuntimeError,'quantity'):self.run_tool()
        self.assertEqual(self.backend.commits,1)

    def test_identity_timeout_does_not_block_later_run(self):
        self.backend.mode='identity_timeout'
        with self.assertRaisesRegex(RuntimeError,'Read-only'):self.run_tool()
        self.assertFalse(self.journal.exists());self.backend.mode=None
        self.assertEqual(self.run_tool()[0],0)

    def test_read_failure_before_first_commit_does_not_lock_journal(self):
        self.backend.fail_reads={3}
        with self.assertRaises(OSError):self.run_tool()
        self.assertFalse(self.journal.exists());self.assertEqual(self.backend.commits,0)
        self.assertEqual(self.run_tool()[0],0);self.assertEqual(self.backend.commits,2)

    def test_readback_failure_after_successful_commit_resumes_from_receipt(self):
        self.backend.fail_reads={4}
        with self.assertRaises(OSError):self.run_tool()
        self.assertEqual(self.backend.commits,1)
        self.assertEqual(self.run_tool()[0],0);self.assertEqual(self.backend.commits,2)

    def test_legacy_unfinished_read_intent_does_not_lock_journal(self):
        plan=app.load_plan(self.preview,self.approval,self.count)
        journal=app.Journal(self.journal,plan);journal.start()
        journal.append(kind='intent',phase='preview',lot_id=107,product_id=107,route='/adjust',payload={})
        self.assertEqual(self.run_tool()[0],0)

    def test_mac_clock_ahead_cannot_reject_post_or_receiptless_resume(self):
        with patch.object(app,'now',return_value='2099-01-01T00:00:00+00:00'):
            self.backend.mode='timeout_after'
            with self.assertRaises(RuntimeError):self.run_tool()
            self.assertEqual(self.run_tool()[0],0);self.assertEqual(self.backend.commits,2)

    def test_mac_clock_behind_cannot_adopt_old_matching_transaction(self):
        self.s['lines'].append(dict(self.s['lines'][0],line_id=1000,transaction_id=1000,type='adjust',
            quantity_lb=f.D(-4),adjust_reason=f.opening_balance_reason(self.cut),occurred_at=self.cut.isoformat()))
        self.s['transaction_line_counts']['1000']=1
        with self.count.open() as stream:rows=list(csv.DictReader(stream))
        for r in rows:
            if r['lot code']=='FAKE-107':r['counted qty']='2'
        f.write_csv(self.count,f.V2_COLUMNS,rows);self.reissue()
        with patch.object(app,'now',return_value='2001-01-01T00:00:00+00:00'):
            self.assertEqual(self.run_tool()[0],0)
        self.assertEqual(self.backend.commits,2)
        intent=next(json.loads(line) for line in self.journal.read_text().splitlines() if '"kind":"intent"' in line)
        self.assertIn(1000,intent['known_transaction_ids'])

    def test_skus_893_and_1614_are_exact_fields_not_name_or_code_substrings(self):
        with self.count.open() as stream:rows=list(csv.DictReader(stream))
        for pid in (144,164):
            p=next(p for p in self.s['products'] if p['id']==pid)
            self.assertIn(p['odoo_code'],('893','1614'))
            self.s['lots'].append(dict(id=pid,product_id=pid,lot_code=f'FAKE-{pid}',status='active',merged_into_lot_id=None))
            self.s['lines'].append(dict(self.s['lines'][0],line_id=pid,transaction_id=pid,product_id=pid,lot_id=pid,quantity_lb=f.D(20)))
            rows.append({'paper row':f'P{pid}-001','product id':pid,'SKU':p['odoo_code'],'name':p['name'],
                         'area':f.area(p),'count unit':'cases','lot code':f'FAKE-{pid}','counted qty':'1'})
        self.s['catalog']=self.s['products']+[dict(id=99998,name='FAKE 893 in name',odoo_code='8930'),
                                                   dict(id=99999,name='FAKE 1614 in name',odoo_code='116140')]
        f.write_csv(self.count,f.V2_COLUMNS,rows);self.reissue();self.sign(ids='107,108,144,164')
        self.assertEqual(self.run_tool(False)[0],0)
        self.assertEqual(self.run_tool()[0],0)
        self.assertEqual(self.backend.commits,4)

    def test_reviewed_count_has_no_local_age_deadline(self):
        self.s['snapshot_at']=(self.cut+timedelta(days=15)).isoformat()
        plan=app.load_plan(self.preview,self.approval,self.count)
        with patch.object(app,'now',return_value='2001-01-01T00:00:00+00:00'):
            app.preflight(plan,self.s,app.Journal(self.journal,plan))
        self.assertEqual(self.backend.calls,[])

    def test_unapproved_inactive_adjustments_are_visible_in_dry_run(self):
        pid=130
        self.s['lots'].append(dict(id=pid,product_id=pid,lot_code='FAKE-INACTIVE',status='active',merged_into_lot_id=None))
        self.s['lines'].append(dict(self.s['lines'][0],line_id=pid,transaction_id=pid,product_id=pid,lot_id=pid,quantity_lb=f.D(5)))
        self.reissue()
        code,out=self.run_tool(False)
        self.assertEqual(code,0);self.assertIn('Product 130: '+f.INACTIVE_HOLD,out)
        self.assertIn('excluded from approved groups',out);self.assertEqual(self.backend.calls,[])

    def test_owner_excluded_product_cannot_be_approved_even_for_dry_run(self):
        for pid in (171,209):
            with self.subTest(pid=pid):
                self.sign(ids=f'107,108,{pid}')
                with self.assertRaisesRegex(RuntimeError,'excluded by the owner'):self.run_tool(False)
                self.assertEqual(self.backend.calls,[])

    def test_excluded_live_stock_does_not_block_apply_or_full_verification(self):
        old=json.loads((f.OUT/'issued-before-scope-exclusion-2026-10-05/scope-manifest-v2.json').read_text())
        for p in old['products']:
            if p['id'] in f.EXCLUDED_PRODUCT_IDS:
                self.s['products'].append(p)
                self.s['lots'].append(dict(id=p['id'],product_id=p['id'],lot_code='FAKE-EXCLUDED',status='merged',merged_into_lot_id=999))
                self.s['lines'].append(dict(self.s['lines'][0],product_id=p['id'],lot_id=p['id'],line_id=p['id'],transaction_id=p['id'],quantity_lb=f.D(900)))
        self.reissue()
        code,out=self.run_tool()
        self.assertEqual(code,0);self.assertIn('2 lots counted, 2 reconciled, 0 unexplained differences',out)
        self.assertEqual(self.backend.commits,2)
        allowed={p['odoo_code'] for p in self.s['products'] if p['id'] in (107,108)}
        self.assertTrue(all(b['product_name'] in allowed for _,b in self.backend.calls if b and b['mode']=='commit'))

    def test_old_preview_excluded_rows_are_never_proposed_or_held(self):
        with self.preview.open() as stream:rows=list(csv.DictReader(stream))
        exemplar=next(r for r in rows if r['row_type']=='LOT_REVIEW' and r['adjustment_lb'])
        for pid in (171,209):
            rows.append(dict(exemplar,product_id=str(pid),product_active='False',lot_id=str(pid),
                             lot_code='FAKE-EXCLUDED',group_status=f.INACTIVE_HOLD))
        f.write_csv(self.preview,app.COLUMNS,rows);self.sign()
        plan=app.load_plan(self.preview,self.approval,self.count)
        self.assertFalse(plan['held_inactive'])
        self.assertFalse({r['pid'] for r in plan['rows']} & f.EXCLUDED_PRODUCT_IDS)
        code,out=self.run_tool(False)
        self.assertEqual(code,0);self.assertNotIn('Product 171:',out);self.assertNotIn('Product 209:',out)
        self.assertEqual(self.backend.calls,[])

    def test_preflight_cannot_bypass_excluded_product_guard(self):
        plan=app.load_plan(self.preview,self.approval,self.count)
        plan['ids'].add(171)
        with self.assertRaisesRegex(RuntimeError,'Owner-excluded product'):
            app.preflight(plan,self.s,app.Journal(self.journal,plan))
        self.assertEqual(self.backend.calls,[])

    def test_owner_guides_cover_live_count_and_first_use_safeguards(self):
        text=(f.OUT/'apply-guide.md').read_text()
        for phrase in ('Dry run on production','one small active product, whole pounds',
                       'Check the recorded actor and dashboard balances','Only then approve and apply the rest',
                       'owner-approved direct database write','archive-185-execution.txt'):
            with self.subTest(phrase=phrase):self.assertIn(phrase,text)
        floor=(f.OUT/'README.md').read_text()
        self.assertIn('while production continues',floor)
        self.assertNotIn('Apply within 14 days',text+floor)

    def test_actor_registry_read_failure_does_not_leave_journal(self):
        with patch.object(self.backend,'actor',side_effect=OSError('FAKE registry unavailable')):
            with self.assertRaises(OSError):self.run_tool()
        self.assertFalse(self.journal.exists());self.assertEqual(self.backend.commits,0)
        self.assertEqual(self.run_tool()[0],0)

    def test_http_redirect_is_refused(self):
        with self.assertRaisesRegex(RuntimeError,'Redirect refused'):
            app.NoRedirect().redirect_request(None,None,302,None,None,'https://example.invalid')


if __name__=='__main__':
    unittest.main()
