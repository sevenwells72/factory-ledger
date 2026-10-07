"""Offline v3 reconciliation and executor safety tests. No live access; temp files removed."""
import unittest,tempfile,json,csv,contextlib,io,ast
from pathlib import Path
from datetime import datetime,timedelta,timezone
from copy import deepcopy
from unittest.mock import patch
import fresh_start_common as f
import fresh_start_v3 as v
import apply_reset as app
from test_apply_reset import FakeBackend

class V3Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='.FAKE-v3-',dir=f.OUT);self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)
  self.t=datetime.now(timezone.utc).replace(second=0,microsecond=0)-timedelta(hours=3)
  p=dict(id=1,odoo_code='11001',name='TEST ingredient',type='ingredient',active=True,is_service=False,uom='lb')
  self.s=dict(snapshot_at=(self.t+timedelta(hours=2)).isoformat(),products=[p],catalog=[p],lots=[dict(id=1,product_id=1,lot_code='TEST-LOT',status='active',merged_into_lot_id=None)],lines=[],corrections=[],transaction_line_counts={})
  self.line(100,self.t-timedelta(days=1))
  self.data=dict(count=[self.row()],coverage=[dict(product_id='1',SKU='11001',name='TEST ingredient',all_locations_searched='yes',completed_at=(self.t+timedelta(hours=1)).isoformat(),owner_initials='TEST',notes='')],moves=[],review={})
  for name,val in [('KEY_FILE',self.out/'key'),('JOURNAL_ROOT',self.out/'state'),('LOCK',self.out/'lock'),('RESULTS',self.out/'results')]:
   c=patch.object(app,name,val);c.start();self.addCleanup(c.stop)
  app.KEY_FILE.write_text('FAKE_SECRET_MUST_NOT_LEAK');app.KEY_FILE.chmod(0o600)
  for target in ('socket.create_connection','socket.socket.connect','subprocess.run','fresh_start_common.snapshot','fresh_start_v3.snapshot','apply_reset.LiveBackend.request','apply_reset.LiveBackend.actor'):
   c=patch(target,side_effect=AssertionError('LIVE ACCESS FORBIDDEN'));c.start();self.addCleanup(c.stop)
 def row(self,rid='R1',area='A',qty='90',time=None):
  t=time or self.t;r={k:'' for k in v.COUNT_COLUMNS}
  r.update(sheet_id=rid,area=area,count_date=t.date().isoformat(),counter='TEST',start_time=(t-timedelta(minutes=10)).isoformat(),end_time=(t+timedelta(minutes=10)).isoformat(),row_id=rid,product_id='1',SKU='11001',name='TEST ingredient',row_kind='COUNT',lot_code='TEST-LOT',quantity=qty,unit='lb',location=area+' shelf',counted_time=t.isoformat())
  return r
 def line(self,q,t,entered=None,status='posted'):
  i=len(self.s['lines'])+1;x=dict(line_id=i,transaction_id=i,product_id=1,lot_id=1,quantity_lb=f.D(q),occurred_at=t.isoformat(),transaction_created_at=(entered or t).isoformat(),line_created_at=(entered or t).isoformat(),transaction_created_at_source='database',line_created_at_source='database',type='receive' if q>0 else 'make',effective_status=status,adjust_reason=None,operator_id='TEST',notes='',line_correction_at=None,transaction_correction_at=None)
  self.s['lines'].append(x);self.s['transaction_line_counts'][str(i)]=1;return x
 def analyze(self):return v.analyze(self.s,self.data)
 def answer(self,classification='after count',late=True,allocation=None):
  a=self.analyze();reviews={}
  for fp,r in a['review_template']['movement_reviews'].items():
   reviews[fp]=dict(owner='TEST',classifications={k:classification for k in r['classifications']},late_confirmed=late,allocations=allocation or {})
  self.data['review']['movement_reviews']=reviews
 def target(self):return self.analyze()['rows'][0]
 def confirm_multi_area(self):
  self.data['review']['multi_area_reviews']={fp:dict(owner='TEST',no_unlogged_moves=True) for fp in self.analyze()['review_template']['multi_area_reviews']}
 def persist(self):
  for name,rows,cols in [('count',self.data['count'],v.COUNT_COLUMNS),('coverage',self.data['coverage'],list(self.data['coverage'][0])),('moves',self.data['moves'],['move_id','product_id','SKU','lot_code','quantity','unit','from_area','from_location','to_area','to_location','moved_at','tag_id','fl_transaction_id','notes'])]:
   with (self.out/(name+'.csv')).open('w',newline='') as fobj:
    w=csv.DictWriter(fobj,fieldnames=cols);w.writeheader();w.writerows(rows)
  (self.out/'review.json').write_text(json.dumps(self.data['review']))
  data=v.load_inputs(self.out/'count.csv',self.out/'coverage.csv',self.out/'moves.csv',self.out/'review.json')
  self.a=v.analyze(self.s,data);self.preview=self.out/'preview.json';self.preview.write_text(json.dumps(self.a,default=str))
  self.approval=self.out/'approval.md';self.approval.write_text('\n'.join([f'Preview file: {self.preview.name}',f'SHA-256: {app.digest(self.preview.read_bytes())}','Approved product IDs: 1','Owner name: FAKE Owner','Named actor: FAKE Owner',f'Date: {self.t.date()}','Signature: FAKE Owner','Opening entry classification: N']))
  return app.load_plan(self.preview,self.approval,self.out/'count.csv')
 def run_apply(self,backend,apply=True):
  plan=self.persist();phrase=f'APPLY OPENING BALANCE {plan["cutoff"].astimezone(f.PLANT).date()} {plan["sha"]}'
  with contextlib.redirect_stdout(io.StringIO()):return app.run(self.preview,self.approval,self.out/'count.csv',apply,backend,lambda _:phrase)
 def test_baseline_and_decimal_rounding(self):
  self.assertEqual(self.target()['adjustment_lb'],f.D(-10));self.data['count'][0]['quantity']='90.123456';self.assertEqual(self.target()['adjustment_lb'],f.D('-9.8765'))
 def test_root_resolves_inside_this_repository(self):
  repo=Path(__file__).resolve().parents[2]
  self.assertEqual(f.ROOT,repo);self.assertEqual(f.OUT,Path(f.__file__).resolve().parent)
  self.assertTrue(f.OUT.is_relative_to(repo));self.assertTrue((repo/'.git').exists());self.assertTrue((repo/'main.py').is_file())
  self.assertEqual(f.WRAPPER,repo/'scripts/psql_ro.sh')
 def test_inside_window_requires_before_after(self):
  self.line(-10,self.t+timedelta(minutes=1));self.assertEqual(self.target()['status'],'HELD');self.answer();self.assertEqual(self.target()['expected_current_lb'],80)
 def test_before_classification_changes_basis(self):
  self.line(-10,self.t+timedelta(minutes=1));self.answer('before count');self.assertEqual(self.target()['expected_current_lb'],90)
 def test_30_minute_margin(self):
  self.line(-10,self.t+timedelta(minutes=29));self.assertEqual(self.target()['status'],'HELD')
 def test_every_later_movement_included(self):
  self.line(10,self.t+timedelta(minutes=45));self.line(-4,self.t+timedelta(minutes=55));a=self.analyze();self.assertEqual(len(a['activity']),2);self.assertEqual(a['rows'][0]['expected_current_lb'],96)
 def test_lag_over_15_minutes_requires_confirmation(self):
  self.line(-10,self.t,entered=self.t+timedelta(minutes=16));self.answer(late=False);self.assertEqual(self.target()['status'],'HELD');self.answer(late=True);self.assertEqual(self.target()['status'],'READY')
 def test_backdated_entry_before_window(self):
  self.line(5,self.t-timedelta(days=2),entered=self.t+timedelta(minutes=20));self.assertEqual(self.target()['status'],'HELD');self.answer('before count');self.assertEqual(self.target()['status'],'READY')
 def test_multi_area_different_times_allocates_once(self):
  self.data['count']=[self.row(qty='40'),self.row('R2','B','40',self.t+timedelta(minutes=45))]
  self.line(-10,self.t+timedelta(minutes=20));self.assertEqual(self.target()['status'],'HELD');self.answer(allocation={'R1':'-10'});self.assertEqual(self.target()['expected_current_lb'],70)
  self.assertEqual(len(self.analyze()['row_comparison']),2)
 def test_multi_area_missing_allocation_held(self):
  self.data['count'].append(self.row('R2','B','10',self.t+timedelta(minutes=45)));self.line(-10,self.t+timedelta(minutes=20));self.answer();self.assertEqual(self.target()['status'],'HELD')
 def move(self,srcincluded,dstincluded):
  self.data['count']=[self.row(qty='40'),self.row('R2','B','40',self.t+timedelta(minutes=45))]
  m=dict(move_id='M1',product_id='1',SKU='11001',lot_code='TEST-LOT',quantity='10',unit='lb',from_area='A',from_location='A shelf',to_area='B',to_location='B shelf',moved_at=(self.t+timedelta(minutes=20)).isoformat(),tag_id='',fl_transaction_id='',notes='TEST internal move')
  self.data['moves']=[m];self.data['review']['move_reviews']={v.fingerprint(m):dict(owner='TEST',source_row='R1',destination_row='R2',source_included=srcincluded,destination_included=dstincluded,notes='checked both counts')}
 def test_move_counted_to_uncounted_removes_double(self):
  self.move(True,True);self.assertEqual(self.target()['expected_current_lb'],70)
 def test_move_uncounted_to_counted_restores_omission(self):
  self.move(False,False);self.assertEqual(self.target()['expected_current_lb'],90)
 def test_move_unresolved_held(self):
  self.move(True,True);self.data['review']={};self.assertEqual(self.target()['status'],'HELD')
 def test_move_mismatch_held(self):
  self.move(True,True);self.data['moves'][0]['from_area']='OTHER';self.assertEqual(self.target()['status'],'HELD')
 def test_duplicate_ticks_require_owner_review(self):
  self.data['count'][0]['duplicate']='x';self.assertEqual(self.target()['status'],'HELD')
 def test_fingerprint_invalidates_changed_movement(self):
  x=self.line(-10,self.t);self.answer();x['quantity_lb']=f.D(-11);self.assertEqual(self.target()['status'],'HELD')
 def test_estimated_in_reason(self):
  self.data['count'][0]['estimated']='yes';self.assertTrue(self.target()['estimated']);self.assertIn('Estimated=yes',self.target()['reason'])
 def test_packaging_native_units(self):
  self.s['products'][0].update(type='packaging',uom='unit');self.data['count'][0]['unit']='unit';a=self.analyze();self.assertEqual(a['rows'],[]);r=a['packaging_information_only'][0];self.assertEqual(r['unit'],'unit');self.assertEqual(r['quantity'],'90');self.assertNotIn('adjustment_lb',r)
 def test_no_package_pounds_assumption(self):
  self.s['products'][0].update(type='packaging',uom='unit');a=self.analyze();self.assertEqual(a['unidentified_follow_up'],[]);self.assertEqual(a['packaging_information_only'][0]['unit'],'lb');self.assertEqual(a['rows'],[])
 def test_count_time_falls_back_to_sheet_end(self):
  self.data['count'][0]['counted_time']='';self.assertEqual(self.target()['cutoff'],self.data['count'][0]['end_time'])
 def test_unknown_lot_held_seven_day_followup(self):
  self.data['count'][0]['lot_code']='';a=self.analyze();self.assertIn((self.t+timedelta(days=7)).date().isoformat(),a['unidentified_follow_up'][0]['follow_up_due']);self.assertFalse(a['full_scope_reviewed'])
 def test_missing_coverage_never_zeroes_other_lots(self):
  self.s['lots'].append(dict(id=2,product_id=1,lot_code='OTHER',status='active',merged_into_lot_id=None));self.data['coverage'][0]['all_locations_searched']='';self.assertEqual(len(self.analyze()['rows']),1);self.assertEqual(self.target()['status'],'HELD')
 def test_completed_coverage_explicit_absence(self):
  self.s['lots'].append(dict(id=2,product_id=1,lot_code='OTHER',status='active',merged_into_lot_id=None));self.assertEqual(len(self.analyze()['rows']),2)
 def test_inactive_service_negative_hold(self):
  self.s['products'][0]['active']=False;self.assertEqual(self.target()['status'],'HELD')
 def test_correction_hold(self):
  self.s['corrections']=[dict(id=1,target_table='transactions',target_id=1,created_at=(self.t+timedelta(minutes=20)).isoformat(),reason='test')];self.assertEqual(self.target()['status'],'HELD')
 def test_dryrun_has_no_api(self):
  b=FakeBackend(self.s);self.assertEqual(self.run_apply(b,False),0);self.assertEqual(b.calls,[])
 def test_apply_verify_named_actor_and_per_lot_cutoff(self):
  b=FakeBackend(self.s);self.assertEqual(self.run_apply(b),0);self.assertEqual(b.commits,1);body=next(body for route,body in b.calls if body and body['mode']=='commit');self.assertEqual(body['occurred_at'],self.t.isoformat());self.assertTrue(body['backfill'])
 def test_old_reviewed_count_has_no_age_deadline(self):
  self.t-=timedelta(days=30);self.data['count']=[self.row()];self.data['coverage'][0]['completed_at']=(self.t+timedelta(hours=1)).isoformat();self.s['lines'][0]['occurred_at']=(self.t-timedelta(days=1)).isoformat();self.s['lines'][0]['line_created_at']=self.s['lines'][0]['transaction_created_at']=self.s['lines'][0]['occurred_at'];b=FakeBackend(self.s);self.assertEqual(self.run_apply(b,False),0)
 def test_drift_refuses_write(self):
  plan=self.persist();self.s['lines'][0]['quantity_lb']+=1
  with self.assertRaisesRegex(RuntimeError,'changed since approval'):app.preflight(plan,self.s,app.Journal(self.out/'journal',plan))
 def test_modified_review_refused(self):
  self.persist();(self.out/'review.json').write_text('{"changed":true}')
  with self.assertRaisesRegex(RuntimeError,'source changed'):app.load_plan(self.preview,self.approval,self.out/'count.csv')
 def test_held_movement_cannot_apply(self):
  self.line(-10,self.t)
  with self.assertRaisesRegex(RuntimeError,'Held lot'):self.persist()
 def test_uncertain_commit_does_not_retry(self):
  b=FakeBackend(self.s);b.mode='timeout_after'
  plan=self.persist();phrase=f'APPLY OPENING BALANCE {plan["cutoff"].astimezone(f.PLANT).date()} {plan["sha"]}'
  with contextlib.redirect_stdout(io.StringIO()):
   with self.assertRaisesRegex(RuntimeError,'uncertain'):app.run(self.preview,self.approval,self.out/'count.csv',True,b,lambda _:phrase)
   app.run(self.preview,self.approval,self.out/'count.csv',True,b,lambda _:phrase)
  self.assertEqual(b.commits,1)
 def test_malformed_clock_future_and_duplicate_rows(self):
  self.data['count'][0]['counted_time']=(self.t+timedelta(days=2)).isoformat();self.assertFalse(self.analyze()['full_scope_reviewed'])
  self.data['count']=[self.row(),self.row()]
  with self.assertRaisesRegex(ValueError,'Duplicate'):self.analyze()
 def test_standalone_readonly_journal_verification(self):
  b=FakeBackend(self.s);self.assertEqual(self.run_apply(b),0)
  plan=app.load_plan(self.preview,self.approval,self.out/'count.csv')
  ids=v.journal_openings(self.preview,self.approval,app.JOURNAL_ROOT/plan['sha']/'journal.jsonl',self.s,plan['data'])
  result=v.analyze(self.s,plan['data'],ids)
  self.assertTrue(result['full_scope_reviewed']);self.assertEqual(result['rows'][0]['adjustment_lb'],0)
 def test_zero_movement_cannot_allocate_fake_opposing_quantities(self):
  self.data['count'].append(self.row('R2','B','10',self.t+timedelta(minutes=45)));self.line(0,self.t+timedelta(minutes=20));self.answer(allocation={'R1':'10','R2':'-10'});self.assertEqual(self.target()['status'],'HELD')
 def test_duplicate_tag_cannot_link_unrelated_move(self):
  self.move(True,True);self.data['count'][0]['duplicate']='x'
  a=self.analyze();fp=next(iter(a['review_template']['duplicate_reviews']))
  self.data['review']['duplicate_reviews']={fp:dict(owner='TEST',notes='TEST',action='linked_move',move_id='UNRELATED')};self.assertEqual(self.target()['status'],'HELD')
 def test_invalid_lot_still_appears_in_area_hold(self):
  self.data['count'][0]['lot_code']='UNKNOWN';a=self.analyze();self.assertTrue(a['area_reconciliation'][0]['holds']);self.assertIn('R1',a['area_reconciliation'][0]['row_ids'])
 def test_sunshine_requires_office_ownership_evidence(self):
  self.s['products'][0]['id']=145;self.s['products'][0]['type']='finished';self.s['products'][0]['case_size_lb']=f.D('7.5');self.s['catalog']=deepcopy(self.s['products']);self.s['lots'][0]['product_id']=145
  for x in self.s['lines']:x['product_id']=145
  self.data['count'][0]['product_id']='145';self.data['coverage'][0]['product_id']='145'
  a=self.analyze();self.assertIn('Sunshine custody',str(a['holds']))
  fp=next(iter(a['review_template']['ownership_reviews']));self.data['review']['ownership_reviews']={fp:dict(owner='TEST',all_counted_stock_cns_owned=True,evidence_reference='TEST office documents')};self.assertEqual(self.target()['status'],'READY')
 def test_late_entry_between_area_sessions_is_still_late(self):
  self.data['count']=[self.row(qty='40'),self.row('R2','B','40',self.t+timedelta(minutes=45))]
  self.confirm_multi_area()
  self.line(-10,self.t+timedelta(minutes=20),entered=self.t+timedelta(minutes=36));self.answer(late=False,allocation={'R1':'-10'});self.assertEqual(self.target()['status'],'HELD')
  self.answer(late=True,allocation={'R1':'-10'});self.assertEqual(self.target()['status'],'READY')
 def test_apply_uses_different_lot_cutoffs(self):
  self.s['lots'].append(dict(id=2,product_id=1,lot_code='SECOND',status='active',merged_into_lot_id=None))
  x=self.line(50,self.t-timedelta(days=1));x['lot_id']=2
  r=self.row('R2','B','40',self.t+timedelta(minutes=45));r['lot_code']='SECOND';self.data['count'].append(r)
  b=FakeBackend(self.s);self.assertEqual(self.run_apply(b),0)
  payloads=[body for route,body in b.calls if body and body['mode']=='commit'];self.assertEqual({body['occurred_at'] for body in payloads},{self.t.isoformat(),(self.t+timedelta(minutes=45)).isoformat()})
 def test_v3_personal_actor_required(self):
  b=FakeBackend(self.s);b.actor_name='legacy-shared-key'
  with self.assertRaisesRegex(RuntimeError,'named actor'):self.run_apply(b)
  self.assertEqual(b.calls,[])
 def test_v3_corrected_reset_does_not_reconcile(self):
  b=FakeBackend(self.s);self.assertEqual(self.run_apply(b),0)
  plan=app.load_plan(self.preview,self.approval,self.out/'count.csv');self.s['lines'][-1]['line_correction_at']=self.s['snapshot_at']
  with self.assertRaisesRegex(RuntimeError,'corrected'):app.preflight(plan,self.s,app.Journal(app.JOURNAL_ROOT/plan['sha']/'journal.jsonl',plan))
 def packaging(self,pid=76,kind='packaging'):
  p=dict(id=pid,odoo_code='PKG-'+str(pid),name='TEST bags',type=kind,active=True,is_service=False,uom='unit')
  self.s['products'].append(p);self.s['catalog']=deepcopy(self.s['products'])
  self.s['lots'].append(dict(id=pid,product_id=pid,lot_code='PKG-LOT',status='active',merged_into_lot_id=None))
  x=dict(self.s['lines'][0],line_id=pid,transaction_id=pid,product_id=pid,lot_id=pid,quantity_lb=f.D(800))
  self.s['lines'].append(x);self.s['transaction_line_counts'][str(pid)]=1
  r=self.row('P'+str(pid),qty='120');r.update(product_id=str(pid),SKU=p['odoo_code'],name=p['name'],lot_code='PKG-LOT',unit='unit')
  self.data['count'].append(r);return r
 def resign(self):
  old=app.digest(self.preview.read_bytes());self.preview.write_text(json.dumps(self.a,default=str));self.approval.write_text(self.approval.read_text().replace(old,app.digest(self.preview.read_bytes())))
 def test_packaging_positive_ledger_never_reset_or_zeroed(self):
  self.packaging();self.data['coverage'].append(dict(product_id='76',all_locations_searched='yes'))
  a=self.analyze();self.assertEqual(set(a['products']),{1});self.assertEqual(set(a['baseline']),{'1'});self.assertEqual([r['product_id'] for r in a['rows']],[1]);self.assertTrue(a['full_scope_reviewed'])
  self.assertEqual(a['packaging_information_only'][0]['quantity'],'120');self.assertEqual(self.s['lines'][-1]['quantity_lb'],800)
 def test_packaging_unknown_lots_duplicates_and_moves_are_information_only(self):
  r=self.packaging();r.update(lot_code='UNLISTED',counted_time='',notes='possible duplicate',tag_id='SAME')
  self.data['count'].append(dict(r,row_id='P2'));self.data['moves']=[dict(product_id='76',move_id='INCOMPLETE',quantity='10')]
  a=self.analyze();self.assertTrue(a['full_scope_reviewed']);self.assertEqual(a['unidentified_follow_up'],[]);self.assertEqual(a['holds'],{});self.assertEqual(a['moved_during_count'],[]);self.assertEqual(len(a['packaging_moves_information_only']),1);self.assertEqual(len(a['packaging_information_only']),2)
 def test_packaging_blank_is_not_counted_and_explicit_zero_is_only_information(self):
  r=self.packaging();self.data['count'].remove(r);a=self.analyze();self.assertEqual(a['packaging_information_only'][0]['status'],'NOT COUNTED');self.assertEqual(a['packaging_information_only'][0]['quantity'],'');self.assertTrue(a['full_scope_reviewed'])
  r['quantity']='0';self.data['count'].append(r);a=self.analyze();self.assertEqual(a['packaging_information_only'][0]['quantity'],'0');self.assertEqual(len(a['rows']),1)
 def test_billing_ids_excluded_even_if_catalog_type_or_service_flag_changes(self):
  for pid in (102,176):self.packaging(pid,kind='ingredient')
  a=self.analyze();self.assertEqual(set(a['products']),{1});self.assertEqual({r['category'] for r in a['packaging_information_only']},{'billing item'});self.assertTrue(a['full_scope_reviewed'])
 def test_future_packaging_label_is_excluded_by_type(self):
  self.packaging(9001);a=self.analyze();self.assertIn(9001,a['information_products']);self.assertNotIn(9001,a['products']);self.assertEqual(len(a['rows']),1)
 def test_uncatalogued_labels_explicitly_marked_information_only(self):
  r=self.row('LABELS');r.update(product_id='',SKU='',name='Labels',row_kind='PACKAGING_INFO',lot_code='',quantity='3',unit='rolls');self.data['count'].append(r)
  a=self.analyze();self.assertTrue(a['full_scope_reviewed']);self.assertEqual(a['packaging_information_only'][0]['quantity'],'3');self.assertEqual(a['packaging_information_only'][0]['unit'],'rolls')
 def test_known_food_cannot_bypass_reset_by_using_packaging_info(self):
  self.data['count'][0]['row_kind']='PACKAGING_INFO';a=self.analyze();self.assertFalse(a['full_scope_reviewed']);self.assertIn('cannot be marked',str(a['holds']));self.assertEqual(a['packaging_information_only'],[])
 def test_unclassified_unknown_item_still_holds(self):
  r=self.row('UNKNOWN');r.update(product_id='',name='unknown item');self.data['count'].append(r);a=self.analyze();self.assertFalse(a['full_scope_reviewed']);self.assertTrue(a['sheet_follow_up']['UNKNOWN']);self.assertEqual(a['general'],[])
 def test_only_packaging_observations_never_zero_food(self):
  r=self.packaging();self.data['count']=[r];self.data['coverage']=[];a=self.analyze();self.assertEqual(a['rows'],[]);self.assertFalse(a['full_scope_reviewed']);self.assertIn('1',a['holds']);self.assertNotIn('76',a['holds'])
 def test_finished_food_cases_stay_in_reset(self):
  self.s['products'][0].update(type='finished',name='TEST food cases',case_size_lb=f.D(10));self.data['count'][0].update(name='TEST food cases',unit='cases',quantity='9');self.data['coverage'][0]['name']='TEST food cases';self.assertEqual(self.target()['adjustment_lb'],-10)
 def test_report_verify_excludes_packaging_from_signoff(self):
  self.packaging();self.data['count'][0]['quantity']='100';a=self.analyze();result=v.report(a,self.out/'reports',verify=True);self.assertEqual(result,0)
  text=(self.out/'reports/reset-verification-v3.md').read_text();self.assertIn('FULL RESET SCOPE RECONCILED',text);self.assertIn('Reset scope: 1 products',text)
  rows=v.read_csv(self.out/'reports/packaging-information-only.csv');self.assertEqual(rows[0]['quantity'],'120');self.assertFalse(any('adjustment' in k for k in rows[0]));self.assertTrue((self.out/'reports/packaging-information-only.md').exists())
 def test_apply_sends_no_packaging_requests(self):
  self.packaging();b=FakeBackend(self.s);self.assertEqual(self.run_apply(b),0);self.assertEqual(b.commits,1);self.assertTrue(all(body['product_name']=='11001' for route,body in b.calls if body));self.assertEqual(self.s['lines'][1]['quantity_lb'],800)
 def test_old_signed_scope_policy_refused(self):
  self.persist();del self.a['reset_scope_policy'];self.resign()
  with self.assertRaisesRegex(RuntimeError,'scope decision changed'):app.load_plan(self.preview,self.approval,self.out/'count.csv')
 def test_signed_packaging_row_refused_even_zero_adjustment(self):
  self.data['count'][0]['quantity']='100';self.persist();self.a['rows'][0]['product_type']='packaging';self.resign()
  with self.assertRaisesRegex(RuntimeError,'Packaging/billing item'):app.load_plan(self.preview,self.approval,self.out/'count.csv')
 def test_signed_billing_row_refused_even_if_disguised_as_ingredient(self):
  self.persist();self.a['rows'][0]['product_id']=102;self.resign()
  with self.assertRaisesRegex(RuntimeError,'Packaging/billing item'):app.load_plan(self.preview,self.approval,self.out/'count.csv')
 def test_live_preflight_refuses_new_packaging_type_before_api(self):
  plan=self.persist();self.s['products'][0]['type']='packaging';b=FakeBackend(self.s)
  with self.assertRaisesRegex(RuntimeError,'Packaging/billing item'):app.preflight(plan,self.s,app.Journal(self.out/'journal',plan))
  self.assertEqual(b.calls,[])
 def test_live_preflight_refuses_billing_identity_even_with_ingredient_type(self):
  plan=self.persist();self.packaging(176,kind='ingredient');plan['ids']={176}
  with self.assertRaisesRegex(RuntimeError,'Packaging/billing item'):app.preflight(plan,self.s,app.Journal(self.out/'journal',plan))
 def test_readonly_journal_refuses_packaging_rows(self):
  plan=self.persist();self.a['rows'][0]['product_type']='packaging';self.resign();(self.out/'journal').write_text('')
  with self.assertRaisesRegex(ValueError,'Packaging/billing item'):v.journal_openings(self.preview,self.approval,self.out/'journal',self.s,plan['data'])
 def test_readonly_journal_refuses_current_catalog_packaging(self):
  plan=self.persist();self.s['products'][0]['type']='packaging';(self.out/'journal').write_text('')
  with self.assertRaisesRegex(ValueError,'excluded by current scope'):v.journal_openings(self.preview,self.approval,self.out/'journal',self.s,plan['data'])
 def test_readonly_journal_rejects_obsolete_scope_policy(self):
  plan=self.persist();del self.a['reset_scope_policy'];self.resign();(self.out/'journal').write_text('')
  with self.assertRaisesRegex(ValueError,'scope decision changed'):v.journal_openings(self.preview,self.approval,self.out/'journal',self.s,plan['data'])
 def test_saved_scope_178_reset_32_information_floor_rows_unchanged(self):
  s=json.loads((v.V/'raw/01-scope.json').read_text(),parse_float=f.D)
  data=v.load_inputs(v.V/'count-sheet-v3.csv',v.V/'count-coverage-v3.csv',v.V/'moved-during-count-v3.csv');a=v.analyze(s,data)
  self.assertEqual(len(a['products']),178);self.assertEqual(len(a['information_products']),32);self.assertEqual(len(a['holds']),178);self.assertEqual(a['rows'],[]);self.assertEqual(len(data['count']),850);self.assertEqual(len({r['product_id'] for r in data['count'] if r['product_id']}),210)
  self.assertTrue(all(r['quantity']=='' and r['status']=='NOT COUNTED' for r in a['packaging_information_only']))
 def test_fixed_sql_readonly_and_only_apply_has_api_client(self):
  self.assertTrue(v.SQL.startswith('BEGIN;\nSET TRANSACTION READ ONLY;'));self.assertTrue(v.SQL.rstrip().endswith('ROLLBACK;'))
  for name in ('fresh_start_v3.py','reset_preview.py','verify_reset.py','build_v3_materials.py'):
   source=(f.OUT/name).read_text();tree=ast.parse(source)
   self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom)) and ('urllib.request' in ast.get_source_segment(source,n) or 'requests' in ast.get_source_segment(source,n)) for n in ast.walk(tree)))
 def test_H1_multi_area_lot_has_one_hold_listing_all_rows(self):
  self.data['count'] += [self.row('R2','B','10'),self.row('R3','C','20')]
  a=self.analyze();message='owner confirms no unlogged moves between rows R1/R2/R3'
  self.assertEqual(a['lot_holds'],{'1':[message]});self.assertEqual(a['rows'][0]['holds'],[message]);self.assertFalse(a['full_scope_reviewed'])
  self.confirm_multi_area();self.assertEqual(self.target()['status'],'READY')
  self.data['count'][1]['quantity']='11';self.assertEqual(self.target()['holds'],[message])
 def test_H1_hold_is_per_lot_and_same_area_needs_no_hold(self):
  self.data['count'].append(self.row('R2','A','10'));self.assertEqual(self.analyze()['lot_holds'],{})
  self.data['count'][1].update(area='B',location='B shelf')
  self.s['lots'].append(dict(id=2,product_id=1,lot_code='SECOND',status='active',merged_into_lot_id=None))
  self.data['count'].append(dict(self.row('R3','A','0'),lot_code='SECOND'))
  a=self.analyze();self.assertEqual(a['rows'][0]['holds'],['owner confirms no unlogged moves between rows R1/R2']);self.assertEqual(a['rows'][1]['holds'],[])
 def test_H1_reviewed_logged_move_still_needs_no_unlogged_moves_confirmation(self):
  self.move(True,True);a=self.analyze()
  self.assertEqual(a['moved_during_count'][0]['status'],'REVIEWED');self.assertEqual(a['rows'][0]['holds'],['owner confirms no unlogged moves between rows R1/R2'])
 def test_M1_before_start_occurred_after_start_entered_is_late(self):
  self.line(-1,self.t-timedelta(minutes=30),entered=self.t-timedelta(minutes=5))
  entry=next(iter(self.analyze()['review_template']['movement_reviews'].values()))
  self.assertIn('late/back-dated entry',entry['reasons']);self.answer(late=False);self.assertEqual(self.target()['status'],'HELD');self.answer();self.assertEqual(self.target()['status'],'READY')
 def test_M1_entered_before_count_start_is_not_late(self):
  self.line(-1,self.t-timedelta(minutes=29),entered=self.t-timedelta(minutes=11))
  entry=next(iter(self.analyze()['review_template']['movement_reviews'].values()))
  self.assertNotIn('late/back-dated entry',entry['reasons'])
 def test_M1_exactly_fifteen_minutes_is_not_late(self):
  self.line(-1,self.t-timedelta(minutes=10),entered=self.t+timedelta(minutes=5))
  entry=next(iter(self.analyze()['review_template']['movement_reviews'].values()))
  self.assertNotIn('late/back-dated entry',entry['reasons'])
 def test_M1_entered_exactly_at_start_is_late(self):
  self.line(-1,self.t-timedelta(minutes=31),entered=self.t-timedelta(minutes=10))
  entry=next(iter(self.analyze()['review_template']['movement_reviews'].values()))
  self.assertIn('late/back-dated entry',entry['reasons'])
 def test_M2_prior_openings_match_old_hashes_once_per_entry(self):
  for oldhash in ('oldhash12345','differenthash'):
   x=self.line(1,self.t-timedelta(days=1));x.update(type='adjust',adjust_reason=f'OPENING BALANCE 2026-01-01 – physical count | v3 {oldhash} L1 | Estimated=no')
  a=self.analyze();self.assertEqual(a['lot_holds']['1'],['prior opening without journal proof']*2)
  self.assertEqual(a['rows'][0]['status'],'HELD')
  proved=v.analyze(self.s,self.data,[2]);self.assertEqual(proved['lot_holds']['1'],['prior opening without journal proof'])
 def test_M2_prior_opening_prefix_and_lot_boundary_are_exact(self):
  for reason in ('OPENING BALANCE 2026-01-01 | v3 old L10 | Estimated=no','notes OPENING BALANCE 2026-01-01 | v3 old L1'):
   self.line(1,self.t-timedelta(days=1)).update(type='adjust',adjust_reason=reason)
  self.assertEqual(self.analyze()['lot_holds'],{})
 def test_M2_movement_review_includes_type_reason_and_operator(self):
  x=self.line(1,self.t);x.update(type='adjust',adjust_reason='Owner recount',operator_id='TEST PERSON')
  entry=next(iter(self.analyze()['review_template']['movement_reviews'].values()))
  for key in ('type','adjust_reason','operator_id'):self.assertEqual(entry[key],x[key])
 def test_M2_prior_opening_without_timestamp_still_requires_journal_proof(self):
  x=self.line(1,self.t);x.update(type='adjust',adjust_reason='OPENING BALANCE old date | v3 oldhash L1',occurred_at=None)
  a=self.analyze();self.assertIn('prior opening without journal proof',a['rows'][0]['holds']);self.assertIn('Missing ledger timestamp',a['rows'][0]['holds'])
 def test_M3_blank_product_follow_up_is_local_to_its_sheet(self):
  self.data['count'].append(dict(self.row('UNKNOWN'),product_id='',name='unknown item'))
  a=self.analyze();self.assertEqual(a['general'],[]);self.assertNotIn('None',a['holds']);self.assertEqual(list(a['sheet_follow_up']),['UNKNOWN'])
  areas={r['sheet_id']:r for r in a['area_reconciliation']}
  self.assertEqual(areas['R1']['follow_up'],[]);self.assertEqual(areas['R1']['holds'],[])
  self.assertEqual(areas['UNKNOWN']['follow_up'][0]['row_id'],'UNKNOWN');self.assertFalse(a['full_scope_reviewed'])
  self.assertEqual(v.report(a,self.out/'follow-up',verify=True),2)
  rows=v.read_csv(self.out/'follow-up/sheet-follow-up.csv');self.assertEqual(rows[0]['sheet_id'],'UNKNOWN')
 def assert_invalid_move_follow_up(self,pid):
  self.data['moves']=[{},dict(move_id='UNKNOWN-MOVE',product_id=pid,quantity='10')]
  self.data['count'][0]['quantity']='100'
  a=self.analyze();message='moves CSV row 3: move-log row has no valid product_id'
  self.assertEqual(a['general'],[]);self.assertEqual(a['holds'],{});self.assertEqual(a['lot_holds'],{});self.assertEqual(a['sheet_follow_up'],{})
  self.assertTrue(a['full_scope_reviewed']);self.assertEqual(a['rows'][0]['status'],'READY')
  self.assertEqual(a['moved_during_count'],[]);self.assertEqual(a['review_template']['move_reviews'],{})
  self.assertEqual(len(a['move_follow_up']),1);entry=a['move_follow_up'][0]
  self.assertEqual(entry['move_id'],'UNKNOWN-MOVE');self.assertEqual(entry['product_id'],pid);self.assertEqual(entry['detail'],message)
  self.assertEqual(v.report(a,self.out/'move-follow-up',verify=True),0)
  self.assertEqual(v.read_csv(self.out/'move-follow-up/move-follow-up.csv')[0]['detail'],message)
  self.assertIn(message,(self.out/'move-follow-up/reset-verification-v3.md').read_text())
 def test_M3_blank_move_product_has_move_level_follow_up(self):
  self.assert_invalid_move_follow_up('')
 def test_M3_unknown_move_product_has_move_level_follow_up(self):
  self.assert_invalid_move_follow_up('99999')
 def test_M3_bad_row_on_another_sheet_does_not_leak_into_sheet_follow_up(self):
  self.data['count'].append(dict(self.row('R2','B'),lot_code='UNKNOWN'))
  a=self.analyze();areas={r['sheet_id']:r for r in a['area_reconciliation']}
  self.assertEqual(a['general'],[]);self.assertEqual(areas['R1']['follow_up'],[]);self.assertEqual(areas['R1']['holds'],[])
  self.assertEqual(areas['R2']['follow_up'][0]['row_id'],'R2');self.assertTrue(areas['R2']['holds'])
 def test_M3_sheet_with_only_invalid_rows_still_has_follow_up(self):
  self.data['count']=[dict(self.row(),product_id='',start_time='')]
  a=self.analyze();self.assertEqual(a['area_reconciliation'][0]['sheet_id'],'R1');self.assertEqual(a['sheet_follow_up']['R1'][0]['row_id'],'R1');self.assertEqual(a['general'],[])
 def test_M3_blank_product_holds_only_lots_on_its_sheet(self):
  self.data['count'].append(dict(self.row('UNKNOWN'),sheet_id='R1',product_id='',name='unknown item'))
  self.s['lots'].append(dict(id=2,product_id=1,lot_code='SECOND',status='active',merged_into_lot_id=None))
  self.data['count'].append(dict(self.row('R2','B','0'),lot_code='SECOND'))
  a=self.analyze();self.assertEqual(a['rows'][0]['status'],'HELD');self.assertEqual(a['rows'][1]['status'],'READY');self.assertEqual(a['general'],[])
 def test_M3_covered_product_without_count_rows_requires_review(self):
  self.data['count']=[]
  a=self.analyze();entry=next(iter(a['review_template']['coverage_reviews'].values()))
  self.assertEqual(entry['product_id'],1);self.assertEqual(entry['hold'],'coverage certified but no count rows');self.assertIn(entry['hold'],a['holds']['1']);self.assertFalse(a['full_scope_reviewed'])
  self.data['review']['coverage_reviews']={fp:dict(owner='TEST',confirmed=True) for fp in a['review_template']['coverage_reviews']}
  self.assertTrue(self.analyze()['full_scope_reviewed'])
 def test_M3_coverage_review_also_includes_products_without_lots(self):
  self.data['count']=[];self.s['lots']=[];self.s['lines']=[]
  a=self.analyze();self.assertEqual(len(a['review_template']['coverage_reviews']),1);self.assertIn('coverage certified but no count rows',a['holds']['1'])
 def test_L1_excluding_every_duplicate_keeps_one_row_and_holds(self):
  self.data['count']=[dict(self.row(),duplicate='x'),dict(self.row('R2','A','90'),duplicate='x')]
  fp=next(iter(self.analyze()['review_template']['duplicate_reviews']))
  self.data['review']['duplicate_reviews']={fp:dict(owner='TEST',notes='same stock',action='exclude',exclude_rows=['R1','R2'])}
  a=self.analyze();self.assertEqual(a['excluded_duplicate_rows'],['R2']);self.assertEqual(a['rows'][0]['row_ids'],['R1']);self.assertEqual(a['rows'][0]['physical_count_native'],90)
  self.assertEqual(a['rows'][0]['status'],'HELD');self.assertIn('kept R1 so at least one count row remains',str(a['holds']))
 def test_L1_single_duplicate_cannot_be_excluded(self):
  self.data['count'][0]['duplicate']='x';fp=next(iter(self.analyze()['review_template']['duplicate_reviews']))
  self.data['review']['duplicate_reviews']={fp:dict(owner='TEST',notes='duplicate',action='exclude',exclude_rows=['R1'])}
  a=self.analyze();self.assertEqual(a['excluded_duplicate_rows'],[]);self.assertEqual(a['rows'][0]['physical_count_native'],90);self.assertEqual(a['rows'][0]['status'],'HELD')
 def test_L2_notes_and_tags_do_not_mark_unticked_rows_as_duplicates(self):
  self.data['count']=[dict(self.row(),notes='possible duplicate',tag_id='SAME'),dict(self.row('R2','A'),notes='DUPLICATE',tag_id='SAME')]
  a=self.analyze();self.assertEqual(a['review_template']['duplicate_reviews'],{});self.assertEqual(a['excluded_duplicate_rows'],[]);self.assertEqual(a['rows'][0]['status'],'READY')
 def test_L2_only_ticked_rows_can_be_excluded(self):
  self.data['count'].append(dict(self.row('R2','A'),duplicate='☑'))
  fp=next(iter(self.analyze()['review_template']['duplicate_reviews']))
  answer=dict(owner='TEST',notes='same stock',action='exclude',exclude_rows=['R1'])
  self.data['review']['duplicate_reviews']={fp:answer};self.assertEqual(self.analyze()['excluded_duplicate_rows'],[])
  answer['exclude_rows']=['R2'];self.assertEqual(self.analyze()['excluded_duplicate_rows'],['R2']);self.assertEqual(self.target()['status'],'READY')
 def test_L2_template_has_duplicate_tick_column_and_never_overwrites(self):
  path=self.out/'new-count.csv'
  with patch('sys.argv',['fresh_start_v3.py','--write-count-template',str(path)]),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(v.main(),0)
  with path.open(newline='') as stream:self.assertEqual(next(csv.reader(stream)),v.COUNT_COLUMNS)
  rows=v.read_csv(path,v.COUNT_COLUMNS);issued=v.read_csv(v.V/'count-sheet-v3.csv')
  self.assertEqual(len(rows),len(issued));self.assertEqual([r['row_id'] for r in rows],[r['row_id'] for r in issued]);self.assertTrue(all(r['duplicate']==r['quantity']=='' for r in rows))
  self.assertIn('duplicate',v.COUNT_COLUMNS)
  with self.assertRaises(FileExistsError):v.write_count_template(path)
  self.assertIn('duplicate tick column',(f.OUT/'tool-guide.md').read_text())
 def test_L2_duplicate_phrase_in_tick_column_is_rejected(self):
  self.data['count'][0]['duplicate']='possible duplicate';a=self.analyze();self.assertIn('Duplicate must be a tick',str(a['sheet_follow_up']));self.assertEqual(a['excluded_duplicate_rows'],[])
 def test_L3_ten_minute_sheet_has_thirty_minute_margin(self):
  self.assertEqual(v.post_sheet_margin(self.t,self.t+timedelta(minutes=10)),timedelta(minutes=30))
 def test_L3_three_hour_sheet_has_ninety_minute_margin(self):
  self.assertEqual(v.post_sheet_margin(self.t,self.t+timedelta(hours=3)),timedelta(minutes=90))
 def test_L3_existing_larger_margin_is_preserved(self):
  self.assertEqual(v.post_sheet_margin(self.t,self.t+timedelta(hours=3),timedelta(hours=2)),timedelta(hours=2))
 def test_L3_long_sheet_post_sheet_margin_is_used_for_late_review(self):
  self.data['count'][0]['start_time']=(self.t-timedelta(hours=3)).isoformat();self.data['count'][0]['end_time']=self.t.isoformat()
  self.line(-1,self.t+timedelta(minutes=80))
  entry=next(iter(self.analyze()['review_template']['movement_reviews'].values()));self.assertIn('late/back-dated entry',entry['reasons'])
 def test_L4_consumed_destination_stays_held_across_rerun_until_removed(self):
  self.move(True,False);self.data['count']=self.data['count'][:1]
  fp=v.fingerprint(self.data['moves'][0]);self.data['review']['move_reviews'][fp].update(destination_row='CONSUMED',notes='consumed before count; owner reviewed')
  first=self.analyze();second=v.analyze(deepcopy(self.s),deepcopy(self.data))
  for a in (first,second):
   self.assertEqual(a['moved_during_count'][0]['status'],'HELD');self.assertIn('Move needs distinct source/destination count rows',str(a['holds']))
  self.data['moves']=[];self.assertEqual(self.target()['status'],'READY')
 def test_L4_consumed_catalog_lot_destination_is_held(self):
  self.move(True,False);self.s['lots'][0]['status']='consumed'
  for a in (self.analyze(),v.analyze(deepcopy(self.s),deepcopy(self.data))):
   self.assertIn('Move destination is a consumed lot; remove this row from the move log',a['moved_during_count'][0]['detail'])
  self.data['moves']=[];self.assertEqual(self.analyze()['moved_during_count'],[])
 def test_L5_stage_two_guide_report_and_console_limit_move_detection(self):
  for path in (f.OUT/'tool-guide.md',v.V/'STAGE-2.md'):
   self.assertIn(v.MOVE_DETECTION_NOTE,path.read_text())
  v.report(self.analyze(),self.out/'wording');self.assertIn(v.MOVE_DETECTION_NOTE,(self.out/'wording/reset-preview-v3.md').read_text())
  output=io.StringIO()
  with patch('sys.argv',['fresh_start_v3.py','--write-count-template',str(self.out/'wording.csv')]),contextlib.redirect_stdout(output):v.main()
  self.assertIn(v.MOVE_DETECTION_NOTE,output.getvalue())
if __name__=='__main__':unittest.main()
