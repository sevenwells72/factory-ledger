"""Live-count v3. Read-only snapshots, pure reconciliation and local reports only.
No API client, apply path, arbitrary SQL, lot creation or ingredient-total mode.
Reset quantities are ledger-native units; packaging counts are information only.
"""
import argparse,csv,hashlib,json,re,os,subprocess
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from decimal import Decimal as D
from pathlib import Path
from urllib.parse import urlsplit
import fresh_start_common as f
V=f.OUT/'v3'
COUNT_COLUMNS=['sheet_id','area','count_date','counter','start_time','end_time','row_id','product_id','SKU','name','row_kind','lot_code','quantity','unit','partial_cases','location','bin_id','counted_time','estimated','tag_id','physical_marker','duplicate','notes']
MOVE_DETECTION_NOTE='Moves are caught by the tool only when logged, tagged, or noted; for anything else, the H1 multi-area hold is the control.'
SQL=(V/'queries/01-scope.sql').read_text()

def post_sheet_margin(start,end,existing_margin=timedelta(minutes=30)):
 """Late-entry review after sheet end uses the largest of the existing margin,
 30 minutes, and 50% of the sheet's elapsed duration.
 """
 return max(existing_margin,timedelta(minutes=30),(end-start)/2)

def write_count_template(path):
 """Write a blank v3 CSV template with an explicit duplicate tick column."""
 path=f.local_path(path)
 rows=read_csv(V/'count-sheet-v3.csv')
 check(not any(used_count_row(r) for r in rows),'Issued template must contain only blank count rows')
 with path.open('x',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=COUNT_COLUMNS);writer.writeheader();writer.writerows(rows)

def sha(value):return hashlib.sha256(value).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),default=str)
def fingerprint(value):return sha(canonical(value).encode())
def check(ok,message):
 if not ok:raise ValueError(message)
def native(p):
 u=p.get('uom') or 'lb'
 return 'lb' if p['type'] in ('batch','finished') or re.search(r'(?<![a-z])(?:lbs?|pounds?)(?![a-z])',u,re.I) else u

def canonical_lot_code(code):
 """= main.normalize_lot_code_input: what FL stores for a caller-supplied new code."""
 return re.sub(r'\s+',' ',code.strip()).upper()

OPENING=r'^OPENING BALANCE .+? \| v3 \S+ '
def opening_texts(x,lot):
 """Where an executor's reset text lives: the legacy direct-/adjust reason, the A1
 adjust-ticket note (after 'Adjustment: N lb — '), or a found ticket's lot note."""
 texts=[x.get('adjust_reason') or '']
 notes=x.get('notes') or ''
 if ' — ' in notes:texts.append(notes.split(' — ',1)[1])
 if lot and lot.get('entry_source')=='found_inventory':texts.append(lot.get('entry_source_notes') or '')
 return texts

def looks_like_opening(x,lot,lid):
 marks=['L'+str(lid)]+(['NEW '+re.escape(lot['lot_code'])] if lot and lot.get('lot_code') else [])
 pattern=OPENING+'(?:'+'|'.join(marks)+r')(?=\s*\||$)'
 return any(re.match(pattern,t) for t in opening_texts(x,lot))

A3B_UNIT_TO_LB={'lb':D(1),'lbs':D(1),'pound':D(1),'pounds':D(1),'kg':D('2.20462262185'),'kgs':D('2.20462262185'),'kilogram':D('2.20462262185'),'kilograms':D('2.20462262185'),'g':D(1)/D('453.59237'),'gram':D(1)/D('453.59237'),'grams':D(1)/D('453.59237'),'oz':D(1)/D(16),'ounce':D(1)/D(16),'ounces':D(1)/D(16)}
def a3b_lb_equivalent(p,q):
 """Exactly exceptions_enforcement.correction_review for an adjust, so the signed plan
 predicts the hold FL will apply. Since PR #109 an adjust delta on a WEIGHTED case
 product is already pounds (40 lb is 40 lb, not 40 cases × 15 lb); only an unweighted
 case/bag product has no pound equivalent (no 500 lb rule)."""
 unit=(p.get('uom') or 'lb').strip() or 'lb'
 unit='lb' if re.search(r'(?<![a-z])(?:lbs?|pounds?)(?![a-z])',unit,re.I) else unit
 unit=unit.strip().lower().rstrip('.')
 if unit in A3B_UNIT_TO_LB:return abs(q)*A3B_UNIT_TO_LB[unit]
 if unit in ('case','cases','cs'):
  w=p.get('case_size_lb') or p.get('default_case_weight_lb');return abs(q) if w else None
 if unit in ('bag','bags'):
  w=p.get('pack_size_lbs');return abs(q)*f.number(w) if w else None
 return None

def owner_approvals(row,p=None):
 """Which FL owner holds this post will meet (A3b > 500 lb, A12 extra-kosher found)."""
 needs=[]
 if row['action']=='found':needs.append('KOSHER_ATTESTATION')
 elif row['rounded_adjustment_lb']:
  lb=a3b_lb_equivalent(p or dict(uom=row['ledger_unit']),row['rounded_adjustment_lb'])
  if lb is not None and lb>f.LARGE_CORRECTION_LB+f.TOLERANCE:needs.append('LARGE_CORRECTION')
 return needs

def snapshot(target='production'):
 # Fixed statement, explicitly transaction-read-only, rollback, session-mode port.
 check(SQL.startswith('BEGIN;\nSET TRANSACTION READ ONLY;'),'Read-only query wrapper missing')
 check(SQL.rstrip().endswith('ROLLBACK;'),'Read-only rollback missing')
 check(not re.search(r'\b(INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|COMMIT|COPY|CALL|DO)\b',SQL,re.I),'Mutation rejected')
 url=f.target_database_url(target)
 check(urlsplit(url).port==5432,'Read-only connection must use port 5432')
 env=dict(os.environ,DATABASE_URL=url,PGCONNECT_TIMEOUT='15');env.pop('PGOPTIONS',None)
 p=subprocess.run([str(f.WRAPPER),'-X','-A','-t','-v','ON_ERROR_STOP=1'],input=SQL,text=True,capture_output=True,env=env,cwd=f.ROOT,timeout=120)
 check(p.returncode==0,'Read-only query failed; connection details suppressed')
 check('ROLLBACK' in p.stdout.splitlines(),'Rollback acknowledgement missing')
 try:s=json.JSONDecoder(parse_float=D).raw_decode(p.stdout[p.stdout.index('{'):])[0]
 except (ValueError,TypeError):raise RuntimeError('Invalid read-only snapshot; details suppressed') from None
 check(s.get('read_only')=='on','Database did not confirm read-only transaction')
 s['target']=target
 return s

def read_csv(path,headers=None):
 path=f.local_path(path)
 with path.open(newline='',encoding='utf-8-sig') as stream:
  reader=csv.DictReader(stream)
  if headers:check(reader.fieldnames==headers,'Unexpected CSV columns: '+path.name)
  rows=list(reader)
 check(all(None not in r and None not in r.values() for r in rows),'Malformed CSV: '+path.name)
 return rows

def load_inputs(count,coverage,moves,review=None):
 paths={k:f.local_path(p) for k,p in dict(count=count,coverage=coverage,moves=moves,review=review).items() if p}
 # Previously issued sheets have no tick column: preserve their rows, with no
 # duplicate ticks inferred from notes or tags. New templates always include it.
 with paths['count'].open(newline='',encoding='utf-8-sig') as stream:headers=next(csv.reader(stream),[])
 check(headers in (COUNT_COLUMNS,[k for k in COUNT_COLUMNS if k!='duplicate']),'Unexpected CSV columns: '+paths['count'].name)
 counts=read_csv(paths['count'])
 for r in counts:r.setdefault('duplicate','')
 data=dict(count=counts,coverage=read_csv(paths['coverage']),moves=read_csv(paths['moves']),review=json.loads(paths['review'].read_text(),parse_float=D) if review else {})
 data['sources']={k:dict(path=str(p),sha256=sha(p.read_bytes())) for k,p in paths.items()}
 return data

def at(value,date):
 check(bool(value),'Clock time is required')
 if 'T' in value:return f.stamp(value)
 check(bool(re.fullmatch(r'\d{2}:\d{2}(?::\d{2})?',value)),'Use HH:MM or an offset ISO timestamp')
 d=datetime.fromisoformat(date+'T'+value)
 # Ambiguous/nonexistent DST clock times require explicit offset timestamps.
 a=d.replace(tzinfo=f.PLANT,fold=0);b=d.replace(tzinfo=f.PLANT,fold=1)
 check(a.utcoffset()==b.utcoffset(),'DST boundary: use explicit offset timestamp')
 return a

def convert(row,p):
 q=f.number(row['quantity']) if row['quantity'].strip() else D(0)
 check(q>=0,'Negative physical count is invalid')
 partial=row.get('partial_cases','').strip();u=row['unit'].strip()
 if p['type']=='finished' and u in ('case','cases'):
  weight,error=f.case_weight(p)
  if partial:
   match=re.fullmatch(r'(\d+)\s+of\s+(\d+)',partial)
   check(match and D(match[2])>0,'Partials must be X of Y with Y > 0')
   q+=D(match[1])/D(match[2])
  if q==0:return D(0)
  check(weight is not None,error)
  return q*weight
 check(not partial,'Partial cases only apply to finished case counts')
 check(u==native(p),'Unit must be '+native(p)+' for this item')
 if p['type']=='finished' and p['id'] in f.EXPECTED_ZERO_PRODUCTS and q:
  raise ValueError('Legacy package identity requires owner decision; no conversion bypass')
 return q

def baseline(s,pid,exclude=(),exclude_lots=()):
 # exclude/exclude_lots: the executor's own proven reset lines and the extra-kosher
 # lots its found tickets created, so its own posts never read as outside drift.
 xs=[x for x in s['lines'] if x['product_id']==pid and x['line_id'] not in set(exclude)]
 tx={x['transaction_id'] for x in xs};lineids={x['line_id'] for x in xs}
 cs=[c for c in s.get('corrections',[]) if (c['target_table']=='transactions' and c['target_id'] in tx) or (c['target_table']=='transaction_lines' and c['target_id'] in lineids)]
 return fingerprint(dict(product=next((p for p in s['products'] if p['id']==pid),None),lots=sorted([l for l in s['lots'] if l['product_id']==pid and l['id'] not in set(exclude_lots)],key=lambda l:l['id']),lines=sorted(xs,key=lambda l:l['line_id']),corrections=sorted(cs,key=lambda c:str(c['id']))))

def information_reason(row,products):
 pid=int(row['product_id']) if row.get('product_id','').isdigit() else None
 if pid in products:return f.reset_exclusion_reason(products[pid])
 if not row.get('product_id','').strip() and row.get('row_kind')=='PACKAGING_INFO':
  return 'Uncatalogued packaging explicitly marked PACKAGING_INFO; information only.'
 return ''

def used_count_row(row):
 return any(row.get(k,'').strip() for k in ('lot_code','quantity','partial_cases','location','bin_id','counted_time','estimated','tag_id','physical_marker','duplicate','notes'))

def information_counts(data,products):
 """Keep raw observations, including unknown lots, without targets or zero inference."""
 records=[];observed=set()
 for row in data['count']:
  reason=information_reason(row,products)
  if not reason or not used_count_row(row):continue
  pid=int(row['product_id']) if row['product_id'].isdigit() else None
  observed.add(pid);p=products.get(pid)
  records.append(dict(row,category='billing item' if pid in f.BILLING_RESET_EXCLUDED_IDS else 'packaging',catalog_SKU=(p.get('odoo_code') or '') if p else '',catalog_name=p['name'] if p else '',status='REPORTED — information only' if row['quantity'].strip() else 'QUANTITY NOT ENTERED',exclusion_reason=reason))
 for pid,p in sorted(products.items()):
  reason=f.reset_exclusion_reason(p)
  if reason and pid not in observed:
   records.append(dict(product_id=pid,SKU=p.get('odoo_code') or '',name=p['name'],quantity='',unit=native(p),category='billing item' if pid in f.BILLING_RESET_EXCLUDED_IDS else 'packaging',status='NOT COUNTED',exclusion_reason=reason))
 return records

def analyze(s,data,verified_openings=(),areas=None):
 """Physical timing uses occurred_at; late-entry review also uses entered_at.
 Owner answers only affect the local count basis.
 Per-location observations roll to one lot target, with each movement applied once.
 """
 catalog={p['id']:p for p in s['products']}
 products={pid:p for pid,p in catalog.items() if not f.reset_exclusion_reason(p)}
 information_products={pid:p for pid,p in catalog.items() if f.reset_exclusion_reason(p)}
 information=information_counts(data,catalog)
 information_moves=[]
 lots={l['id']:l for l in s['lots'] if l['product_id'] in products};now=f.stamp(s['snapshot_at'])
 lines=s['lines'];review=data.get('review') or {};sources=data.get('sources',{})
 # A3b: open SHORTAGE exceptions. A reset post on such a lot names them so the ADJ
 # ticket closes them as 'counted' (owner decision 2026-10-13); FL refuses otherwise.
 shortages=[x for x in s.get('shortages',[]) if x.get('status') in ('open','escalated')]
 countsha=sources.get('count',{}).get('sha256',fingerprint(data['count']))
 holds=defaultdict(list);lot_holds=defaultdict(list);sheet_follow_up=defaultdict(list);row_holds=set();general=[];observations=[];unknown=[];templates={k:{} for k in ('movement_reviews','move_reviews','duplicate_reviews','correction_reviews','ownership_reviews','multi_area_reviews','coverage_reviews')}
 def hold(pid,msg):
  if msg not in holds[pid]:holds[pid].append(msg)
 seen=set();sheetmeta={};sheetrows=defaultdict(list)
 for r0 in data['count']:
  if information_reason(r0,catalog):continue
  r=dict(r0);rid=r['row_id'];pid=int(r['product_id']) if r['product_id'].isdigit() else None
  check(rid and rid not in seen,'Duplicate or missing row ID');seen.add(rid)
  # Blank issued rows are not zero counts, even if product/name/unit are prefilled.
  used=used_count_row(r)
  if not used:continue
  sheetrows[r['sheet_id']].append(r)
  p=products.get(pid)
  try:
   check(p is not None,'Item absent from FL catalog')
   check(r.get('row_kind')!='PACKAGING_INFO','A catalogued reset item cannot be marked PACKAGING_INFO')
   check(r['SKU']==(p.get('odoo_code') or '') and r['name']==p['name'],'Product/SKU/name mismatch')
   check(r['sheet_id'] and r['area'] and r['counter'] and r['location'],'Sheet, area, counter and location are required')
   start=at(r['start_time'],r['count_date']);end=at(r['end_time'],r['count_date'])
   if end<start and 'T' not in r['end_time']:end+=timedelta(days=1)
   cut=at(r['counted_time'],r['count_date']) if r['counted_time'] else end
   if cut<start and r['counted_time'] and 'T' not in r['counted_time']:cut+=timedelta(days=1)
   check(start<=cut<=end<=now,'Count time must lie inside the completed sheet, not in the future')
   meta=(r['area'],r['counter'],start.isoformat(),end.isoformat())
   check(r['sheet_id'] not in sheetmeta or sheetmeta[r['sheet_id']]==meta,'Inconsistent area/header times on one sheet')
   sheetmeta[r['sheet_id']]=meta
   check(r['quantity'].strip() or r['partial_cases'].strip(),'Quantity is missing')
   matches=[l for l in lots.values() if l['product_id']==pid and l['lot_code']==r['lot_code']]
   if not matches and r['lot_code'].strip() and p.get('kosher_tier')==f.EXTRA_KOSHER:
    # P1.21: counted extra-kosher stock has no FL lot yet. It is loaded as a NEW lot
    # through an owner-approved found ticket; regular lots are never created or retagged.
    # FL stores the canonical code, so a lot made since (or by this reset) matches it.
    code=canonical_lot_code(r['lot_code'])
    matches=[l for l in lots.values() if l['product_id']==pid and l['lot_code']==code]
   if not matches and r['lot_code'].strip() and p.get('kosher_tier')==f.EXTRA_KOSHER:
    check(not any(l['product_id']==pid and l['lot_code'].casefold()==code.casefold() for l in lots.values() if not l.get('new')),'Case-insensitive lot ambiguity')
    key=f'NEW-P{pid}-{code}'
    lots.setdefault(key,dict(id=key,product_id=pid,lot_code=code,status=None,merged_into_lot_id=None,kosher_tier=f.EXTRA_KOSHER,new=True))
    matches=[lots[key]]
   check(r['lot_code'] and len(matches)==1,'Unidentified or unmatched lot; never create/match automatically')
   lot=matches[0]
   check(lot.get('new') or sum(l['lot_code'].casefold()==lot['lot_code'].casefold() for l in lots.values() if l['product_id']==pid and not l.get('new'))==1,'Case-insensitive lot ambiguity')
   check(r['estimated'].strip().lower() in ('','n','no','false','0','y','yes','true','1','x'),'Estimated must be yes/no or a tick')
   check(r.get('duplicate','').strip() in ('','x','X','✓','✔','☑'),'Duplicate must be a tick (x, X, ✓, ✔ or ☑), or blank')
   obs=dict(r,pid=pid,lid=lot['id'],start=start,end=end,cut=cut,qty=convert(r,p),estimated=r['estimated'].strip().lower() in ('y','yes','true','1','x'),inferred=False)
   observations.append(obs)
  except ValueError as e:
   message=f'{rid}: {e}';row_holds.add(message)
   if pid is not None:hold(pid,message)
   sheet_follow_up[r['sheet_id']].append(dict(sheet_id=r['sheet_id'],row_id=rid,product_id=pid,hold=message))
   due=None
   try:due=(at(r['counted_time'] or r['end_time'],r['count_date'])+timedelta(days=7)).isoformat()
   except ValueError:pass
   unknown.append(dict(sheet_id=r['sheet_id'],row_id=rid,product_id=pid,quantity=r['quantity'],unit=r['unit'],description=r['notes'] or r['name'],location=r['location'],lot_code=r['lot_code'],hold=str(e),follow_up_due=due or '7 days after actual count time is supplied'))
 # Even fully logged moves cannot prove that there were no unlogged moves.
 # Bind one owner confirmation per lot to ALL observed rows, before exclusions.
 observed_lots=defaultdict(list)
 for r in observations:
  observed_lots[r['lid']].append(r)
 for lid,rs in observed_lots.items():
  if len({r['area'] for r in rs})<2:continue
  refs=[r['row_id'] for r in rs];message='owner confirms no unlogged moves between rows '+'/'.join(refs)
  # Bound to product + lot code, not the lot id: a new extra-kosher lot's synthetic key
  # becomes a real id once its found ticket posts, and the confirmation must still hold.
  fp=fingerprint(dict(product_id=rs[0]['pid'],lot_code=rs[0]['lot_code'],rows=[{k:x for k,x in r.items() if k!='lid'} for r in rs]));ans=review.get('multi_area_reviews',{}).get(fp,{})
  templates['multi_area_reviews'][fp]=dict(product_id=rs[0]['pid'],lot_id=lid,rows=refs,hold=message,owner='',no_unlogged_moves=False)
  if not (ans.get('owner') and ans.get('no_unlogged_moves') is True):lot_holds[lid].append(message)
 # Only ticked rows nominate duplicate groups; tags group evidence but never
 # mark a row as a duplicate. Free text has no duplicate-selection semantics.
 duplicate_groups=defaultdict(list)
 for r in observations:duplicate_groups[(r['pid'],r['lid'],r['tag_id'].strip())].append(r)
 duplicate_sets=[rs for rs in duplicate_groups.values() if any(r.get('duplicate','').strip() for r in rs)]
 excluded=set();linked_duplicates=[]
 for rs in duplicate_sets:
  fp=fingerprint([r0 for r0 in data['count'] if r0['row_id'] in {r['row_id'] for r in rs}]);ans=review.get('duplicate_reviews',{}).get(fp,{})
  templates['duplicate_reviews'][fp]=dict(rows=[r['row_id'] for r in rs],owner='',action='',exclude_rows=[],move_id='',notes='')
  valid=bool(ans.get('owner') and ans.get('notes'))
  if ans.get('action')=='distinct' and valid:pass
  elif ans.get('action')=='exclude' and valid and ans.get('exclude_rows') and set(ans['exclude_rows'])<={r['row_id'] for r in rs if r.get('duplicate','').strip()}:
   selected=set(ans['exclude_rows'])
   if selected=={r['row_id'] for r in rs}:
    kept=rs[0]['row_id'];selected.remove(kept)
    hold(rs[0]['pid'],'Every row in duplicate group was excluded; kept '+kept+' so at least one count row remains ['+fp+']')
   excluded.update(selected)
  elif ans.get('action')=='linked_move' and valid and ans.get('move_id'):linked_duplicates.append((rs,ans['move_id']))
  else:
   for r in rs:hold(r['pid'],'Duplicate tick needs owner disposition: '+fp)
 observations=[r for r in observations if r['row_id'] not in excluded]
 coverage={};coverage_ids=set()
 for r in data['coverage']:
  if information_reason(r,catalog):continue
  if not r.get('product_id','').strip():continue
  pid=int(r['product_id']);check(pid not in coverage_ids,'Duplicate coverage product');coverage_ids.add(pid)
  if r.get('all_locations_searched','').lower() not in ('yes','y','true'):continue
  try:
   p=products[pid]
   check(r.get('owner_initials'),'Coverage needs owner initials')
   check(r.get('SKU')==(p.get('odoo_code') or '') and r.get('name')==p['name'],'Coverage identity mismatch')
   end=f.stamp(r.get('completed_at'));check(end and end<=now,'Coverage completion time missing/future')
   check(all(end>=o['end'] for o in observations if o['pid']==pid),'Coverage completed before its sheets')
   coverage[pid]=end
  except (ValueError,KeyError) as e:hold(pid,'Coverage: '+str(e))
 for pid in coverage:
  if any(r['pid']==pid for r in observations):continue
  fp=fingerprint(dict(coverage=next(r for r in data['coverage'] if r['product_id']==str(pid)),rows=[r for r in data['count'] if r['product_id']==str(pid)]))
  ans=review.get('coverage_reviews',{}).get(fp,{})
  templates['coverage_reviews'][fp]=dict(product_id=pid,hold='coverage certified but no count rows',owner='',confirmed=False)
  if not (ans.get('owner') and ans.get('confirmed') is True):hold(pid,'coverage certified but no count rows')
 # Missing lots are zero ONLY after a product's all-location search is certified.
 obskeys={(r['pid'],r['lid']) for r in observations}
 for l in lots.values():
  pid=l['product_id']
  if pid not in coverage or (pid,l['id']) in obskeys:continue
  t=coverage[pid]
  observations.append(dict(row_id=f'COVERAGE-P{pid}-L{l["id"]}',pid=pid,lid=l['id'],product_id=str(pid),sheet_id=f'COVERAGE-P{pid}',area='ALL LOCATIONS SEARCHED',location='Absence certified by product coverage',counter='owner coverage',bin_id='',tag_id='',physical_marker='',notes='Inferred absent only from signed all-location coverage',start=t,end=t,cut=t,qty=D(0),estimated=False,inferred=True,lot_code=l['lot_code'],unit=native(products[pid]),quantity='0'))
 groups=defaultdict(list)
 for r in observations:groups[(r['pid'],r['lid'])].append(r)
 # Cross-area moves are physical internal movements, not inventory adjustments.
 move_effect=defaultdict(lambda:D(0));move_records=[];move_follow_up=[];resolved_moves=set();internal_tx=set();used_moveids=set()
 byrow={r['row_id']:r for r in observations}
 for row_number,m in enumerate(data['moves'],start=2):
  if not any(m.values()):continue
  pid=int(m['product_id']) if m.get('product_id','').isdigit() else None
  if pid not in catalog:
   row_ref=f'moves CSV row {row_number}'
   move_follow_up.append(dict(m,row_ref=row_ref,detail=row_ref+': move-log row has no valid product_id'))
   continue
  if information_reason(m,catalog):information_moves.append(m);continue
  fp=fingerprint(m);ans=review.get('move_reviews',{}).get(fp,{})
  templates['move_reviews'][fp]=dict(move=m,owner='',source_row='',destination_row='',source_included=None,destination_included=None,notes='')
  error=''
  try:
   check(m.get('move_id') and m['move_id'] not in used_moveids,'Missing/duplicate move ID');used_moveids.add(m['move_id'])
   p=products[pid];check(m.get('SKU')==(p.get('odoo_code') or ''),'Move SKU mismatch')
   check(ans.get('owner') and ans.get('notes'),'Move needs owner review')
   src=byrow.get(ans.get('source_row'));dst=byrow.get(ans.get('destination_row'))
   # A destination consumed before it could be counted has no destination row.
   # Owner notes cannot release this hold: remove that row from the move log;
   # retain the consumption in the ordinary ledger movement review.
   check(src and dst and src['row_id']!=dst['row_id'],'Move needs distinct source/destination count rows')
   check(lots[dst['lid']].get('status')!='consumed','Move destination is a consumed lot; remove this row from the move log')
   check(src['pid']==dst['pid']==pid and src['lid']==dst['lid'] and m['lot_code']==src['lot_code'],'Move item/lot mismatch')
   check(src['area']==m['from_area'] and dst['area']==m['to_area'] and src['location']==m['from_location'] and dst['location']==m['to_location'],'Move locations must match both count rows')
   t=f.stamp(m['moved_at']);check(t and t<=now,'Move time missing/future')
   check(type(ans.get('source_included')) is bool and type(ans.get('destination_included')) is bool,'Owner must confirm inclusion in both counts')
   amount=convert(dict(quantity=m['quantity'],unit=m['unit'],partial_cases=''),p);check(amount>0,'Move quantity must be positive')
   for rr,key in ((src,'source_included'),(dst,'destination_included')):
    if ans[key]:check(rr['qty']>=amount,'Move quantity exceeds row count marked included')
   if m.get('fl_transaction_id'):
    tid=int(m['fl_transaction_id']);xs=[x for x in lines if x['transaction_id']==tid and x['product_id']==pid and x['lot_id']==src['lid'] and x['effective_status']=='posted']
    check(xs and sum(f.number(x['quantity_lb']) for x in xs)==0,'Logged internal move FL transaction must net zero for this lot')
    internal_tx.add(tid)
   effect=amount*(1-int(ans['source_included'])-int(ans['destination_included']))
   move_effect[(pid,src['lid'])]+=effect;resolved_moves.add(m['move_id'])
  except (ValueError,KeyError,TypeError) as e:error=str(e);hold(pid,'Moved during count: '+error+' ['+fp+']');effect=None
  move_records.append(dict(m,fingerprint=fp,owner_review=ans,effect_native=effect,status='HELD' if error else 'REVIEWED',detail=error))
 for rs,mid in linked_duplicates:
  matches=[m for m in move_records if m.get('move_id')==mid and m['status']=='REVIEWED']
  linked=bool(matches) and {r['row_id'] for r in rs}<={matches[0]['owner_review'].get('source_row'),matches[0]['owner_review'].get('destination_row')}
  if mid not in resolved_moves or not linked:
   for r in rs:hold(r['pid'],'Duplicate tag linked to unresolved/unrelated move '+mid)
 # A logged late movement for any counted item is surfaced, even on an unobserved lot.
 activity=[];row_comparison=[];lotrows=[];area_records=[];own=set(verified_openings)
 count_start=min([r['start'] for rs in observed_lots.values() for r in rs]+[r['start'] for r in observations],default=now)
 for (pid,lid),rs in sorted(groups.items()):
  p=products[pid];lot=lots[lid];xs=[x for x in lines if x['product_id']==pid and x['lot_id']==lid]
  cuts=[r['cut'] for r in rs];earliest=min(cuts);cutoff=max(cuts);after=D(0);existing=D(0)
  current=sum((f.number(x['quantity_lb']) for x in xs if x['effective_status']=='posted'),D(0))
  # Reasons bind count source, lot and actual lot cutoff; estimates remain in the adjustment note.
  # A lot this reset's found ticket created keeps its NEW mark, so rechecks match the signed reason.
  made=lot.get('entry_source')=='found_inventory' and re.match(OPENING+'NEW '+re.escape(lot['lot_code'])+r'(?=\s*\||$)',lot.get('entry_source_notes') or '')
  mark=('NEW '+lot['lot_code']) if lot.get('new') or made else f'L{lid}'
  reason=f'OPENING BALANCE {cutoff.astimezone(f.PLANT).date()} – physical count | v3 {countsha[:12]} {mark} | Estimated={"yes" if any(r["estimated"] for r in rs) else "no"}'
  for x in xs:
   is_own=x['line_id'] in own
   looks_open=looks_like_opening(x,lot,lid)
   # Conservatively hold even voided prior openings until journal proof is supplied.
   if looks_open and not is_own:lot_holds[lid].append('prior opening without journal proof')
   t=f.stamp(x.get('occurred_at'));entered=f.entered_at(x)
   if not t or not entered:hold(pid,'Missing ledger timestamp');continue
   if t>now:hold(pid,'Future ledger movement requires review');continue
   if is_own:existing+=f.number(x['quantity_lb']);continue
   relevant=[];reasons=[]
   for r in rs:
    inside=r['start']<=t<=r['end'];near=abs(t-r['cut'])<=timedelta(minutes=30)
    between=earliest<=t<=cutoff and len(set(cuts))>1
    late=(entered>=count_start and entered-t>timedelta(minutes=15)) or (entered>r['end'] and t<=r['end']+post_sheet_margin(r['start'],r['end']))
    if inside or near or between or late:
     relevant.append(r['row_id'])
     if inside:reasons.append('inside sheet window')
     if near:reasons.append('within 30 minutes')
     if between:reasons.append('between area count times')
     if late:reasons.append('late/back-dated entry')
   q=f.number(x['quantity_lb']);posted=x['effective_status']=='posted'
   fp=fingerprint(dict(line=x,rows=[{k:r[k] for k in ('row_id','sheet_id','start','end','cut','qty','location')} for r in rs]))
   ans=review.get('movement_reviews',{}).get(fp,{})
   carried=q if posted and t>cutoff else D(0)
   if relevant:
    templates['movement_reviews'][fp]=dict(transaction_id=x['transaction_id'],line_id=x['line_id'],product_id=pid,lot_code=lot['lot_code'],type=x['type'],adjust_reason=x.get('adjust_reason'),operator_id=x.get('operator_id'),quantity_native=str(q),unit=native(p),occurred_at=x['occurred_at'],entered_at=entered.isoformat(),reasons=sorted(set(reasons)),owner='',classifications={rid:'' for rid in relevant},allocations={},late_confirmed=False)
    valid=bool(ans.get('owner')) and all(ans.get('classifications',{}).get(rid) in ('before count','after count') for rid in relevant)
    if 'late/back-dated entry' in reasons:valid=valid and ans.get('late_confirmed') is True
    allocation=ans.get('allocations',{})
    if len(rs)==1 and valid:
     carried=q if posted and ans['classifications'][rs[0]['row_id']]=='after count' else D(0)
    elif valid:
     # FL has no location balances. Owner must allocate signed movement among count rows.
     try:
      check(all(rid in byrow and byrow[rid]['pid']==pid and byrow[rid]['lid']==lid for rid in allocation),'Allocation outside lot count rows')
      amounts={rid:f.number(v) for rid,v in allocation.items()}
      check(amounts and sum(amounts.values())==q and all((v*q>0 or v==0) for v in amounts.values()),'Movement allocation must sum to signed ledger quantity')
      check(all(ans.get('classifications',{}).get(rid) in ('before count','after count') for rid in amounts),'Allocated rows need before/after answers')
      carried=sum((v for rid,v in amounts.items() if ans['classifications'][rid]=='after count'),D(0)) if posted else D(0)
     except ValueError:valid=False
    if not valid:hold(pid,'Unclassified movement / allocation: TX'+str(x['transaction_id'])+' ['+fp+']')
   if x['transaction_id'] in internal_tx:carried=D(0)
   after+=carried
   # List every movement since EACH row/sheet cutoff, plus all review-window entries.
   since=[r['row_id'] for r in rs if t>r['cut']]
   if relevant or since:
    activity.append(dict(product_id=pid,lot_id=lid,lot_code=lot['lot_code'],transaction_id=x['transaction_id'],line_id=x['line_id'],type=x['type'],status=x['effective_status'],quantity_native=q,unit=native(p),occurred_at=t.isoformat(),entered_at=entered.isoformat(),operator_id=x.get('operator_id'),since_row_cutoffs=since,within_sheet_windows=[r['sheet_id'] for r in rs if r['start']<=t<=r['end']],review_rows=relevant,reasons=sorted(set(reasons)),fingerprint=fp,classification=ans,carried_after_native=carried))
  physical=sum(r['qty'] for r in rs);target=physical+move_effect[(pid,lid)]+after;original=target-current;rounded=f.round_adjustment(original)
  if target<0:hold(pid,'Negative expected current stock')
  if rounded and (not p.get('active') or p.get('is_service')):hold(pid,'Inactive/service product quantity change needs owner identity decision')
  if (physical or current or rounded) and (lot.get('merged_into_lot_id') or lot.get('status') not in ('active',None)):hold(pid,'Merged/inactive lot with stock')
  # A3b posts a short make/pack negative on purpose and opens a SHORTAGE; the count
  # closes it as 'counted'. Any other negative lot still needs the owner's investigation.
  if current<0 and not any(x['lot_id']==lid for x in shortages):hold(pid,'Negative ledger lot needs owner investigation')
  if pid not in coverage:hold(pid,'All-location product coverage is incomplete')
  new=bool(lot.get('new'));open_short=sorted(x['id'] for x in shortages if x['lot_id']==lid)
  if new and rounded<0:hold(pid,'New extra-kosher lot cannot be loaded negative')
  # A shortage the count does not change cannot close through an adjust ticket; never
  # report the lot reset while it stays open (Codex PR #108 review).
  if open_short and not new and not rounded:hold(pid,'Open shortage #'+',#'.join(map(str,open_short))+' on lot '+lot['lot_code']+' and the count matches FL: resolve it as counted on the dashboard (XR receipt), then refresh the preview')
  if new and any(x['product_id']==pid for x in shortages):hold(pid,'Open shortage on this extra-kosher product: FL refuses found stock until it is resolved')
  if (lot.get('kosher_tier') or p.get('kosher_tier') or 'regular')!=(p.get('kosher_tier') or 'regular') and not new:hold(pid,'Lot kosher tier differs from its product; owner investigation')
  lotrows.append(dict(product_id=pid,SKU=p.get('odoo_code') or '',name=p['name'],product_active=p.get('active'),product_type=p['type'],ledger_unit=native(p),lot_id=None if new else lid,lot_key=str(lid),new_lot=new,action='found' if new else 'adjust',kosher_tier=p.get('kosher_tier') or 'regular',resolves_shortage_ids=open_short if rounded and not new else [],open_shortage_ids=open_short,lot_code=lot['lot_code'],row_ids=[r['row_id'] for r in rs],cutoff=cutoff.isoformat(),row_cutoffs={r['row_id']:r['cut'].isoformat() for r in rs},estimated=any(r['estimated'] for r in rs),physical_count_native=physical,internal_move_correction_native=move_effect[(pid,lid)],post_count_movement_native=after,counted_qty=physical,counted_lb=physical,adjustment_basis_lb=current-after-move_effect[(pid,lid)],candidate_adjustment_lb=original,rounded_adjustment_lb=rounded,rounding_delta_lb=rounded-original,planned_balance_lb=current+rounded,current_lb=current,expected_current_lb=target,existing_opening_lb=existing,reason=reason,backfill=True,time_since_count_hours=(now-cutoff).total_seconds()/3600))
  lotrows[-1]['owner_approvals']=owner_approvals(lotrows[-1],p)
  for r in rs:
   fl=sum((f.number(x['quantity_lb']) for x in xs if x['effective_status']=='posted' and x.get('occurred_at') and f.stamp(x['occurred_at'])<=r['cut']),D(0))
   row_comparison.append(dict(row_id=r['row_id'],sheet_id=r['sheet_id'],area=r['area'],location=r['location'],product_id=pid,lot_id=lid,lot_code=r['lot_code'],count_native=r['qty'],unit=native(p),cutoff=r['cut'].isoformat(),cutoff_source='coverage absence' if r['inferred'] else 'row time' if r.get('counted_time') else 'sheet end fallback',fl_whole_lot_at_row_time=fl,row_minus_whole_lot_reference=r['qty']-fl if len(rs)==1 else None,comparison_note='Whole-lot reference only: FL has no location balances' if len(rs)>1 else 'One observation of lot',estimated=r['estimated'],time_since_count_hours=(now-r['cut']).total_seconds()/3600))
 # Corrections and unassigned lines must never vanish from reconciliation.
 earliest=min((r['start'] for r in observations),default=now)
 pids={r['pid'] for r in observations}
 for x in lines:
  if x['product_id'] not in pids:continue
  if x.get('lot_id') not in lots or lots[x['lot_id']]['product_id']!=x['product_id']:
   hold(x['product_id'],'Ledger line lacks a usable matching lot: '+str(x['line_id']))
  if (x['product_id'],x.get('lot_id')) not in groups and x.get('occurred_at') and f.stamp(x['occurred_at'])>=earliest:
   hold(x['product_id'],'Activity on uncounted lot during/after count: '+str(x['line_id']))
 for c in s.get('corrections',[]):
  if f.stamp(c['created_at'])<earliest:continue
  affected={x['product_id'] for x in lines if (c['target_table']=='transactions' and x['transaction_id']==c['target_id']) or (c['target_table']=='transaction_lines' and x['line_id']==c['target_id'])}&pids
  if not affected:continue
  fp=fingerprint(c);ans=review.get('correction_reviews',{}).get(fp,{})
  templates['correction_reviews'][fp]=dict(correction=c,owner='',confirmed=False)
  if not (ans.get('owner') and ans.get('confirmed') is True):
   for pid in affected:hold(pid,'Correction needs owner confirmation: '+fp)
 for pid,p in products.items():
  if pid in {145,146,147,148,149} and (pid in coverage or any(r['pid']==pid for r in observations)):
   evidence=[r for r in data['count'] if r['product_id']==str(pid)]
   fp=fingerprint(dict(product_id=pid,count_rows=evidence));ans=review.get('ownership_reviews',{}).get(fp,{})
   templates['ownership_reviews'][fp]=dict(product_id=pid,owner='',all_counted_stock_cns_owned=None,evidence_reference='',notes='Office only: reconcile invoices, credits and physical custody after counting')
   if not (ans.get('owner') and ans.get('all_counted_stock_cns_owned') is True and ans.get('evidence_reference')):
    hold(pid,'Sunshine custody/ownership reconciliation pending; customer-owned stock needs separate supported treatment')
  if pid not in coverage:hold(pid,'All-location product coverage is incomplete')
 if areas:
  # One area group at a time (P1.19 option A): a product is in this group only when
  # every one of its count rows is in the selected areas. Everything else waits.
  chosen=set(areas)
  for pid in {r['product_id'] for r in lotrows}:
   seen_areas={o['area'] for o in observations if o['pid']==pid and not o['inferred']}
   if not seen_areas:hold(pid,'Area group: no count rows in '+', '.join(sorted(chosen))+'; apply in the final full pass')
   elif seen_areas-chosen:hold(pid,'Area group: also counted in '+', '.join(sorted(seen_areas-chosen))+'; include those areas or wait')
 for r in lotrows:
  # Unknown products cannot be assigned a product hold. Keep affected sheets'
  # lots held without blocking unrelated sheets through the general bucket.
  # New extra-kosher lots have no FL id yet: their observations and lot holds (e.g. the
  # multi-area no-unlogged-moves confirmation) live under the synthetic key.
  key=r['lot_key'] if r['new_lot'] else r['lot_id']
  sids={obs['sheet_id'] for obs in observed_lots.get(key,[])}
  local=[item['hold'] for sid in sorted(sids) for item in sheet_follow_up[sid] if item['product_id'] is None]
  r['holds']=holds.get(r['product_id'],[])+lot_holds.get(key,[])+local;r['status']='HELD' if r['holds'] else 'READY';r['adjustment_lb']=None if r['holds'] else r['rounded_adjustment_lb']
 for sid,raws in sheetrows.items():
  rs=[r for r in observations if r['sheet_id']==sid];pids_sheet={int(r['product_id']) for r in raws if r['product_id'].isdigit()}
  meta=sheetmeta.get(sid,(raws[0]['area'],raws[0]['counter'],raws[0]['start_time'],raws[0]['end_time']))
  events=[x for x in activity if any(r['row_id'] in x['review_rows'] or r['row_id'] in x['since_row_cutoffs'] for r in rs)]
  local_holds={h for pid in pids_sheet for h in holds[pid] if h not in row_holds}|{x['hold'] for x in sheet_follow_up[sid]}|{h for lid,obs in observed_lots.items() if any(r['sheet_id']==sid for r in obs) for h in lot_holds[lid]}
  area_records.append(dict(sheet_id=sid,area=meta[0],counter=meta[1],start=meta[2],end=meta[3],row_ids=[r['row_id'] for r in raws],movement_line_ids=[x['line_id'] for x in events],classifications_needed=[x['fingerprint'] for x in events if x['review_rows']],holds=sorted(local_holds),follow_up=sheet_follow_up[sid]))
 if holds.get(None):general+=holds[None]
 allready=not any(holds.values()) and not any(lot_holds.values()) and not any(sheet_follow_up.values()) and not move_follow_up and not general and len(coverage)==len(products)
 return dict(version=3,reset_scope_policy=f.RESET_SCOPE_POLICY,information_products=information_products,packaging_information_only=information,packaging_moves_information_only=information_moves,baseline={str(pid):baseline(s,pid) for pid in products},snapshot_at=s['snapshot_at'],target=s.get('target','production'),ledger_lines=[dict(transaction_id=x['transaction_id'],product_id=x['product_id'],lot_id=x.get('lot_id')) for x in lines if x['product_id'] in {r['product_id'] for r in lotrows}],area_group=sorted(areas) if areas else None,input_sha256=countsha,sources=sources,data=data,rows=lotrows,row_comparison=row_comparison,activity=activity,area_reconciliation=area_records,moved_during_count=move_records,move_follow_up=move_follow_up,unidentified_follow_up=unknown,sheet_follow_up={sid:items for sid,items in sheet_follow_up.items() if items},review_template=templates,holds={str(k):v for k,v in holds.items() if v},lot_holds={str(k):v for k,v in lot_holds.items() if v},general=general,products=products,coverage_complete=sorted(coverage),full_scope_reviewed=allready,excluded_duplicate_rows=sorted(excluded))

def write_csv(path,rows):
 if not rows:path.write_text('status\nNo rows\n');return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=keys);w.writeheader()
  w.writerows({k:canonical(v) if isinstance(v,(dict,list)) else f.fmt(v) for k,v in r.items()} for r in rows)
def report(a,output,verify=False):
 output=f.local_path(output);output.mkdir(parents=True,exist_ok=True)
 for src in a['sources'].values():check(not Path(src['path']).is_relative_to(output),'Keep filled inputs outside output directory to prevent overwrites')
 suffix='verification' if verify else 'preview'
 (output/f'reset-{suffix}-v3.json').write_text(json.dumps(a,indent=2,default=str)+'\n')
 for name,key in [('lot-comparison','rows'),('row-cutoffs','row_comparison'),('all-movements','activity'),('area-reconciliation','area_reconciliation'),('moved-during-count-review','moved_during_count'),('move-follow-up','move_follow_up'),('unidentified-seven-day-follow-up','unidentified_follow_up'),('packaging-information-only','packaging_information_only'),('packaging-moves-information-only','packaging_moves_information_only')]:
  records=a[key]
  if key=='rows':records=[{(k[:-3]+'_native' if k.endswith('_lb') else k):value for k,value in r.items()} for r in records]
  write_csv(output/(name+'.csv'),records)
 write_csv(output/'sheet-follow-up.csv',[r for rows in a['sheet_follow_up'].values() for r in rows])
 info=['# Packaging counts — information only','', '**Plain-English summary:** Packaging is counted on the floor to show what is there. It is excluded from the reset because FL does not deduct packaging when packing, so a loaded balance would immediately become unreliable. Pallets (102) and Pallet Charge (176) are billing items and are also excluded. No balance target, adjustment or lot creation is proposed for these items.','',f'This report uses count inputs preserved in the companion reset JSON and the saved catalog snapshot at {a["snapshot_at"]}. There are {len(a["information_products"])} excluded catalog products. Missing quantities mean NOT COUNTED or QUANTITY NOT ENTERED, never zero. An explicitly written zero is retained as an observation only.','', 'Rows below are the original observations, not reconciled totals. Units, lot text, locations, estimates and notes remain as entered in `packaging-information-only.csv`; unknown lots, duplicated tags and moving stock may need an informational recount. Do not sum different units or overlapping observations. `packaging-moves-information-only.csv` preserves logged moves without generating reset holds or corrections. Neither report is a stock valuation or reset sign-off requirement.','']
 info+=f.table(['ID','SKU / name','Area / location','Lot','Quantity as entered','Unit','Status'],[[r['product_id'],str(r.get('SKU',''))+' / '+r.get('name',''),r.get('area','')+' / '+r.get('location',''),r.get('lot_code',''),r.get('quantity',''),r.get('unit',''),r['status']] for r in a['packaging_information_only']])
 (output/'packaging-information-only.md').write_text('\n'.join(info)+'\n')
 path=output/'owner-review-template.json';i=1
 while path.exists():i+=1;path=output/f'owner-review-template-{i}.json'
 path.write_text(json.dumps(a['review_template'],indent=2,default=str)+'\n')
 ages=[r['time_since_count_hours'] for r in a['row_comparison']]
 remaining=[r for r in a['rows'] if abs(f.number(r['expected_current_lb'])-f.number(r['current_lb']))>f.TOLERANCE]
 complete=a['full_scope_reviewed'] and not remaining
 text=['# Live-count v3 '+suffix,'',f'**Plain-English summary:** {len(a["row_comparison"])} usable reset lot/location observations; {sum(r["status"]=="READY" for r in a["rows"])} ready lot comparisons; {len(a["holds"])} product/identity groups, {len(a["lot_holds"])} lots and {len(a["sheet_follow_up"])} sheets held. No stock was changed. '+('FULL RESET SCOPE RECONCILED.' if verify and complete else 'This is not full-reset sign-off.'),'',f'Reset scope: {len(a["products"])} products. Packaging and billing items are excluded ({len(a["information_products"])} catalog products); see [the separate information-only report](packaging-information-only.md). Packaging coverage, unknown lots and quantities cannot generate reset adjustments or holds. This verdict concerns only the reset scope, not completion of the informational floor count.','',f'**TIME SINCE COUNT: {min(ages):.2f} to {max(ages):.2f} hours** at snapshot {a["snapshot_at"]}.' if ages else '**TIME SINCE COUNT: unavailable; no completed physical count rows supplied.**','No waiting period or age deadline. Every logged reset-scope movement since each cutoff is in `all-movements.csv`. New activity requires a refreshed review.','', 'Counts use each row time, falling back to sheet end. Multiple locations cannot each be compared as if FL held separate location balances: FL only has the total lot balance. `row-cutoffs.csv` shows that whole-lot reference at each observed time. Classified movements are allocated once among locations to bring the combined count forward. The proposed adjustment is one per lot, dated at its latest count time, with all underlying row cutoffs retained.','', 'Estimated counts are marked on the preview and in the exact adjustment reason. Unidentified reset-scope lots remain held with a seven-day follow-up; no new lots are created. Empty count rows are not zeros. Missing lot zeros require certified product coverage.','', '## Per-area reconciliation','']
 text+=f.table(['Sheet','Area','Start / end','Rows','Movement lines','Holds'],[[r['sheet_id'],r['area'],r['start']+' / '+r['end'],', '.join(r['row_ids']),', '.join(map(str,r['movement_line_ids'])),'; '.join(r['holds'])] for r in a['area_reconciliation']])
 text+=['','## Lot comparisons','']
 text+=f.table(['Item / lot','Unit','Physical total','After-count movement','Internal move correction','FL now','Expected now','Proposed change','Estimated','Status'],[[str(r['product_id'])+' / '+r['lot_code'],r['ledger_unit'],r['physical_count_native'],r['post_count_movement_native'],r['internal_move_correction_native'],r['current_lb'],r['expected_current_lb'],r['adjustment_lb'],r['estimated'],r['status']] for r in a['rows']])
 special=[r for r in a['rows'] if r['status']=='READY' and r['adjustment_lb'] and (r['owner_approvals'] or r['resolves_shortage_ids'] or r['new_lot'])]
 text+=['','## Owner-approved posts in this preview','', 'Every reset post is an FL ticket under the named owner key with reason `physical_count` and a receipt number. Rows below meet an FL owner hold or close an open shortage. Signing the approval with `Owner approves held entries: Y` and listing every new extra-kosher lot authorizes the executor to approve exactly these holds with the same key; anything else stops.','']
 text+=f.table(['Item / lot','FL action','Change','Owner holds','Closes shortages'],[[str(r['product_id'])+' / '+r['lot_code']+(' (NEW extra-kosher lot)' if r['new_lot'] else ''),'found ticket (FND)' if r['action']=='found' else 'adjust ticket (ADJ)',r['adjustment_lb'],', '.join(r['owner_approvals']) or '-',', '.join(map(str,r['resolves_shortage_ids'])) or '-'] for r in special])
 stays=[r for r in a['rows'] if r['open_shortage_ids'] and not r['resolves_shortage_ids']]
 if stays:text+=['','Open shortages that this preview will NOT close (no change or held): '+'; '.join(str(r['product_id'])+' / '+r['lot_code']+' #'+','.join(map(str,r['open_shortage_ids'])) for r in stays)]
 if a.get('area_group'):text+=['',f'**Area group:** {", ".join(a["area_group"])}. Products also counted elsewhere, or with no rows here, are held for a later group or the final full pass.']
 text+=['','## Holds','']
 for pid,hs in a['holds'].items():text.append('- '+pid+': '+'; '.join(hs))
 for lid,hs in a['lot_holds'].items():text.append('- Lot '+lid+': '+'; '.join(hs))
 for sid,items in a['sheet_follow_up'].items():text.append('- Sheet '+(sid or '(missing sheet ID)')+': '+'; '.join(r['hold'] for r in items))
 text+=['','## Move-log follow-up','', 'Correct these rows in the move log and rerun. An unidentified move does not block review of unrelated catalog products; see `move-follow-up.csv`.','']
 text+=f.table(['Row reference','Move ID','Product ID as entered','Follow-up'],[[r['row_ref'],r.get('move_id',''),r.get('product_id',''),r['detail']] for r in a['move_follow_up']])
 text+=['',MOVE_DETECTION_NOTE]
 text+=['','## Owner review','',f'Fill `{path.name}` in a separate inputs folder. Answers bind exact movement, row times and quantities through fingerprints. Each required movement needs before/after answers; multi-area movements also need signed quantity allocations. Late entries need explicit confirmation. Move-log entries need both endpoint rows and inclusion decisions. Duplicate ticks and corrections need owner disposition. Each multi-area lot also needs confirmation of no unlogged moves; sheet follow-ups and coverage-without-rows reviews must be resolved.','', 'Only the separate `apply_reset.py` can submit an approved reset. This report and generator have no production-write capability. Intentional historical posting (`backfill=true`) is included in the signed v3 plan so actual count times can be retained even on an older reviewed count. There is no automatic retry of an uncertain write.']
 (output/f'reset-{suffix}-v3.md').write_text('\n'.join(text)+'\n')
 return 0 if not verify or complete else 2

RECEIPT=re.compile(r'(ADJ|FND)-\d{6}-\d{3,}')


def ticket_payload(r):
    """The exact prepare body for one lot; journaled and re-derived for proof."""
    common = dict(occurred_at=r['cutoff'], backfill=True, client_source='api', reason_code='physical_count')
    if r.get('new_lot'):
        return dict(common, product_id=r['pid'], quantity=float(r['change']), uom='lb', lot_code=r['lot_code'],
                    notes=r['reason'], found_location='Cutover physical count', estimated_age='unknown')
    body = dict(common, lot_id=r['lid'], delta_lb=float(r['change']), note=r['reason'])
    if r.get('resolves_shortage_ids'):
        body['resolves_shortage_ids'] = list(r['resolves_shortage_ids'])
    return body


def actor_id(actor_name, s):
    found = [a for a in s.get('actors', []) if a['name'] == actor_name and a.get('active')]
    check(len(found) == 1, 'Named actor is not exactly one active FL actor.')
    return found[0]['id']


def zero_stock_products(a):
    """Reset-scope products a signed preview may approve with no lot row: coverage-certified,
    no counted or on-hand lot, no product hold. Shared by apply_reset and the whole-reset proof."""
    with_rows = {r['product_id'] for r in a['rows']}
    return {int(pid) for pid in a['products'] if int(pid) not in with_rows and int(pid) in set(a['coverage_complete'])
            and str(pid) not in a['holds']}


TICKET_DEAD = ('expired', 'rejected', 'superseded')   # FL never posts these (commit refuses them)


def reconcile_tickets(rows, s, attempts_list, actor_name):
    """Completed lot -> its one posted ledger line, proved by the ticket that posted it.
    Also returns lots whose last ticket is held for the owner (resume by approval) and
    lots whose commit outcome is UNCERTAIN: the ticket is still `prepared` and some
    commit we sent for it has no definite answer (timeout, crash, lost response). Such
    a commit may still land, so the executor must re-commit THAT ticket (FL replays or
    posts it exactly once under the ticket row lock) and never prepare a second one
    (Codex PR #108 review: timeout + rerun posted twice).
    Read-only: shared by apply_reset (preflight) and verify (journal proof)."""
    tickets = {t['id']: t for t in s.get('correction_tickets', [])}
    lots = {l['id']: l for l in s['lots']}
    owner_id = actor_id(actor_name, s)
    completed, held, uncertain, attempts = {}, {}, {}, {}
    for intent, receipt in attempts_list:
        check(intent['phase'] in ('commit', 'approve'), 'Unknown journal phase.')
        check(intent['lot_id'] in rows and intent['product_id'] == rows[intent['lot_id']]['pid'], 'Out-of-plan journal intent.')
        attempts.setdefault(intent['lot_id'], []).append((intent, receipt))
    for lid, items in attempts.items():
        r = rows[lid]
        posted, open_commits, waiting = [], {}, set()
        for intent, receipt in items:
            tid = intent['ticket_id']
            check(type(tid) is int and tid > 0, 'Journal intent has no ticket.')
            if intent['phase'] == 'commit':
                check(intent['payload'] == ticket_payload(r), 'Journal request differs from approved plan.')
            lines = [x for x in s['lines'] if x.get('ticket_id') == tid]
            ticket = tickets.get(tid)
            check(ticket is not None and ticket['action'] == ('found' if r.get('new_lot') else 'adjust'),
                    f'Lot {lid}: journal ticket is missing from FL; manual reconciliation required.')
            if lines:
                posted.append((tid, lines, ticket))
            elif ticket['status'] == 'awaiting_approval':
                held[lid] = tid
                waiting.add(tid)
            elif ticket['status'] == 'prepared':
                # Not posted YET. Definite only if every commit we sent was refused (4xx).
                if intent['phase'] == 'commit' and (receipt is None or receipt.get('outcome') != 'refused'):
                    open_commits[tid] = intent
            else:
                check(ticket['status'] in TICKET_DEAD, f'Lot {lid}: ticket {ticket["status"]} with no ledger line; stop.')
        check(len(open_commits) <= 1, f'Lot {lid}: several tickets with uncertain commits; manual reconciliation required.')
        if not posted:
            if open_commits:
                uncertain[lid] = next(iter(open_commits.values()))
            continue
        check(not open_commits and len({tid for tid, _, _ in posted}) == 1 and len(posted[0][1]) == 1,
                f'Lot {lid}: more than one posted or pending reset ticket; stop, never resubmit.')
        tid, [x], ticket = posted[0]
        # A second ticket still held for the owner could post the same lot again once
        # approved: never treat the lot as done (Codex PR #120 review).
        check(not waiting - {tid}, f'Lot {lid}: a reset is posted and ticket {sorted(waiting - {tid})} is still held for the owner; '
              'manual reconciliation required, never approve it.')
        lot = lots.get(x['lot_id'])
        check(lot is not None and lot['product_id'] == r['pid'] and x['product_id'] == r['pid'] and
                (lot['id'] == lid if not r.get('new_lot') else lot['lot_code'] == r['lot_code']),
                f'Lot {lid}: posted line is on the wrong product/lot.')
        check(x['effective_status'] == 'posted' and x['type'] == 'adjust' and
                f.number(x['quantity_lb']) == f.number(str(float(r['change']))) and
                f.stamp(x['occurred_at']) == f.stamp(r['cutoff']) and x.get('operator_id') == actor_name and
                x.get('reason_code') == 'physical_count' and x.get('receipt_number') and RECEIPT.fullmatch(x['receipt_number']) and
                x['receipt_number'] == ticket['receipt_number'] and ticket['status'] == 'committed' and
                not x.get('line_correction_at') and not x.get('transaction_correction_at'),
                f'Lot {lid}: posted reset has wrong actor/date/quantity/reason/receipt or was corrected; stop.')
        check(s['transaction_line_counts'].get(str(x['transaction_id'])) == 1, 'Reset transaction has multiple lines or no complete readback; stop.')
        if r.get('new_lot'):
            check(x['receipt_number'].startswith('FND-') and lot.get('kosher_tier') == f.EXTRA_KOSHER and
                    lot.get('kosher_attested_by_actor_id') == owner_id and x.get('kosher_attested_by_actor_id') == owner_id and
                    x.get('transaction_kosher_tier') == f.EXTRA_KOSHER and lot.get('entry_source') == 'found_inventory' and
                    lot.get('entry_source_notes') == r['reason'],
                    f'Lot {lid}: extra-kosher load lacks tier/attestation by {actor_name}; stop.')
        else:
            check(x['receipt_number'].startswith('ADJ-') and r['reason'] in (x.get('notes') or ''), f'Lot {lid}: reset note missing; stop.')
            for exception_id in r.get('resolves_shortage_ids') or []:
                shortage = next((e for e in s.get('shortages', []) if e['id'] == exception_id), None)
                check(shortage and shortage['status'] == 'resolved' and shortage['resolution_kind'] == 'counted' and
                        shortage['resolution_ticket_id'] == tid, f'Lot {lid}: shortage #{exception_id} was not closed by this count.')
        for e in s.get('held_exceptions', []):
            if e['ticket_id'] == tid:
                check(e['status'] == 'resolved' and e['resolution_kind'] in ('approved', 'photo_attached') and
                        e['kind'] in (r.get('owner_approvals') or []), f'Lot {lid}: owner hold #{e["id"]} not released as approved.')
        held.pop(lid, None)
        completed[lid] = x
    for lid in held:
        check(lid not in uncertain, f'Lot {lid}: a held ticket and an uncertain commit; manual reconciliation required.')
    return completed, held, uncertain


def journal_openings(preview_path,approval_path,journal_path,s,data):
 """Read-only proof of prior executor writes. Never imports the write executor."""
 return {x['line_id'] for x in journal_proof(preview_path,approval_path,journal_path,s,data)['completed'].values()}


def journal_proof(preview_path,approval_path,journal_path,s,data):
 """journal_openings with its evidence: proved posts per lot, held/uncertain lots,
 the plan rows with a change, approval fields and the journal's intents/receipts.
 whole_reset_proof.py uses this per area group (O-3)."""
 preview_path,approval_path,journal_path=map(f.local_path,(preview_path,approval_path,journal_path))
 raw=preview_path.read_bytes();plan=json.loads(raw,parse_float=D);signed=approval_path.read_bytes()
 check(plan.get('version')==3,'Journal verification requires a v3 preview')
 check(plan.get('reset_scope_policy')==f.RESET_SCOPE_POLICY,'Reset scope decision changed; regenerate and reapprove preview')
 for r in plan['rows']:
  check(not f.reset_exclusion_reason(dict(id=r['product_id'],type=r['product_type'])),'Packaging/billing item in reset journal plan')
 current={p['id']:p for p in s['products']}
 for r in plan['rows']:
  check(r['product_id'] in current and not f.reset_exclusion_reason(current[r['product_id']]),'Reset journal product is missing or excluded by current scope')
 fields={}
 for line in signed.decode().splitlines():
  if ':' in line:
   k,value=line.split(':',1)
   if k in ('SHA-256','Named actor','Owner name','Signature'):
    check(k not in fields,'Duplicate approval field');fields[k]=value.strip()
 check(fields.get('SHA-256')==sha(raw) and fields.get('Signature')==fields.get('Owner name') and fields.get('Named actor'),'Signed original preview required')
 check(canonical(plan['data'])==canonical(data),'Verification inputs differ from the signed count/review')
 events=[];previous='';journal=journal_path.read_bytes()
 check(journal and journal.endswith(b'\n'),'Missing/torn executor journal')
 for index,line in enumerate(journal.splitlines()):
  event=json.loads(line);saved=event.pop('sha256')
  check(event['seq']==index and event['previous']==previous and sha(canonical(event).encode())==saved,'Journal chain mismatch')
  previous=saved;events.append(event)
 header=events[0]
 check(header.get('kind')=='header' and header.get('preview_sha256')==sha(raw) and header.get('approval_sha256')==sha(signed) and header.get('count_sha256')==data['sources']['count']['sha256'] and header.get('actor')==fields['Named actor'],'Journal belongs to another approval')
 intents={};receipts={}
 for event in events[1:]:
  if event['kind']=='intent':intents[event['seq']]=event
  elif event['kind']=='receipt':
   check(event['intent'] in intents and event['intent'] not in receipts,'Invalid journal receipt');receipts[event['intent']]=event
  else:check(False,'Unknown journal record')
 rows={}
 for r in plan['rows']:
  if r.get('adjustment_lb') is None or not f.number(r['adjustment_lb']):continue
  lid=r['lot_key'] if r.get('new_lot') else r['lot_id']
  rows[lid]=dict(r,pid=r['product_id'],lid=lid,change=f.number(r['adjustment_lb']))
 completed,held,uncertain=reconcile_tickets(rows,s,[(i,receipts.get(seq)) for seq,i in intents.items()],fields['Named actor'])
 return dict(completed=completed,held=held,uncertain=uncertain,rows=rows,fields=fields,plan=plan,intents=intents,receipts=receipts,
             preview_sha256=sha(raw),approval_sha256=sha(signed),journal_sha256=sha(journal))


def main(verify=False):
 p=argparse.ArgumentParser(description=__doc__+' '+MOVE_DETECTION_NOTE);p.add_argument('count_csv',nargs='?');p.add_argument('--coverage');p.add_argument('--moves');p.add_argument('--review');p.add_argument('--snapshot',help='Saved read-only evidence; omitted means a fresh read-only snapshot');p.add_argument('--output-dir')
 p.add_argument('--write-count-template',help='Copy the blank issued CSV with the duplicate tick column; no snapshot is read')
 p.add_argument('--approved-preview');p.add_argument('--approval');p.add_argument('--journal')
 p.add_argument('--target',choices=sorted(f.TARGETS),default='production',help='Which FL to read (default production; staging only for rehearsal)')
 p.add_argument('--area',action='append',help='Limit approvable posts to products counted only in these areas (repeat per area)')
 args=p.parse_args()
 if args.write_count_template:
  check(not verify and not any((args.count_csv,args.coverage,args.moves,args.review,args.snapshot,args.output_dir,args.approved_preview,args.approval,args.journal)),'Template mode cannot be combined with reconciliation inputs')
  write_count_template(args.write_count_template);print(MOVE_DETECTION_NOTE);return 0
 if not all((args.count_csv,args.coverage,args.moves,args.output_dir)):p.error('count_csv, --coverage, --moves and --output-dir are required')
 data=load_inputs(args.count_csv,args.coverage,args.moves,args.review)
 s=json.loads(f.local_path(args.snapshot).read_text(),parse_float=D) if args.snapshot else snapshot(args.target)
 check(s.get('target',args.target)==args.target,'Saved snapshot was read from another target')
 s['target']=args.target
 own=set()
 if args.approved_preview or args.approval or args.journal:
  check(verify and args.approved_preview and args.approval and args.journal,'Standalone journal verification needs all three proof files')
  own=journal_openings(args.approved_preview,args.approval,args.journal,s,data)
 return report(analyze(s,data,own,args.area),args.output_dir,verify)
if __name__=='__main__':raise SystemExit(main())
