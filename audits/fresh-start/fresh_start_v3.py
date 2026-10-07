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

def snapshot():
 # Fixed statement, explicitly transaction-read-only, rollback, session-mode port.
 check(SQL.startswith('BEGIN;\nSET TRANSACTION READ ONLY;'),'Read-only query wrapper missing')
 check(SQL.rstrip().endswith('ROLLBACK;'),'Read-only rollback missing')
 check(not re.search(r'\b(INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|COMMIT|COPY|CALL|DO)\b',SQL,re.I),'Mutation rejected')
 url=f.CREDENTIAL.read_text().strip().replace(':6543/',':5432/')
 check(urlsplit(url).port==5432,'Read-only connection must use port 5432')
 env=dict(os.environ,DATABASE_URL=url,PGCONNECT_TIMEOUT='15');env.pop('PGOPTIONS',None)
 p=subprocess.run([str(f.WRAPPER),'-X','-A','-t','-v','ON_ERROR_STOP=1'],input=SQL,text=True,capture_output=True,env=env,cwd=f.ROOT,timeout=120)
 check(p.returncode==0,'Read-only query failed; connection details suppressed')
 check('ROLLBACK' in p.stdout.splitlines(),'Rollback acknowledgement missing')
 try:s=json.JSONDecoder(parse_float=D).raw_decode(p.stdout[p.stdout.index('{'):])[0]
 except (ValueError,TypeError):raise RuntimeError('Invalid read-only snapshot; details suppressed') from None
 check(s.get('read_only')=='on','Database did not confirm read-only transaction')
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

def baseline(s,pid,exclude=()):
 xs=[x for x in s['lines'] if x['product_id']==pid and x['line_id'] not in set(exclude)]
 tx={x['transaction_id'] for x in xs};lineids={x['line_id'] for x in xs}
 cs=[c for c in s.get('corrections',[]) if (c['target_table']=='transactions' and c['target_id'] in tx) or (c['target_table']=='transaction_lines' and c['target_id'] in lineids)]
 return fingerprint(dict(product=next((p for p in s['products'] if p['id']==pid),None),lots=sorted([l for l in s['lots'] if l['product_id']==pid],key=lambda l:l['id']),lines=sorted(xs,key=lambda l:l['line_id']),corrections=sorted(cs,key=lambda c:str(c['id']))))

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

def analyze(s,data,verified_openings=()):
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
   check(r['lot_code'] and len(matches)==1,'Unidentified or unmatched lot; never create/match automatically')
   lot=matches[0]
   check(sum(l['lot_code'].casefold()==r['lot_code'].casefold() for l in lots.values() if l['product_id']==pid)==1,'Case-insensitive lot ambiguity')
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
  fp=fingerprint(dict(lot_id=lid,rows=rs));ans=review.get('multi_area_reviews',{}).get(fp,{})
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
  reason=f'OPENING BALANCE {cutoff.astimezone(f.PLANT).date()} – physical count | v3 {countsha[:12]} L{lid} | Estimated={"yes" if any(r["estimated"] for r in rs) else "no"}'
  for x in xs:
   is_own=x['line_id'] in own
   looks_open=bool(re.match(r'^OPENING BALANCE .+? \| v3 \S+ L'+str(lid)+r'(?=\s*\||$)',x.get('adjust_reason') or ''))
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
  if current<0:hold(pid,'Negative ledger lot needs owner investigation')
  if pid not in coverage:hold(pid,'All-location product coverage is incomplete')
  lotrows.append(dict(product_id=pid,SKU=p.get('odoo_code') or '',name=p['name'],product_active=p.get('active'),product_type=p['type'],ledger_unit=native(p),lot_id=lid,lot_code=lot['lot_code'],row_ids=[r['row_id'] for r in rs],cutoff=cutoff.isoformat(),row_cutoffs={r['row_id']:r['cut'].isoformat() for r in rs},estimated=any(r['estimated'] for r in rs),physical_count_native=physical,internal_move_correction_native=move_effect[(pid,lid)],post_count_movement_native=after,counted_qty=physical,counted_lb=physical,adjustment_basis_lb=current-after-move_effect[(pid,lid)],candidate_adjustment_lb=original,rounded_adjustment_lb=rounded,rounding_delta_lb=rounded-original,planned_balance_lb=current+rounded,current_lb=current,expected_current_lb=target,existing_opening_lb=existing,reason=reason,backfill=True,time_since_count_hours=(now-cutoff).total_seconds()/3600))
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
 for r in lotrows:
  # Unknown products cannot be assigned a product hold. Keep affected sheets'
  # lots held without blocking unrelated sheets through the general bucket.
  sids={obs['sheet_id'] for obs in observed_lots.get(r['lot_id'],[])}
  local=[item['hold'] for sid in sorted(sids) for item in sheet_follow_up[sid] if item['product_id'] is None]
  r['holds']=holds.get(r['product_id'],[])+lot_holds.get(r['lot_id'],[])+local;r['status']='HELD' if r['holds'] else 'READY';r['adjustment_lb']=None if r['holds'] else r['rounded_adjustment_lb']
 for sid,raws in sheetrows.items():
  rs=[r for r in observations if r['sheet_id']==sid];pids_sheet={int(r['product_id']) for r in raws if r['product_id'].isdigit()}
  meta=sheetmeta.get(sid,(raws[0]['area'],raws[0]['counter'],raws[0]['start_time'],raws[0]['end_time']))
  events=[x for x in activity if any(r['row_id'] in x['review_rows'] or r['row_id'] in x['since_row_cutoffs'] for r in rs)]
  local_holds={h for pid in pids_sheet for h in holds[pid] if h not in row_holds}|{x['hold'] for x in sheet_follow_up[sid]}|{h for lid,obs in observed_lots.items() if any(r['sheet_id']==sid for r in obs) for h in lot_holds[lid]}
  area_records.append(dict(sheet_id=sid,area=meta[0],counter=meta[1],start=meta[2],end=meta[3],row_ids=[r['row_id'] for r in raws],movement_line_ids=[x['line_id'] for x in events],classifications_needed=[x['fingerprint'] for x in events if x['review_rows']],holds=sorted(local_holds),follow_up=sheet_follow_up[sid]))
 if holds.get(None):general+=holds[None]
 allready=not any(holds.values()) and not any(lot_holds.values()) and not any(sheet_follow_up.values()) and not general and len(coverage)==len(products)
 return dict(version=3,reset_scope_policy=f.RESET_SCOPE_POLICY,information_products=information_products,packaging_information_only=information,packaging_moves_information_only=information_moves,baseline={str(pid):baseline(s,pid) for pid in products},snapshot_at=s['snapshot_at'],input_sha256=countsha,sources=sources,data=data,rows=lotrows,row_comparison=row_comparison,activity=activity,area_reconciliation=area_records,moved_during_count=move_records,move_follow_up=move_follow_up,unidentified_follow_up=unknown,sheet_follow_up={sid:items for sid,items in sheet_follow_up.items() if items},review_template=templates,holds={str(k):v for k,v in holds.items() if v},lot_holds={str(k):v for k,v in lot_holds.items() if v},general=general,products=products,coverage_complete=sorted(coverage),full_scope_reviewed=allready,excluded_duplicate_rows=sorted(excluded))

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

def journal_openings(preview_path,approval_path,journal_path,s,data):
 """Read-only proof of prior executor writes. Never imports the write executor."""
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
 planned={r['lot_id']:r for r in plan['rows']};verified=set();used=set();intents={};receipts={}
 for event in events[1:]:
  if event['kind']=='intent':intents[event['seq']]=event
  elif event['kind']=='receipt':
   check(event['intent'] in intents and event['intent'] not in receipts,'Invalid journal receipt');receipts[event['intent']]=event
  else:check(False,'Unknown journal record')
 for seq,intent in intents.items():
  if intent['phase']!='commit':continue
  lid=intent['lot_id'];check(lid in planned and lid not in used,'Duplicate/out-of-plan commit');used.add(lid);r=planned[lid]
  q=f.number(r['adjustment_lb']);check(q!=0,'Zero/held commit in journal')
  body=dict(mode='commit',product_name=r['SKU'],lot_code=r['lot_code'],adjustment_lb=float(q),reason=r['reason'],occurred_at=r['cutoff'],backfill=True)
  check(intent['payload']==body and intent['product_id']==r['product_id'],'Journal payload differs from preview')
  known=intent.get('known_transaction_ids');check(isinstance(known,list),'Journal transaction baseline missing')
  xs=[x for x in s['lines'] if x['product_id']==r['product_id'] and x['lot_id']==lid and x['type']=='adjust' and x.get('adjust_reason')==r['reason'] and x['transaction_id'] not in known]
  check(len(xs)==1,'Uncertain write cannot be proved uniquely');x=xs[0]
  check(x['effective_status']=='posted' and f.number(x['quantity_lb'])==f.number(str(float(q))) and x['operator_id']==fields['Named actor'] and f.stamp(x['occurred_at'])==f.stamp(r['cutoff']) and not x.get('line_correction_at') and not x.get('transaction_correction_at'),'Posted reset actor/date/quantity/status mismatch')
  check(s['transaction_line_counts'].get(str(x['transaction_id']))==1,'Reset transaction is not one line')
  receipt=receipts.get(seq,{})
  check(not receipt.get('transaction_id') or receipt['transaction_id']==x['transaction_id'],'Receipt differs from ledger')
  verified.add(x['line_id'])
 return verified


def main(verify=False):
 p=argparse.ArgumentParser(description=__doc__+' '+MOVE_DETECTION_NOTE);p.add_argument('count_csv',nargs='?');p.add_argument('--coverage');p.add_argument('--moves');p.add_argument('--review');p.add_argument('--snapshot',help='Saved read-only evidence; omitted means a fresh read-only snapshot');p.add_argument('--output-dir')
 p.add_argument('--write-count-template',help='Copy the blank issued CSV with the duplicate tick column; no snapshot is read')
 p.add_argument('--approved-preview');p.add_argument('--approval');p.add_argument('--journal')
 args=p.parse_args()
 if args.write_count_template:
  check(not verify and not any((args.count_csv,args.coverage,args.moves,args.review,args.snapshot,args.output_dir,args.approved_preview,args.approval,args.journal)),'Template mode cannot be combined with reconciliation inputs')
  write_count_template(args.write_count_template);print(MOVE_DETECTION_NOTE);return 0
 if not all((args.count_csv,args.coverage,args.moves,args.output_dir)):p.error('count_csv, --coverage, --moves and --output-dir are required')
 data=load_inputs(args.count_csv,args.coverage,args.moves,args.review)
 s=json.loads(f.local_path(args.snapshot).read_text(),parse_float=D) if args.snapshot else snapshot()
 own=set()
 if args.approved_preview or args.approval or args.journal:
  check(verify and args.approved_preview and args.approval and args.journal,'Standalone journal verification needs all three proof files')
  own=journal_openings(args.approved_preview,args.approval,args.journal,s,data)
 return report(analyze(s,data,own),args.output_dir,verify)
if __name__=='__main__':raise SystemExit(main())
