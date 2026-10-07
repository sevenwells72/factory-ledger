"""Offline blind materials and scope report. No database/API/write-to-FL path."""
import csv,json,re
from pathlib import Path
from collections import defaultdict,Counter
from decimal import Decimal as D
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape,letter
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
BASE=Path(__file__).resolve().parent
V=BASE/'v3'
S=json.loads((V/'raw/01-scope.json').read_text(),parse_float=D)
COUNT_COLUMNS=['sheet_id','area','count_date','counter','start_time','end_time','row_id','product_id','SKU','name','row_kind','lot_code','quantity','unit','partial_cases','location','bin_id','counted_time','estimated','tag_id','physical_marker','notes']
MOVE_COLUMNS=['move_id','product_id','SKU','lot_code','quantity','unit','from_area','from_location','to_area','to_location','moved_at','tag_id','fl_transaction_id','notes']

def category(p):
 if p['type']=='packaging':return '07 Packaging - bags, boxes, tape and pallets'
 if p['type']=='consumable':return '08 Consumables'
 if p['id']==291:return '03 Other WIP - container stock'
 if p['type']=='ingredient':return '02 Coconut raw materials' if 'coconut' in p['name'].lower() else '01 Ingredients'
 if p['type']=='batch':return '04 Coconut bulk WIP' if 'coconut' in p['name'].lower() else '05 Granola bulk WIP'
 return '06 Finished goods - all customers'

def unit(p):
 if p['type']=='finished':return 'cases'
 u=p.get('uom') or 'lb'
 return 'lb' if re.search(r'(?<![a-z])(?:lbs?|pounds?)(?![a-z])',u,re.I) else u

def csvout(path,columns,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def ascii(s):return str(s).translate(str.maketrans({'–':'-','—':'-','’':"'",'×':'x','“':'"','”':'"'}))
def text(c,s,x,y,size=10,bold=False):
 c.setFillColor(colors.HexColor('#172b39'));c.setFont('Helvetica-Bold' if bold else 'Helvetica',size);c.drawString(x,y,ascii(s))
def para(c,s,x,top,w,size=10,bold=False):
 p=Paragraph(escape(ascii(s)),ParagraphStyle('x',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,leading=size+2))
 _,h=p.wrap(w,800);p.drawOn(c,x,top-h);return h

products=sorted(S['products'],key=lambda p:(category(p),p['name'],p['id']))
by=defaultdict(list)
for p in products:by[category(p)].append(p)
pages=[];rows=[];register=[]
for area,ps in by.items():
 for i in range(0,len(ps),2):
  sid=f'S{len(pages)+1:03d}'
  pages.append((sid,area,ps[i:i+2]))
  register.append(dict(sheet_id=sid,suggested_group=area,actual_area='',count_date='',counter='',start_time='',end_time='',returned_complete='',notes=''))
  for p in ps[i:i+2]:
   for n in range(1,5):
    r={k:'' for k in COUNT_COLUMNS}
    r.update(sheet_id=sid,row_id=f'{sid}-P{p["id"]}-{n}',product_id=p['id'],SKU=p.get('odoo_code') or '',name=p['name'],row_kind='UNIDENTIFIED' if n==4 else 'COUNT',unit=unit(p))
    rows.append(r)
# Generic catch-all rows capture physical things not in the catalog, including labels.
for i in range(10):
 r={k:'' for k in COUNT_COLUMNS};r.update(sheet_id='U001',row_id=f'U001-{i+1:02}',row_kind='UNIDENTIFIED');rows.append(r)
csvout(V/'count-sheet-v3.csv',COUNT_COLUMNS,rows)
csvout(V/'sheet-register-v3.csv',list(register[0]),register+[dict(register[0],sheet_id='U001',suggested_group='Other / not in catalog')])
csvout(V/'moved-during-count-v3.csv',MOVE_COLUMNS,[{k:'' for k in MOVE_COLUMNS} for _ in range(10)])
coverage=[dict(product_id=p['id'],SKU=p.get('odoo_code') or '',name=p['name'],all_locations_searched='',completed_at='',owner_initials='',notes='') for p in products]
csvout(V/'count-coverage-v3.csv',list(coverage[0]),coverage)
# Office-only catalog list, never handed to counters as a balance report.
cat=[];counts=[]
for group,ps in by.items():
 ids={p['id'] for p in ps};ls=[l for l in S['lots'] if l['product_id'] in ids]
 positive=set();active=set()
 for l in ls:
  bal=sum(D(str(x['quantity_lb'])) for x in S['lines'] if x['lot_id']==l['id'] and x['effective_status']=='posted')
  if bal:positive.add(l['id'])
  if l['status']=='active':active.add(l['id'])
 counts.append([group,len(ps),sum(p['active'] for p in ps),len(ls),len(active),len(positive)])
 for p in ps:
  lots=[l for l in ls if l['product_id']==p['id']]
  cat.append(dict(category=group,product_id=p['id'],SKU=p.get('odoo_code'),name=p['name'],catalog_type=p['type'],catalog_uom=p.get('uom'),active=p['active'],service_flag=p.get('is_service'),lots=len(lots),lot_codes='; '.join(l['lot_code'] for l in lots),posted_lines=sum(x['product_id']==p['id'] and x['effective_status']=='posted' for x in S['lines'])))
csvout(V/'scope-products-and-lots.csv',list(cat[0]),cat)
def table(headers,rs):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v).replace('|','/') for v in r)+' |' for r in rs])
report=f'''# All-inventory scope check - v3

**Plain-English summary:** Count all physical inventory, by item, printed lot and actual location, while production continues. FL has **{len(products)} catalog products and {len(S['lots'])} lot records** in this read-only snapshot. Packaging is present in the catalog, but **30 of its 32 products have no lots or posted movement history**. Count what is found. Packaging is information only and excluded from the reset; Pallets 102 and Pallet Charge 176 are excluded billing items. Reset scope is **178 products**; the **32 packaging/billing products** have a separate information-only report. Missing FL records for reset-scope stock are holds, never permission to create lots or discard stock. This v3 count replaces the October 5 ingredient-total proposal and the v2 frozen-count instructions.

Snapshot: {S['snapshot_at']}. Fixed query `queries/01-scope.sql`, `BEGIN; SET TRANSACTION READ ONLY; ... ROLLBACK;`, port 5432; rollback confirmed. No writes/API calls. The category groupings below are suggested work groups, **not a verified physical location map**. Arturo fills actual Area and Location on every sheet/row, duplicating blank sheets for additional areas with new sheet and row IDs.

{table(['Suggested work group','Products','Active products','All lots','Active lots','Nonzero lots'],counts)}

There are 78 ingredients (including raw coconut and one container-based WIP item), 25 batches (21 granola, four coconut), 75 finished products (65 active, ten inactive), 32 packaging products and **zero consumable products**. Every catalog item appears in the count packet; the former exclusions 171/209 are superseded for physical counting by the owner's ALL-inventory scope. Within reset scope, inactive, service, unresolved-unit and unmatched-lot adjustments remain held. Packaging never generates reset adjustments or holds. No catalog status changes were made.

## Packaging: which records exist?

{table(['ID / SKU','Product','Catalog unit','Lots / printed codes','Posted lines'],[[str(p['product_id'])+' / '+str(p['SKU']),p['name'],p['catalog_uom'],str(p['lots'])+' / '+p['lot_codes'],p['posted_lines']] for p in cat if p['catalog_type']=='packaging'])}

Packaging values use unit/each counts, not pounds, even though the ledger column is named `quantity_lb`. The owner classifies Product 102 Pallets and 176 Pallet Charge as billing items: both are excluded from the reset regardless of catalog type. Count each physical pallet once for information; do not count a billing charge as an additional pallet. Bags, boxes, tape, cases and labels are likewise information only because FL does not deduct packaging on pack. No lot is created or reset for packaging, including products with no existing lots.

## Physical inventory that FL does not establish

No standalone label product, consumable product, location/bin master, container tare table or location-level balance was found. Bag/box names are catalog descriptions, not proof of what is actually in each room. `lots.found_location` is an occasional discovery note and `trace_events.biz_location` is a broad plant trace label; neither supplies current stock by shelf or area. Label rolls, film, adhesives, cleaning materials, miscellaneous supplies, unidentified WIP and any other unlisted stock must be captured on Other / not-in-catalog sheets. Their actual existence and quantities cannot be determined from a database alone; Arturo's walk-through establishes them. No unlisted material is assumed zero.

The complete identity inventory is in `scope-products-and-lots.csv` (office use). All historical/merged lot identities remain visible there; counters write the lot physically printed, with no FL quantities or suggested lot defaults on the floor sheet.
'''
(V/'scope-check.md').write_text(report)
README='''# Live physical count - Arturo and helpers (v3)

**Count everything you physically find, one area at a time, while production continues. Write the item, printed lot, quantity and location. The office will compare the count to FL afterward. Use v3 only; the October 5 ingredient-total plan and older freeze instructions are superseded.**

**October 6 owner decision:** Keep counting packaging (bags, boxes, tape, cases and labels) for information. Packaging and billing items Pallets 102 / Pallet Charge 176 will not be reset. Count physical pallets once; a billing charge is not another pallet. The office keeps these counts in a separate information-only report. The issued blind floor sheets remain valid.

1. **Start an area.** Write Area, date, Counter and actual Start time before counting; write End time when finished. On every row write time counted (HH:MM), exact printed lot, quantity and unit, Location and Bin ID if marked. If another area needs the same item, copy a blank sheet and assign a new sheet/row ID. Do not invent official location names.
2. **Record every movement promptly.** Every make, pack, shipment, receipt, transfer, adjustment or other movement during count days must be entered in FL as it happens, or with its true occurred time. No end-of-day batch entry. If FL cannot record a location-only move, log it immediately on “Moved during count”; never invent a stock adjustment for an internal move.
3. **Tag counted stock.** Mark each counted pallet/container COUNTED with sheet/row and time. For any move between a counted and uncounted area, in either direction, log item, lot, quantity/unit, from/to, time and tag ID. Production pulls still need their normal FL movement with the true time. The transfer log supplements FL; it does not replace it.
4. **Avoid doubles and omissions.** Try to count every location of an item in the same session. Each area's rows keep their own times. Check tags before recounting a pallet. If there is no tag and you are unsure, count it and write “possible duplicate”. The office will hold it for review.
5. **Use the right units.** Ingredients/bulk: net lb. Without a scale weight, tick Estimated; opened bags/pails are estimated to the nearest 5 lb. Subtract known tare only; note unknown tare. Finished goods: whole cases plus loose packs as “X of Y”. Packaging: pieces/units, with package conversion in notes. Container WIP: count containers, and note weight separately if measured.
6. **Unreadable or missing lot?** Use an Unidentified lot row: leave lot blank, write quantity/unit, description and location. Do not guess or create a lot. Use the Other sheet for items absent from the catalog, including labels and supplies. Blank rows do not mean zero.
7. **Sunshine pouches are ordinary finished-goods counts:** SKU, physical lot, quantity and location. Only copy a physical hold/sold/customer tag if present. Do not decide ownership or invoicing on the floor. Record each physical stack once.
8. **Finish and hand in.** Photograph every page and movement log, then transcribe without overwriting originals. Return the sheet register. Mark a product's coverage complete only after all its locations have been searched; record actual completion time. The owner resolves movement timing, duplicates and missing identities before approving any adjustment. Production need not wait for the whole review. No fixed waiting period after count: prepare the reset as soon as reviewed.

Print `v3/count-sheet-v3.pdf` landscape, actual size. It contains these instructions, area-grouped blank count sheets, an Other sheet and a movement log. Use the matching `v3/count-sheet-v3.csv`, `sheet-register-v3.csv`, `count-coverage-v3.csv` and `moved-during-count-v3.csv`. No FL balances appear on the floor packet. Keep office scope/preview reports away from counters.
'''
(BASE/'README.md').write_text(README)
# A single print packet: one-page instructions, area-grouped sheets, catch-all and moves.
w,h=landscape(letter);c=canvas.Canvas(str(V/'count-sheet-v3.pdf'),pagesize=(w,h),invariant=1)
c.setTitle('All inventory - live blind count v3');c.setAuthor('CNS')
page_count=1+len(pages)+2

def footer(n):
 text(c,'CNS | BLIND COUNT V3 | No FL balances | America/New_York',24,17,9)
 c.setFont('Helvetica',9);c.drawRightString(w-24,17,f'{n} / {page_count}');c.showPage()
text(c,'LIVE PHYSICAL COUNT',24,h-35,23,True)
text(c,'Arturo and helpers - production continues',24,h-57,13)
y=h-78
for para_text in [line for line in README.splitlines() if re.match(r'^[1-8]\.',line)]:
 para_text=para_text.replace('**','').replace('`','')
 y-=para(c,para_text,24,y,w-48,11)+11
assert y>30,y
footer(1)
colwidths=[74,78,55,67,124,65,62,50,169]
headers=['Row / type','Lot as printed','Qty','Unit / partials','Location','Bin / tag','Time HH:MM','Est.','Notes / physical marker only']
assert sum(colwidths)==w-48

def header(sid,group):
 text(c,'BLIND COUNT V3 | '+sid,24,h-27,16,True)
 text(c,'Suggested work group: '+group,24,h-47,11)
 text(c,'Actual Area: _________________________  Counter: ____________________  Date: ___________',24,h-70,12)
 text(c,'Start time: ____________   End time: ____________   All times: actual New York clock time',24,h-92,11)
 text(c,'Write printed lots; no defaults. For another area, use a new sheet ID. Tag counted stock with row/time.',24,h-111,10)

def draw_rows(rs,top):
 x=24
 for label,ww in zip(headers,colwidths):
  c.setFillColor(colors.HexColor('#e8eef2'));c.rect(x,top-30,ww,30,fill=1,stroke=0);para(c,label,x+4,top-4,ww-8,9,True);x+=ww
 y=top-30
 for r in rs:
  x=24
  vals=[r['row_id']+(' UNID' if r['row_kind']=='UNIDENTIFIED' else ''),'','','','','','','','']
  for j,(value,ww) in enumerate(zip(vals,colwidths)):
   c.setStrokeColor(colors.HexColor('#8c9aa3'));c.rect(x,y-35,ww,35,stroke=1,fill=0)
   if value:para(c,value,x+3,y-3,ww-6,8)
   if j==7:c.rect(x+20,y-20,10,10,stroke=1,fill=0)
   x+=ww
  y-=35
 return y
for n,(sid,group,ps) in enumerate(pages,2):
 header(sid,group);top=h-135
 for p in ps:
  label=f'ID {p["id"]} | SKU {p.get("odoo_code") or "(none)"} | {p["name"]} | Count unit: {unit(p)}'
  used=para(c,label,24,top,w-48,12,True);top-=max(used,28)+3
  top=draw_rows([r for r in rows if r['sheet_id']==sid and r['product_id']==p['id']],top)-18
 assert top>29,(sid,top)
 footer(n)
header('U001','Other / not in catalog - write item description in Notes')
unid=[r for r in rows if r['sheet_id']=='U001'][:10]
draw_rows(unid,h-136);footer(page_count-1)
text(c,'MOVED DURING COUNT',24,h-33,22,True)
text(c,'Date: __________   Recorder: __________________   Enter FL movements with true times as well.',24,h-61,11)
text(c,'Crossing counted / uncounted areas in either direction: log immediately. No guessing and no double counting.',24,h-83,10)
cols=[140,86,115,115,70,80,138];labels=['Item / SKU / lot','Qty / unit','From area / location','To area / location','Time','Tag / move ID','FL TX / notes']
top=h-104;x=24
for label,ww in zip(labels,cols):
 para(c,label,x+4,top-5,ww-8,10,True);x+=ww
for i in range(10):
 x=24;y=top-34-i*40
 for ww in cols:c.rect(x,y-40,ww,40);x+=ww
footer(page_count)
c.save()
(V/'count-sheet-v3-pagination.json').write_text(json.dumps({'pages':page_count,'sheets':[{'page':n,'sheet_id':s,'suggested_group':g,'product_ids':[p['id'] for p in ps]} for n,(s,g,ps) in enumerate(pages,2)],'all_products':len(products),'blank_rows':len(rows),'snapshot_at':S['snapshot_at']},indent=2,default=str)+'\n')
print(json.dumps({'pages':page_count,'products':len(products),'rows':len(rows),'category_counts':counts}))
