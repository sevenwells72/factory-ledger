#!/usr/bin/env python3
"""Time candidate audit reports from this Mac using transaction-scoped read-only SQL.

No app lifespan, schema setup, writes, session-level settings or deployment.
Credentials are read from the existing local config and never printed.
"""
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['DATABASE_URL']='postgresql://localhost:5432/factory_ledger_test'
import main
import psycopg2
from psycopg2.extras import RealDictCursor


def run():
    out=ROOT/'work'/'review-production'
    out.mkdir(parents=True,exist_ok=True)
    conn=psycopg2.connect(Path.home().joinpath('.config/factory-ledger/db_url').read_text().strip(),port=5432)
    conn.autocommit=True
    counts=[]
    class Counted:
        def __init__(self,cur):self.cur=cur;self.count=0
        def execute(self,*args,**kwargs):self.count+=1;return self.cur.execute(*args,**kwargs)
        def fetchall(self):return self.cur.fetchall()
        def fetchone(self):return self.cur.fetchone()
    @contextmanager
    def readonly():
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('BEGIN');cur.execute('SET TRANSACTION READ ONLY')
            cur.execute('SHOW transaction_read_only')
            assert cur.fetchone()['transaction_read_only']=='on'
            counted=Counted(cur)
            try:yield counted
            finally:counts.append(counted.count);cur.execute('ROLLBACK')
    main.get_transaction=readonly
    results=[]
    try:
        for code,pid,direction in [('25120',13,'forward'),('26-02-03-FOUND-016',39,'forward'),('26-10-22-QUAL-001',32,'forward'),('BB110927',145,'backward')]:
            durations=[]
            for _ in range(3 if code in ('25120','26-02-03-FOUND-016') else 1):
                start=time.perf_counter();r=main.dashboard_api_lot_report(code,pid,direction)
                encoded=json.dumps(r,default=str);durations.append(round(time.perf_counter()-start,3))
            g=r['forward_trace']
            row={'lot':code,'product_id':pid,'seconds':durations,'queries':counts[-1],
                 'lots':len(g['lots']),'edges':len(g['consumption']),
                 'shipments':sum(len(n['shipments']) for n in g['lots']),'bytes':len(encoded)}
            if direction=='forward':assert row['queries']==6
            if code=='26-10-22-QUAL-001':
                makes={e['transaction_id'] for e in g['consumption'] if e['source_lot_id']==g['start_lot_id'] and e['type']=='make'}
                assert (len(makes),row['lots'],row['shipments'])==(17,20,3)
                with readonly() as cur:
                    cur.execute("""SELECT DISTINCT t.id FROM ledger_current_transactions t
                                   JOIN ledger_current_transaction_lines i ON i.transaction_id=t.id
                                   WHERE t.effective_status='posted' AND t.type='make' AND i.lot_id=1637 AND i.quantity_lb<0""")
                    assert makes=={x['id'] for x in cur.fetchall()}
            if code=='BB110927':
                root=r['input_tree'];batches=root['transactions'][0]['inputs']
                assert len(batches)==4 and root['output_lb']==3855 and r['on_hand_lb']==3855
                assert [len(t['inputs']) for b in batches for t in b['origin']['transactions']]==[7,7,8,7,8]
                with readonly() as cur:
                    cur.execute("""SELECT i.product_id,i.lot_id,-SUM(i.quantity_lb) AS quantity
                                   FROM ledger_current_transaction_lines i JOIN ledger_current_transactions t ON t.id=i.transaction_id
                                   WHERE t.id=2947 AND t.effective_status='posted' AND i.quantity_lb<0
                                   GROUP BY i.product_id,i.lot_id""")
                    actual={(x['product_id'],x['lot_id']):float(x['quantity']) for x in cur.fetchall()}
                    t=next(t for b in batches for t in b['origin']['transactions'] if t['transaction_id']==2947)
                    assert actual=={(x['product_id'],x['lot_id']):x['quantity'] for x in t['inputs']}
            (out/(code+'.json')).write_text(encoded)
            results.append(row);print(json.dumps(row),flush=True)
        (out/'timings.json').write_text(json.dumps(results,indent=2))
        assert all(max(r['seconds'])<5 for r in results[:2]),'Large forward report exceeded five seconds'
    finally:conn.close()

if __name__=='__main__':run()
