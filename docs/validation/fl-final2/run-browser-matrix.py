#!/usr/bin/env python3
"""Run all Assistant browser runners, full cases and both session modes."""
import json, os, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path('/private/tmp/fl_final2/browser');OUT.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,PLAYWRIGHT_MODULE='/Users/michaelgross/Documents/factory-ledger/node_modules/playwright/index.mjs',
 FIXB_PYTHON='/Users/michaelgross/dev/fl_final_venv/bin/python',FIXC_PYTHON='/Users/michaelgross/dev/fl_final_venv/bin/python',
 TEST_DB_PREFIX='fl_final2_',PGHOST='127.0.0.1',PGPORT='5432',FIXC_ENGINES='chromium,webkit',PYTHONDONTWRITEBYTECODE='1')
for key in ['DATABASE_URL','TEST_DATABASE_URL','PGOPTIONS','PGHOSTADDR','PGSERVICE','PGSERVICEFILE','PILOT_SESSIONS']:
 env.pop(key,None)
plan=[]
for mode in [0,1]:
 for name,runner,outvar in [('existing','run-fl-assistant.mjs','F1_VISUAL_OUT'),('round-a','run-fl-assistant-round-a.mjs','FIX_A_VISUAL_OUT')]:
  label=f'{name}-{mode}';plan.append((label,runner,{'PILOT_SESSIONS':str(mode),outvar:str(OUT/label)}))
for name,runner,outvar in [
 ('round-b-contract','run-assistant-round-b.mjs','FIXB_BROWSER_OUT'),
 ('round-b-page','run-fl-assistant-round-b-page.mjs','FIXB_PAGE_OUT'),
 ('round-c-transcripts','run-fl-assistant-round-c.mjs','FIXC_OUT'),
 ('round-c-phone','run-fl-assistant-round-c-phone.mjs','FIXC_PHONE_OUT'),
 ('mobile-ux','run-fl-assistant-mobile-ux.mjs','MOBILE_UX_OUT'),
 ('final-floor','run-fl-assistant-final-floor.mjs','FL_FINAL_UI_OUT')]:
 plan.append((name,runner,{outvar:str(OUT/(name+'.json' if name=='round-b-contract' else name))}))
results=[]
for name,runner,extra in plan:
 started=time.monotonic()
 with (OUT/(name+'.log')).open('w') as log:
  p=subprocess.run(['node','tests/visual/'+runner],cwd=ROOT,env={**env,**extra},stdout=log,stderr=subprocess.STDOUT)
 row={'name':name,'runner':runner,'returncode':p.returncode,'seconds':round(time.monotonic()-started,2),'log':name+'.log'}
 results.append(row);(OUT/'matrix.json').write_text(json.dumps(results,indent=2)+'\n')
 print(name,p.returncode,flush=True)
raise SystemExit(any(r['returncode'] for r in results))
