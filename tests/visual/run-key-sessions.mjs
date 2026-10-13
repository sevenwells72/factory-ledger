import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {startStaticServer} from './lib/server.mjs';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const server=await startStaticServer(root);
const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL || 'chrome'});
const out=path.join(root,'.venv-test/key-session-visual');await fs.mkdir(out,{recursive:true});
try {
  for (const enabled of [false,true]) {
    const context=await browser.newContext();
    await context.addInitScript(()=>{window.FL_SESSION_API_BASE=location.origin});
    const page=await context.newPage(), errors=[], writes=[];let exchanges=0,logouts=0,expired=false;
    page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/auth/session/config',r=>r.fulfill({json:{key_sessions_enabled:enabled}}));
    await page.route('**/auth/session',r=>{assert.equal(r.request().postDataJSON().actor_key,'fixture-personal');exchanges++;
      return r.fulfill({json:{session_token:'fls_fixture',key_kind:'session',actor:{id:4,name:'Arturo',role:'floor'}}})});
    await page.route('**/auth/whoami',r=>r.fulfill({json:{key_kind:'session',actor:{id:4,name:'Arturo',role:'floor'}}}));
    await page.route('**/auth/session/logout',r=>{logouts++;return r.fulfill({json:{signed_out:true}})});
    await page.route('**/production/runs**',r=>{
      if(r.request().method()!=='GET') {writes.push(r.request().headers()['x-api-key']);
        if(expired)return r.fulfill({status:401,json:{detail:{error_code:'SESSION_EXPIRED'}}});
      }
      return r.fulfill({json:{runs:[],success:true}});
    });
    await page.goto(server.origin+'/dashboard/runs.html');
    await page.locator('#week-board').waitFor({state:'visible'});
    if(enabled) {
      await page.getByRole('button',{name:'Sign in',exact:true}).click();
      await page.locator('.fl-session-dialog input').fill('fixture-personal');
      await page.locator('.fl-session-dialog button[type=submit]').click();
      await page.getByText('Signed in as Arturo',{exact:true}).waitFor();
      assert.equal(await page.locator('.fl-session-dialog input').inputValue(),'');
      assert.equal(await page.evaluate(()=>JSON.stringify(sessionStorage).includes('fixture-personal')),false);
    } else assert.equal(await page.locator('.fl-session-bar').count(),0);
    await page.evaluate(()=>FL.fetchWithTimeout('/production/runs',{method:'POST',headers:{'X-API-Key':'shared-fixture'},body:'{}'}));
    assert.equal(writes.at(-1),enabled?'fls_fixture':'shared-fixture');
    if(enabled) {
      await page.reload();await page.getByText('Signed in as Arturo',{exact:true}).waitFor();assert.equal(exchanges,1);
      for(const width of [1440,390]) {
        await page.setViewportSize({width,height:844});
        assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
        await page.screenshot({path:path.join(out,'dashboard-signed-in-'+width+'.png')});
      }
      expired=true;
      await page.evaluate(()=>FL.fetchWithTimeout('/production/runs',{method:'POST',headers:{'X-API-Key':'shared-fixture'},body:'{}'}));
      await page.getByRole('button',{name:'Sign in',exact:true}).waitFor({state:'visible'});
      assert.equal(await page.evaluate(()=>sessionStorage.getItem('fl-key-session-token')),null);
      expired=false;await page.getByRole('button',{name:'Sign in',exact:true}).click();
      await page.locator('.fl-session-dialog input').fill('fixture-personal');await page.locator('.fl-session-dialog button[type=submit]').click();
      await page.getByText('Signed in as Arturo',{exact:true}).waitFor();
      await page.getByRole('button',{name:'Sign out',exact:true}).click();
      await page.getByRole('button',{name:'Sign in',exact:true}).waitFor({state:'visible'});assert.equal(logouts,1);
    }
    assert.deepEqual(errors,[]);await context.close();
  }
  console.log('Dashboard browser: flag-off parity, token-only writes, reload, person banner, expiry re-ask, logout and desktop/mobile layout passed.');
} finally {await browser.close();await server.close();}
