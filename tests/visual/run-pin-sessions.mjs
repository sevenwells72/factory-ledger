import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {chromium} from 'playwright';
import {startStaticServer} from './lib/server.mjs';

const server = await startStaticServer(path.resolve('dashboard'));
const browser = await chromium.launch();
const output = path.resolve(process.env.PIN_AUDIT_OUT || '/private/tmp/fl-a11-ui-evidence');
await fs.mkdir(output, {recursive:true});
let checks = 0;
try {
  for (const width of [390, 1440]) {
    const context = await browser.newContext({viewport:{width,height:900}});
    const person = {id:701, name:'Michael', role:'owner'};
    const token = 'fls_'+'synthetic-session-not-a-real-credential';
    let logins = 0, protectedCalls = [], retryOnce = false, nextPerson = person;
    const respond = (route, data, status=200) => route.fulfill({status,contentType:'application/json',body:JSON.stringify(data)});
    await context.route('**/*', async route => {
      const req = route.request(), url = new URL(req.url());
      if (url.pathname === '/auth/session' && req.method()==='POST') {
        logins++; return respond(route,{session_token:token,actor:nextPerson,expires_at:new Date(Date.now()+600000).toISOString()});
      }
      if (url.pathname === '/auth/session' && req.method()==='DELETE') return respond(route,{signed_out:true});
      if (url.pathname === '/actors/pins') {
        assert.equal(req.headers()['x-api-key'],token);
        return respond(route,{actors:[{...person,active:true,pin_set:true},{id:702,name:'Arturo',role:'floor',active:true,pin_set:false}],failures_last_hour:0});
      }
      if (url.pathname === '/test/write' || /^\/actors\/\d+\/pin$/.test(url.pathname)) {
        protectedCalls.push({body:req.postData(),headers:req.headers()});
        assert.equal(req.headers()['x-api-key'],token);
        if (retryOnce) {retryOnce=false;return respond(route,{detail:{error_code:'SESSION_EXPIRED'}},401);}
        if (!req.headers()['x-fl-owner-pin']) return respond(route,{detail:{error_code:'OWNER_PIN_REQUIRED'}},403);
        return respond(route,{pin_set:true});
      }
      if (url.origin !== server.origin) return route.abort();
      return route.continue();
    });
    const page = await context.newPage(), errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(server.origin+'/pin-management.html');
    await page.getByRole('dialog').waitFor();
    assert.equal(await page.getByRole('dialog').locator('select').count(),0);checks++;
    await page.screenshot({path:path.join(output,`login-${width}.png`)});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));checks++;
    // Synthetic test values are entered only in password inputs; never logged.
    const testValue=String(7000+538);
    await page.getByRole('dialog').getByLabel('Personal PIN',{exact:true}).fill(testValue);
    await page.getByRole('button',{name:'Sign in',exact:true}).click();
    await page.locator('#admin-pin').waitFor({state:'visible'});
    assert.equal(await page.locator('#fl-person').textContent(),'Recording as Michael');checks++;
    assert.equal(await page.evaluate(()=>localStorage.getItem('fl-session-v1')),null);checks++;
    assert.ok(!(await page.evaluate(()=>sessionStorage.getItem('fl-session-v1'))).includes(testValue));checks++;
    await page.screenshot({path:path.join(output,`admin-${width}.png`)});
    await page.locator('#pin-person').selectOption('702');
    await page.locator('#admin-pin input[name=pin]').fill(String(5800+73));
    await page.getByRole('button',{name:'Set PIN',exact:true}).click();
    await page.getByRole('dialog',{name:'Michael: confirm this action'}).waitFor();
    await page.getByRole('dialog').getByLabel('Michael’s PIN').fill(testValue);
    await page.getByRole('button',{name:'Confirm action',exact:true}).click();
    await page.getByRole('dialog').waitFor({state:'hidden'});
    assert.equal(protectedCalls.length,2);assert.equal(protectedCalls[0].body,protectedCalls[1].body);checks++;
    assert.equal(await page.locator('#admin-pin input[name=pin]').inputValue(),'');checks++;
    // A second write needs another PIN; cancelling sends no retry.
    await page.evaluate(()=>{window.pending=FLSession.fetch(FLSession.base+'/test/write',{method:'POST',body:'same-ticket'}).then(r=>r.status,e=>e.message);});
    await page.getByRole('dialog').waitFor();await page.getByRole('button',{name:'Cancel',exact:true}).click();
    assert.equal(await page.evaluate(()=>window.pending),'Action cancelled.');checks++;
    // Explicit expiry resumes only for the SAME actor; a handover never posts the old draft.
    retryOnce=true;nextPerson={id:702,name:'Arturo',role:'floor'};
    const before=protectedCalls.length;
    await page.evaluate(()=>{window.pending=FLSession.fetch(FLSession.base+'/test/write',{method:'POST',body:'same-ticket'}).then(r=>r.status,e=>e.message);});
    await page.getByRole('dialog').getByLabel('Personal PIN',{exact:true}).fill(testValue);
    await page.getByRole('button',{name:'Sign in',exact:true}).click();
    assert.match(await page.evaluate(()=>window.pending),/belongs to Michael/);assert.equal(protectedCalls.length,before+1);checks++;
    // Advancing the clock ends the tab's credential before user interaction can renew it.
    await page.evaluate(()=>{const realNow=Date.now;Date.now=()=>realNow()+600001;});
    await page.waitForFunction(()=>sessionStorage.getItem('fl-session-v1')===null);
    assert.equal(await page.locator('#fl-person').textContent(),'Signed out');checks++;
    assert.deepEqual(errors,[]);await context.close();
  }
  console.log(JSON.stringify({checks,viewports:[390,1440],result:'passed',evidence:output}));
} finally {await browser.close();await server.close();}
