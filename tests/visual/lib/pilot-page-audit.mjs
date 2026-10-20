// Observe the real pilot walkthrough screenshots; never substitutes app/API data.
// Run with PILOT_PAGE_AUDIT_OUT=<directory> node --import ./tests/visual/lib/pilot-page-audit.mjs tests/visual/run-pilot-ux-round.mjs
import fs from 'node:fs/promises';
import path from 'node:path';
import { CHECK_SOURCE } from './checks.mjs';
const output=process.env.PILOT_PAGE_AUDIT_OUT;
if(output){
  const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
  const report={scope:'Record entry, Office, Receipt, Exceptions and receipt-based Today on actual local pages',captures:[]};
  await fs.mkdir(output,{recursive:true});
  const launch=chromium.launch.bind(chromium);
  chromium.launch=async(...args)=>{
    const browser=await launch(...args),newContext=browser.newContext.bind(browser);
    browser.newContext=async(...args)=>{
      const context=await newContext(...args);
      context.on('page',page=>{
        const screenshot=page.screenshot.bind(page);
        page.screenshot=async(options={})=>{
          const url=new URL(page.url());
          if(/\/(entry|office|receipt|exceptions)\.html$/.test(url.pathname)){
            await page.evaluate(CHECK_SOURCE);
            const measures=await page.evaluate(()=>({
              title:document.querySelector('main h1')?.textContent,
              language:document.documentElement.lang,width:innerWidth,
              touch:window.__FL_AUDIT.touchTargets(),contrast:window.__FL_AUDIT.contrast(),
              overflow:window.__FL_AUDIT.horizontalOverflow(),
            }));
            report.captures.push({page:url.pathname,view:url.searchParams.get('view'),screenshot:options.path,...measures});
            await fs.writeFile(path.join(output,'pilot-page-audit.json'),JSON.stringify(report,null,2));
          }
          return screenshot(options);
        };
      });
      return context;
    };
    return browser;
  };
}
