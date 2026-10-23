// Optional rendered-background supplement for the unchanged general harness:
// WALLPAPER_RASTER_OUT=/absolute/results.json node --import ./tests/visual/lib/wallpaper-audit-hook.mjs tests/visual/run-visual-audit.mjs ...
import fs from 'node:fs/promises';
import path from 'node:path';
import {chromium} from 'playwright';
import {renderedContrast} from './rendered-contrast.mjs';
const output=process.env.WALLPAPER_RASTER_OUT;
if(output){
  const records=[];let pending=Promise.resolve();
  await fs.mkdir(path.dirname(output),{recursive:true});
  const launch=chromium.launch.bind(chromium);
  chromium.launch=async(...args)=>{
    const browser=await launch(...args),newContext=browser.newContext.bind(browser);
    browser.newContext=async(...args)=>{
      const context=await newContext(...args);
      context.on('page',page=>{
        const screenshot=page.screenshot.bind(page);let measuring=false;
        page.screenshot=async(options={})=>{
          if(!measuring&&options.path){
            measuring=true;
            try{
              const contrast=await renderedContrast(page,{viewportOnly:true});
              records.push({screenshot:options.path,url:page.url(),...contrast});
              const data=JSON.stringify(records,null,2)+'\n';
              pending=pending.then(()=>fs.writeFile(output,data));await pending;
            }finally{measuring=false;}
          }
          return screenshot(options);
        };
      });
      return context;
    };
    return browser;
  };
}
