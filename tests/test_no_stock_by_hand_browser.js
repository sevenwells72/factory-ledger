const test=require('node:test');
const assert=require('node:assert/strict');
const {execFile}=require('node:child_process');
const {promisify}=require('node:util');
const path=require('node:path');
test('Receiving safeguards: real Receive, Adjust, Found and receipt screens',{timeout:180000},async t=>{
 const {stdout}=await promisify(execFile)(process.execPath,['tests/visual/run-no-stock-by-hand.mjs'],{cwd:path.resolve(__dirname,'..'),timeout:170000,maxBuffer:2*1024*1024});
 assert.match(stdout,/PASS: Receive, Adjust, Found and voided receipt screens/);t.diagnostic(stdout.trim());
});
