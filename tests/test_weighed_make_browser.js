// Included by the default node --test tests/*.js suite. Real local HTTP and a throwaway local
// PostgreSQL DB, never route stubs. Run after pytest, not concurrently against its DB.
const test=require('node:test');
const assert=require('node:assert/strict');
const {execFile}=require('node:child_process');
const {promisify}=require('node:util');
const path=require('node:path');
test('Make screen: weighed batch output is booked on the lot, blank keeps pans x pan lb, a typo is refused', {timeout:180000}, async(t)=>{
  const {stdout}=await promisify(execFile)(process.execPath,['tests/visual/run-weighed-make.mjs'],{
    cwd:path.resolve(__dirname,'..'),timeout:170000,maxBuffer:2*1024*1024
  });
  assert.match(stdout,/PASS: Make screen books a weighed 4,365 lb on 12 coconut pans with recipe x 12 ingredients \(MK-\d{6}-\d{3,}\), keeps 360 lb when blank \(MK-\d{6}-\d{3,}\), and refuses a 43,650 lb typo/);
  t.diagnostic(stdout.trim());
});
