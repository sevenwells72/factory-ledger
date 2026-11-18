// Included by the default node --test tests/*.js suite. Real local HTTP and a throwaway local
// PostgreSQL DB (migration 096 applied to that DB only), never route stubs. Run after pytest,
// not concurrently against its DB.
const test=require('node:test');
const assert=require('node:assert/strict');
const {execFile}=require('node:child_process');
const {promisify}=require('node:util');
const path=require('node:path');
test('Adjust form: destinations on a removal, sales order for shipped stock, over 500 lb recorded at once', {timeout:180000}, async(t)=>{
  const {stdout}=await promisify(execFile)(process.execPath,['tests/visual/run-adjust-destinations.mjs'],{
    cwd:path.resolve(__dirname,'..'),timeout:170000,maxBuffer:2*1024*1024
  });
  assert.match(stdout,/PASS: Adjust form offers the five destinations \(plus wrong lot \/ data-entry error with a note\) on a removal, names the sales order for shipped stock, and records 600 lb at once with no photo and no hold \(ADJ-\d{6}-\d{3,}\)/);
  t.diagnostic(stdout.trim());
});
