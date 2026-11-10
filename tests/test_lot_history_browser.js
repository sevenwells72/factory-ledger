// Included by the default node --test tests/*.js suite. Real HTTP and local
// PostgreSQL, never route stubs. Run after pytest, not concurrently against its DB.
const test=require('node:test');
const assert=require('node:assert/strict');
const {execFile}=require('node:child_process');
const {promisify}=require('node:util');
const path=require('node:path');
test('lot-history browser sends API keys over real HTTP and renders complete reports', {timeout:180000}, async(t)=>{
  const {stdout}=await promisify(execFile)(process.execPath,['tests/visual/run-lot-history.mjs'],{
    cwd:path.resolve(__dirname,'..'),timeout:170000,maxBuffer:2*1024*1024
  });
  assert.match(stdout,/PASS: \d+ actual HTTP requests carried X-API-Key/);
  t.diagnostic(stdout.trim());
});
