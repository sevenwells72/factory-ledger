// Included by the default node --test tests/*.js suite. Real local HTTP and a throwaway local
// PostgreSQL DB, never route stubs. Run after pytest, not concurrently against its DB.
const test=require('node:test');
const assert=require('node:assert/strict');
const {execFile}=require('node:child_process');
const {promisify}=require('node:util');
const path=require('node:path');
test('Office and Assistant Dashboard buttons and New order / Close order confirmations all navigate home', {timeout:180000}, async(t)=>{
  const {stdout}=await promisify(execFile)(process.execPath,['tests/visual/run-back-to-dashboard.mjs'],{
    cwd:path.resolve(__dirname,'..'),timeout:170000,maxBuffer:2*1024*1024
  });
  assert.match(stdout,/PASS: Office Dashboard button, Assistant header, New order and Close order "Back to dashboard" all reach \/dashboard\/index\.html/);
  t.diagnostic(stdout.trim());
});
