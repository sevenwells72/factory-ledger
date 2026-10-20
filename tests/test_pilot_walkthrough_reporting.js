const { test } = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const { pathToFileURL } = require('node:url');
const path = require('node:path');
const helper = pathToFileURL(path.join(__dirname, 'visual/lib/pilot-walkthrough-report.mjs')).href;

// Exercise the same scenario wrapper, check recording, final report and exit decision
// used by the real browser runner. The sign-in helper fails after a genuine pass.
test('a mid-run sign-in error fails the walkthrough process and reports every skipped check', () => {
  const source = `import { createWalkthroughReporter } from ${JSON.stringify(helper)};
    const scenario = { mode: 1, width: 390, lang: 'es' }, report = { results: [] };
    const run = createWalkthroughReporter(report, [scenario], () => ['header', 'office', 'floor']);
    await run.runScenario(scenario, async () => {
      run.recordCheck(scenario, { item: 'header', sessions: 1, width: 390, language: 'es', passed: true });
      const login = async () => { throw Error('Sign-in error: forced mid-run timeout'); };
      await login();
      run.recordCheck(scenario, { item: 'office', sessions: 1, width: 390, language: 'es', passed: true });
    });
    const summary = await run.finalize();
    console.log(JSON.stringify({ report, summary }));
    if (summary.failed) process.exitCode = 1;`;
  const child = spawnSync(process.execPath, ['--input-type=module', '-e', source], { encoding: 'utf8' });
  assert.equal(child.status, 1);
  const { report, summary } = JSON.parse(child.stdout);
  assert.equal(summary.passed, 1);
  assert.equal(summary.failed, 1);
  assert.equal(summary.completed_scenarios, 0);
  assert.equal(summary.planned_checks, 3);
  assert.deepEqual(report.scenarios[0].missing_checks, ['office', 'floor']);
  assert.match(child.stderr, /Sign-in error: forced mid-run timeout/);
  assert.match(child.stderr, /incomplete: missing checks office, floor/);
});

test('an early return without an exception still fails incomplete planned coverage', async () => {
  const { createWalkthroughReporter } = await import(helper);
  const scenario = { mode: 0, width: 390, lang: 'en' }, report = { results: [] };
  const run = createWalkthroughReporter(report, [scenario], () => [1, 'off'], { log() {} });
  await run.runScenario(scenario, async () => run.recordCheck(scenario, { item: 1, sessions: 0, width: 390, language: 'en', passed: true }));
  assert.equal((await run.finalize()).failed, 1);
  assert.deepEqual(report.scenarios[0].missing_checks, ['off']);
});

test('all planned checks in all walkthroughs are required for a successful run', async () => {
  const { createWalkthroughReporter } = await import(helper);
  const scenarios = [{ mode: 0, width: 390, lang: 'en' }, { mode: 1, width: 390, lang: 'es' }], report = { results: [] };
  const run = createWalkthroughReporter(report, scenarios, () => ['required'], { log() {} });
  for (const scenario of scenarios) await run.runScenario(scenario, async () => run.recordCheck(scenario, { item: 'required', sessions: scenario.mode, width: scenario.width, language: scenario.lang, passed: true }));
  assert.deepEqual(await run.finalize(), { checks: 2, passed: 2, failed: 0, planned_checks: 2, scenarios: 2, completed_scenarios: 2 });
});

test('a walkthrough that never starts also fails the final report', async () => {
  const { createWalkthroughReporter } = await import(helper);
  const scenario = { mode: 1, width: 390, lang: 'en' }, report = { results: [] };
  const run = createWalkthroughReporter(report, [scenario], () => ['required'], { log() {} });
  assert.equal((await run.finalize()).failed, 1);
  assert.equal(report.scenarios[0].error, 'Walkthrough never started');
});
