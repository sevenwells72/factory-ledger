// Shared run accounting: setup/sign-in errors and skipped planned checks fail the run.
export const pilotCheckPlan = ({ mode }) => mode ? [
  1, 2, '9.office', '3-5.customer', 6, 'luz.1', 'luz.2', '7+luz.3', '8+luz.4',
  '15-16', '5.entry', '9.entry', '10+arturo.receive', '11-12+arturo.make',
  '13+arturo.pack', '10+arturo.ship', '14+arturo.found', '14+arturo.adjust',
  'arturo.void', '8+arturo.today', 17, 'integrity'
] : [1, 'off'];

const key = ({ mode, width, lang }) => `${mode}/${width}/${lang}`;
export function createWalkthroughReporter(report, scenarios, plan, { save = async () => {}, log = message => console.error(message) } = {}) {
  report.scenarios = scenarios.map(scenario => ({ ...scenario, planned_checks: plan(scenario).map(String), completed_checks: [], started: false, complete: false, passed: false }));
  const find = scenario => {
    const row = report.scenarios.find(row => key(row) === key(scenario));
    if (!row) throw Error(`Unplanned walkthrough: ${key(scenario)}`);
    return row;
  };
  function recordCheck(scenario, check) {
    const row = find(scenario), item = String(check.item);
    if (!row.planned_checks.includes(item)) throw Error(`Unplanned check ${item} in walkthrough ${key(scenario)}`);
    if (row.completed_checks.includes(item)) throw Error(`Duplicate check ${item} in walkthrough ${key(scenario)}`);
    row.completed_checks.push(item);
    report.results.push(check);
  }
  async function runScenario(scenario, body) {
    const row = find(scenario);
    if (row.started) throw Error(`Walkthrough already started: ${key(scenario)}`);
    row.started = true;
    try { await body(); }
    catch (error) { row.error = error.stack || String(error); }
    finally {
      row.missing_checks = row.planned_checks.filter(item => !row.completed_checks.includes(item));
      row.complete = !row.missing_checks.length;
      const checks = report.results.filter(check => check.sessions === scenario.mode && check.width === scenario.width && check.language === scenario.lang);
      row.failed_checks = checks.filter(check => !check.passed).length;
      row.passed = row.complete && !row.error && !row.failed_checks;
      if (!row.passed) log(`Walkthrough ${key(scenario)} FAILED: ${row.error || 'one or more checks did not pass'}${row.missing_checks.length ? `; incomplete: missing checks ${row.missing_checks.join(', ')}` : ''}`);
      await save();
    }
  }
  async function finalize() {
    for (const row of report.scenarios.filter(row => !row.started)) {
      row.error = 'Walkthrough never started';
      row.missing_checks = row.planned_checks;
      log(`Walkthrough ${key(row)} FAILED: ${row.error}`);
    }
    const failedChecks = report.results.filter(row => !row.passed).length;
    // Do not count a wrapped assertion twice, but always count unwrapped errors/incomplete runs.
    const failedScenarios = report.scenarios.filter(row => !row.passed && !row.failed_checks).length;
    const summary = { checks: report.results.length, passed: report.results.filter(row => row.passed).length,
      failed: failedChecks + failedScenarios, planned_checks: report.scenarios.reduce((total, row) => total + row.planned_checks.length, 0),
      scenarios: report.scenarios.length, completed_scenarios: report.scenarios.filter(row => row.complete).length };
    report.summary = summary;
    await save();
    return summary;
  }
  return { recordCheck, runScenario, finalize };
}
