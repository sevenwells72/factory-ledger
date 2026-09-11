import fs from 'node:fs/promises';
import vm from 'node:vm';
// Summarize raw captures using the existing runner's exact verdict function.
// No measurements or thresholds are changed. Optional rawPath supports the
// committed canonical JSON even after diagnostic runs replace build output.
const [root, outDir, phase, rawPath] = process.argv.slice(2);
const source = await fs.readFile(root + '/tests/visual/run-visual-audit.mjs', 'utf8');
const verdict = vm.runInNewContext('(' + source.slice(source.indexOf('function verdict('), source.indexOf('\nasync function main()')) + ')');
const raw = JSON.parse(await fs.readFile(rawPath || root + '/docs/design/audit/screenshots/results.json', 'utf8'));
function scopeOf(id) { const n = Number(id.slice(2)); return n >= 24 && n <= 29 ? 'list' : n >= 30 && n <= 40 ? 'detail' : 'other'; }
const scopes = {};
const failures = [];
const cells = [];
for (const capture of raw.results) {
  const scope = scopeOf(capture.id);
  const summary = scopes[scope] ||= { screens: [], captures: 0, perRule: {} };
  if (!summary.screens.includes(capture.id)) summary.screens.push(capture.id);
  summary.captures++;
  for (const rule of raw.rules) {
    const data = capture.rules[rule];
    const result = verdict(rule, data);
    const stats = summary.perRule[rule] ||= { PASS: 0, FAIL: 0, WARN: 0, 'N/A': 0, ERROR: 0, instances: rule === 'LAYOUT-020' ? null : 0 };
    stats[result.status]++;
    if (result.status === 'FAIL' && stats.instances !== null) {
      stats.instances += rule === 'LAYOUT-003/ACCESS-001' ? data.offenderCount
        : rule === 'LAYOUT-011' ? data.coveredCount : data.failures;
    }
    const cell = {id:capture.id,name:capture.name,variant:capture.variant,rule,scope,...result};
    cells.push(cell);
    if (['FAIL','ERROR'].includes(result.status)) failures.push(cell);
  }
}
const summary = {generatedAt:raw.generatedAt,totalCaptures:raw.results.length,instanceCountMeaning:'STATUS, touch and contrast use measured failures; horizontal overflow uses offenderCount; occlusion uses coveredCount; refresh CLS has no single rendered-instance count (null). Counts include failing cells only.',scopes,failures,errors:raw.results.filter(r => r.error).map(({id,variant,error}) => ({id,variant,error})),cells};
await fs.mkdir(outDir,{recursive:true});
await fs.writeFile(`${outDir}/${phase}-summary.json`, JSON.stringify(summary, null, 2));
const lines = [`# Sales Order detail redesign — ${phase} failures`, '', 'Counts are failing screen/variant captures, followed by offending rendered instances in parentheses. List: S-24–S-29; detail: S-30–S-40 excluding native dialogs S-35/S-38.', '', '| Rule | List failures (instances) | Detail failures (instances) | Other failures (instances) |', '|---|---:|---:|---:|'];
for (const rule of raw.rules.filter(r => r.startsWith('STATUS-'))) lines.push(`| ${rule} | ` + ['list','detail','other'].map(scope => {const s=scopes[scope].perRule[rule];return `${s.FAIL} (${s.instances})`;}).join(' | ') + ' |');
lines.push('', `Captures: ${raw.results.length}; runner errors: ${summary.errors.length}.`, '', 'Failing capture/rule pairs across all 13 rules: ' + ['list','detail','other'].map(scope => `${scope} ${failures.filter(f => f.scope === scope && f.status === 'FAIL').length}`).join('; ') + '.', '');
await fs.writeFile(`${outDir}/${phase}-summary.md`,lines.join('\n'));
console.log(lines.join('\n'));
