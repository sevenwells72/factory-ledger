// Cutover-off supersedes the former new-form personal-key fallback.
// This runner asserts production appearance and legacy behavior against real local FL.
process.env.D12_BASE ||= 'http://127.0.0.1:8125';
process.env.D12_GATE_ON = '0';
await import('./run-d12-gating.mjs');
