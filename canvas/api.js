// Read-only API client. Only GET is implemented on purpose.
(function () {
  const cfg = window.FL_CANVAS_CONFIG;
  const base = cfg.apiBase.replace(/\/$/, '');
  let keyPromise = null;

  function key() {
    if (cfg.apiKey) return Promise.resolve(cfg.apiKey);
    if (!keyPromise) {
      keyPromise = fetch(base + '/auth/dashboard-key', { cache: 'no-store' })
        .then((r) => {
          if (!r.ok) throw new Error('dashboard key unavailable (HTTP ' + r.status + ')');
          return r.json();
        })
        .then((j) => j.dashboard_key)
        .catch((e) => { keyPromise = null; throw e; });
    }
    return keyPromise;
  }

  // The API's database pool is small (10 connections); keep this page's share low.
  const MAX_IN_FLIGHT = 4;
  let inFlight = 0;
  const waiting = [];
  function slot() {
    if (inFlight < MAX_IN_FLIGHT) { inFlight++; return Promise.resolve(); }
    return new Promise((resolve) => waiting.push(resolve));
  }
  function release() {
    const next = waiting.shift();
    if (next) next(); else inFlight--;
  }

  async function get(path, params) {
    const url = new URL(base + path);
    for (const [k, v] of Object.entries(params || {})) {
      if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
    }
    const k = await key();
    // Hold the slot until the body is read, so at most 4 requests are ever on the wire.
    await slot();
    try {
      const r = await fetch(url, { method: 'GET', headers: { 'X-API-Key': k } });
      if (!r.ok) {
        const err = new Error(path + ' → HTTP ' + r.status);
        err.status = r.status;
        throw err;
      }
      return await r.json();
    } finally {
      release();
    }
  }

  window.FLApi = { get, base };
})();
