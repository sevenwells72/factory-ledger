/* === Shared fetch wrapper with a timeout (IMP-005) ===
   FEEDBACK-003 (Critical, Hard rule), ERROR-001, ERROR-002.

   Before this file no fetch anywhere in the product had a timeout, an
   AbortController or a signal. If the API accepted a connection and then hung
   — the common cold-container failure — every loading state waited forever,
   and the dashboard's indicator is static text, which the rule treats as a
   freeze by definition.

   Loaded as a plain script (no module system in this product) by every page
   in dashboard/, and exposed as window.FL. Every fetch in dashboard/ goes
   through FL.fetchWithTimeout. */
(function (global) {
  'use strict';

  var DEFAULT_TIMEOUT_MS = 15000;

  /* Thrown when the request exceeded the timeout, so callers can tell a stall
     apart from a refusal, a 500, or an offline browser. */
  function StallError(url, ms) {
    var err = new Error('The server did not respond within ' + Math.round(ms / 1000) + ' seconds.');
    err.name = 'StallError';
    err.isStall = true;
    err.url = url;
    err.timeoutMs = ms;
    return err;
  }

  function isStall(error) {
    return Boolean(error && (error.isStall || error.name === 'StallError'));
  }

  /* Drop-in replacement for fetch(). Accepts the same arguments plus
     options.timeoutMs. Rejects with a StallError on timeout; anything the
     caller passed as options.signal still aborts the request as usual. */
  var pilotSessions;
  // Same-origin Railway staging works without an override. Other hosting
  // explicitly sets FL_SESSION_API_BASE before this script at deployment.
  var pilotBase = (global.FLDashboard && global.FLDashboard.base) || global.FL_SESSION_API_BASE || (global.location && /\.up\.railway\.app$/.test(global.location.hostname) ? global.location.origin : null);
  if (global.FLKeySessions && global.document && pilotBase) {
    pilotSessions = global.FLKeySessions.installDashboard(pilotBase);
  }
  function fetchWithTimeout(url, options) {
    if(global.FLDashboard)url=global.FLDashboard.route(url,global.FLDashboard.base,global.location);
    if (pilotSessions) {
      var destination = new URL(url, global.location.href);
      // On the hosted staging dashboard, existing ledger URLs point to this
      // staging origin; no requests are sent to another environment.
      if (/\.up\.railway\.app$/.test(destination.hostname)) destination = new URL(destination.pathname + destination.search, pilotBase);
      return pilotSessions.prepare(destination.href, options || {}).then(function (prepared) {
        return timedFetch(destination.href, prepared).then(pilotSessions.observe).then(function(response){return global.FLDashboard ? global.FLDashboard.checkReadAccess(response,prepared) : response;});
      });
    }
    return timedFetch(url, options).then(function(response){return global.FLDashboard ? global.FLDashboard.checkReadAccess(response,options) : response;});
  }
  function timedFetch(url, options) {
    var opts = options || {};
    var ms = typeof opts.timeoutMs === 'number' ? opts.timeoutMs : DEFAULT_TIMEOUT_MS;

    if (typeof AbortController !== 'function') {
      return global.fetch(url, opts); // no abort support; behave as before
    }

    var controller = new AbortController();
    var timedOut = false;
    var timer = setTimeout(function () {
      timedOut = true;
      controller.abort();
    }, ms);

    /* Honour a caller-supplied signal alongside our own. */
    if (opts.signal) {
      if (opts.signal.aborted) controller.abort();
      else opts.signal.addEventListener('abort', function () { controller.abort(); });
    }

    var passthrough = {};
    for (var k in opts) {
      if (Object.prototype.hasOwnProperty.call(opts, k) && k !== 'timeoutMs') passthrough[k] = opts[k];
    }
    passthrough.signal = controller.signal;

    return global.fetch(url, passthrough).then(
      function (res) { clearTimeout(timer); return res; },
      function (err) {
        clearTimeout(timer);
        if (timedOut) throw StallError(url, ms);
        throw err;
      }
    );
  }

  /* Render the stall into an element the caller names, over whatever is
     already there, with a Retry that re-runs the caller's own loader.
     `asOf` is the timestamp of the data still on screen, when there is one. */
  function renderStall(el, retry, asOf) {
    if (!el) return;
    var box = el.querySelector ? el.querySelector('.fl-stall') : null;
    if (!box) {
      box = document.createElement('div');
      box.className = 'fl-stall';
      box.setAttribute('role', 'status');
      el.insertBefore(box, el.firstChild);
    }
    box.textContent = '';

    var msg = document.createElement('span');
    msg.className = 'fl-stall-msg';
    msg.textContent = asOf
      ? 'Server not responding — showing data from ' + asOf + '.'
      : 'Server not responding.';
    box.appendChild(msg);

    if (typeof retry === 'function') {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'fl-stall-retry';
      btn.textContent = 'Retry';
      btn.addEventListener('click', function () {
        btn.disabled = true;
        btn.classList.add('is-submitting');
        btn.textContent = 'Retrying…';
        Promise.resolve()
          .then(retry)
          .catch(function () { /* the loader reports its own failure */ })
          .then(function () {
            if (btn.isConnected) {
              btn.disabled = false;
              btn.classList.remove('is-submitting');
              btn.textContent = 'Retry';
            }
          });
      });
      box.appendChild(btn);
    }
  }

  function clearStall(el) {
    if (!el || !el.querySelector) return;
    var box = el.querySelector('.fl-stall');
    if (box && box.parentNode) box.parentNode.removeChild(box);
  }

  global.FL = global.FL || {};
  global.FL.DEFAULT_TIMEOUT_MS = DEFAULT_TIMEOUT_MS;
  global.FL.fetchWithTimeout = fetchWithTimeout;
  global.FL.isStall = isStall;
  global.FL.renderStall = renderStall;
  global.FL.clearStall = clearStall;
})(typeof window !== 'undefined' ? window : this);
