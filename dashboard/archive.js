// Product archive controls for the dashboard's item lists (finished goods,
// batches, ingredients/materials, supplies). Archive hides an item from active
// lists; it never deletes and never changes stock, lots or history.
// Backend: product_archive.py (POST /admin/products/archive | /unarchive,
// owner only). Lists ask for archived rows with ?include_archived=true.
(function (root) {
  'use strict';

  const PREF = 'fl-show-archived:';
  const OWNER = 'Michael';
  const ICON_ARCHIVE = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false"><path fill="none" stroke="currentColor" stroke-width="1.5" d="M2 3.5h12v3H2zM3 6.5v6h10v-6M6.5 9h3"/></svg>';
  const ICON_RESTORE = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false"><path fill="none" stroke="currentColor" stroke-width="1.5" d="M3 8a5 5 0 1 0 1.5-3.5M3 2.5v3h3"/></svg>';

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  // ── Pure helpers (unit-tested in tests/test_archive_ui.js) ──
  function confirmText(names, restore = false) {
    const list = names.length === 1 ? names[0] : `${names.length} items`;
    if (restore) {
      return { en: `Restore ${list}? It will show in active lists again.`,
               es: `¿Restaurar ${list}? Volverá a aparecer en las listas activas.` };
    }
    return { en: `Archive ${list}? It will be hidden but history is kept.`,
             es: `¿Archivar ${list}? Se ocultará, pero se conserva el historial.` };
  }

  function errorText(status, body) {
    const detail = body && (body.detail || body.error_detail);
    if (detail && typeof detail === 'object' && detail.message) {
      return detail.message_es ? `${detail.message} / ${detail.message_es}` : detail.message;
    }
    if (typeof detail === 'string') return `HTTP ${status}: ${detail}`;
    return `HTTP ${status}: request failed. Nothing changed. / No se cambió nada.`;
  }

  function withArchivedQuery(path, show) {
    if (!show) return path;
    return path + (path.includes('?') ? '&' : '?') + 'include_archived=true';
  }

  // The archived tag shown next to an item name (lists and search).
  function tag() { return '<span class="archived-tag">Archived</span>'; }

  // Name-cell controls. `item` needs product_id, name and archived.
  function rowControls(listKey, item, canArchive) {
    if (!canArchive || item.product_id == null) return item.archived ? tag() : '';
    const name = esc(item.name);
    let html = item.archived ? tag() : '';
    if (item.archived) {
      html += `<button type="button" class="archive-row-btn archive-restore" data-archive-restore="${item.product_id}" data-archive-name="${name}" title="Unarchive / Restaurar" aria-label="Unarchive ${name} / Restaurar">${ICON_RESTORE}</button>`;
    } else {
      html += `<button type="button" class="archive-row-btn" data-archive-one="${item.product_id}" data-archive-name="${name}" title="Archive / Archivar" aria-label="Archive ${name} / Archivar">${ICON_ARCHIVE}</button>`;
    }
    return html;
  }

  function rowCheckbox(listKey, item, canArchive) {
    if (!canArchive || item.archived || item.product_id == null) return '';
    return `<input type="checkbox" class="archive-select" data-archive-list="${esc(listKey)}" data-product-id="${item.product_id}" data-archive-name="${esc(item.name)}" aria-label="Select ${esc(item.name)} / Seleccionar">`;
  }

  function rowClass(item) { return item.archived ? ' archived-row' : ''; }

  // ── Browser-only state and actions ──
  function storage() { try { return root.localStorage; } catch (_) { return null; } }
  function showArchived(listKey) {
    try { return storage()?.getItem(PREF + listKey) === '1'; } catch (_) { return false; }
  }
  function setShowArchived(listKey, on) {
    try { const s = storage(); if (on) s?.setItem(PREF + listKey, '1'); else s?.removeItem(PREF + listKey); } catch (_) { /* per-viewer only */ }
  }

  function person() {
    try { return root.FLDashboard?.person?.()?.person || null; } catch (_) { return null; }
  }
  function canArchive() { return person()?.display_name === OWNER; }

  function toolbar(listKey) {
    const on = showArchived(listKey);
    let html = `<div class="archive-toolbar" data-archive-toolbar="${esc(listKey)}">`;
    html += `<label class="archive-toggle"><input type="checkbox" data-archive-toggle="${esc(listKey)}"${on ? ' checked' : ''}> Show archived / Mostrar archivados</label>`;
    if (canArchive()) {
      html += `<button type="button" class="btn-sm archive-bulk-btn" data-archive-bulk="${esc(listKey)}" disabled>Archive selected / Archivar seleccionados</button>`;
    }
    return html + '</div>';
  }

  // In-page confirm (no window.confirm: keeps the page automation-safe and
  // lets the message carry both languages).
  function confirmDialog(text, okLabel) {
    const doc = root.document;
    return new Promise(resolve => {
      const overlay = doc.createElement('div');
      overlay.className = 'archive-confirm-overlay';
      overlay.innerHTML = `<div class="archive-confirm" role="dialog" aria-modal="true" aria-labelledby="archive-confirm-text">
        <p id="archive-confirm-text"><strong>${esc(text.en)}</strong><br><span lang="es">${esc(text.es)}</span></p>
        <div class="archive-confirm-actions"><button type="button" class="btn-sm" data-archive-cancel>Cancel / Cancelar</button>
        <button type="button" class="btn-sm archive-confirm-ok" data-archive-ok>${esc(okLabel)}</button></div></div>`;
      const done = value => { overlay.remove(); doc.removeEventListener('keydown', onKey); resolve(value); };
      const onKey = event => { if (event.key === 'Escape') done(false); };
      overlay.addEventListener('click', event => {
        if (event.target === overlay || event.target.closest('[data-archive-cancel]')) done(false);
        else if (event.target.closest('[data-archive-ok]')) done(true);
      });
      doc.addEventListener('keydown', onKey);
      doc.body.appendChild(overlay);
      overlay.querySelector('[data-archive-ok]').focus();
    });
  }

  async function post(path, ids) {
    const client = root.FLDashboard.person();
    const headers = { 'Content-Type': 'application/json', ...client.headers({ write: true }) };
    const response = await root.fetch(root.FLDashboard.base + path, {
      method: 'POST', headers, body: JSON.stringify({ product_ids: ids }), cache: 'no-store' });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorText(response.status, body));
    return body;
  }

  async function run(ids, names, restore, onDone, onError) {
    const ok = await confirmDialog(confirmText(names, restore),
      restore ? 'Unarchive / Restaurar' : 'Archive / Archivar');
    if (!ok) return;
    try {
      await post(restore ? '/admin/products/unarchive' : '/admin/products/archive', ids);
      onDone();
    } catch (error) {
      onError(error.message || String(error));
    }
  }

  // Wire one rendered list. `refresh` re-fetches it; `onError` shows a message.
  function bind(container, listKey, refresh, onError) {
    const scope = container;
    const toggle = scope.querySelector(`[data-archive-toggle="${listKey}"]`);
    if (toggle && !toggle.dataset.bound) {
      toggle.dataset.bound = '1';
      toggle.addEventListener('change', () => { setShowArchived(listKey, toggle.checked); refresh(); });
    }
    const bulk = scope.querySelector(`[data-archive-bulk="${listKey}"]`);
    const selected = () => [...container.querySelectorAll(`.archive-select[data-archive-list="${listKey}"]:checked`)];
    const sync = () => { if (bulk) bulk.disabled = selected().length === 0; };
    container.querySelectorAll('.archive-select, .archive-row-btn').forEach(el => {
      // Rows expand on click; the archive controls must not.
      el.addEventListener('click', event => event.stopPropagation());
      el.addEventListener('keydown', event => event.stopPropagation());
    });
    container.querySelectorAll('.archive-select').forEach(box => box.addEventListener('change', sync));
    container.querySelectorAll('[data-archive-one]').forEach(btn => btn.addEventListener('click', () =>
      run([Number(btn.dataset.archiveOne)], [btn.dataset.archiveName], false, refresh, onError)));
    container.querySelectorAll('[data-archive-restore]').forEach(btn => btn.addEventListener('click', () =>
      run([Number(btn.dataset.archiveRestore)], [btn.dataset.archiveName], true, refresh, onError)));
    if (bulk) {
      bulk.onclick = () => {
        const boxes = selected();
        if (boxes.length) run(boxes.map(b => Number(b.dataset.productId)), boxes.map(b => b.dataset.archiveName), false, refresh, onError);
      };
      sync();
    }
  }

  const api = { confirmText, errorText, withArchivedQuery, tag, rowControls, rowCheckbox, rowClass,
    showArchived, setShowArchived, canArchive, toolbar, bind, confirmDialog };
  if (typeof module !== 'undefined') module.exports = api;
  root.FLArchive = api;
})(typeof window === 'undefined' ? globalThis : window);
