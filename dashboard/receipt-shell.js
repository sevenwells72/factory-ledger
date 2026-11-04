/* Receipt navigation shared by every dashboard page, including standalone tools. */
(function () {
  'use strict';
  const asset = new URL(document.currentScript.src), directory = new URL('.', asset);
  const page = (name, params = {}) => { const url = new URL(name, directory); Object.entries(params).forEach(([k,v]) => url.searchParams.set(k,v)); return url.href; };
  window.FLReceipts = { page, link: number => page('receipt.html', {number}) };
  function mount() {
    if (document.querySelector('.receipt-lookup')) return;
    const host = document.querySelector('[data-receipt-tools-host]') || document.querySelector('header, .header, .topbar') || document.body;
    const bar = document.createElement('nav'); bar.className = 'receipt-tools'; bar.setAttribute('aria-label', 'Ledger tools / Herramientas');
    const form = document.createElement('form'); form.className = 'receipt-lookup';
    const label = document.createElement('label'); label.textContent = 'Receipt # / Recibo #';
    const input = document.createElement('input'); input.name = 'number'; input.required = true; input.autocomplete = 'off'; input.placeholder = 'RCV-261013-001'; label.append(input);
    const button = document.createElement('button'); button.type = 'submit'; button.textContent = 'Look up / Buscar';
    form.append(label, button); form.onsubmit = event => {event.preventDefault(); location.assign(page('receipt.html',{number:input.value.trim().toUpperCase()}));};
    bar.append(form);
    for (const [name,text] of [['exceptions.html','Exceptions / Excepciones'],['entry.html','Record entry / Registrar'],['office.html','Office / Oficina'],['fl-assistant.html','FL Assistant / Asistente FL']]) {const a=document.createElement('a');a.dataset.flWriteUi='';a.href=page(name);a.textContent=text;bar.append(a);}
    // Read-only views of the day work without personal sign-in, so this link always shows.
    const shift=document.createElement('a');shift.href=page('shift-summary.html');shift.textContent='Shift summary / Resumen del turno';bar.append(shift);
    host.append(bar);
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',mount); else mount();
})();
