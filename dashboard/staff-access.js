/* Staff & phones (owner only): who has which phone set up for a PIN, and the
   four owner actions: remove a phone, reset its PIN, remove all of a person's
   sign-ins, deactivate/reactivate a person. Roles are never changed here. */
(() => {
  const root = window;
  const ui = FLLedger; ui.bind();
  const content = document.getElementById('content');
  const status = {
    active: ['PIN set', 'PIN activo'], locked: ['Locked for 15 minutes (wrong PINs)', 'Bloqueado 15 minutos (PIN incorrectos)'],
    blocked: ['Blocked: too many wrong PINs, reset needed', 'Bloqueado: demasiados PIN incorrectos, hay que restablecer'],
    reset_pending: ['Waiting for a new PIN', 'Esperando un PIN nuevo'],
  };
  const roles = { owner: ['owner', 'dueño'], floor: ['floor', 'planta'], office: ['office', 'oficina'] };
  const when = value => value ? ui.plantTime(value, ui.lang) : ui.t('never', 'nunca');

  async function act(question, path, body, success) {
    if (!confirm(String(question))) return;
    try { await ui.request(path, body || {}); ui.notice(success, 'good'); await load(); }
    catch (error) { ui.failure(error); }
  }
  function button(host, text, onclick) { const b = ui.el('button', text, host); b.type = 'button'; b.onclick = onclick; return b; }

  function render(people) {
    content.replaceChildren();
    button(content, ui.t('Refresh', 'Actualizar'), load);
    ui.el('p', ui.t('Phones are set up by each person on their own phone (personal key once, then a PIN). Removing a phone or resetting a PIN signs that phone out at once.',
      'Cada persona configura su propio teléfono (clave personal una vez, luego un PIN). Quitar un teléfono o restablecer un PIN cierra su sesión de inmediato.'), content);
    const me = ui.session.actor;
    for (const person of people) {
      const card = ui.el('article', undefined, content); card.className = 'staff-card';
      ui.el('h2', `${person.name} · ${roles[person.role]?.[ui.lang === 'es' ? 1 : 0] || person.role}`, card);
      if (!person.active) ui.el('p', ui.t('Deactivated: cannot sign in with a key or a PIN.', 'Desactivado: no puede iniciar sesión con clave ni PIN.'), card).className = 'bad';
      if (!person.devices.length) ui.el('p', ui.t('No phone set up.', 'Ningún teléfono configurado.'), card);
      for (const device of person.devices) {
        const row = ui.el('div', undefined, card); row.className = 'staff-device';
        ui.el('strong', device.label, row);
        const [en, es] = status[device.status] || [device.status, device.status];
        ui.el('p', ui.t(`${en} · set up ${when(device.enrolled_at)} · last PIN sign-in ${when(device.last_used_at)}`,
          `${es} · configurado ${when(device.enrolled_at)} · último inicio con PIN ${when(device.last_used_at)}`), row);
        // Another owner's phones are not managed here (the server refuses too).
        if (person.role === 'owner' && person.id !== me?.id) continue;
        button(row, ui.t('Reset PIN', 'Restablecer PIN'), () => act(
          ui.t(`Reset the PIN on ${device.label}? ${person.name} then has 60 minutes to choose a new PIN on that phone. If the phone is lost, use Remove phone instead.`,
            `¿Restablecer el PIN en ${device.label}? ${person.name} tendrá 60 minutos para elegir uno nuevo en ese teléfono. Si el teléfono se perdió, usa Quitar teléfono.`),
          `/auth/devices/${device.id}/reset-pin`, null,
          ui.t(`PIN reset. Tell ${person.name} to open My PIN on that phone within 60 minutes.`, `PIN restablecido. Dile a ${person.name} que abra Mi PIN en ese teléfono dentro de 60 minutos.`)));
        button(row, ui.t('Remove phone', 'Quitar teléfono'), () => act(
          ui.t(`Remove ${device.label}? It is signed out now and needs the personal key to be set up again.`,
            `¿Quitar ${device.label}? Se cierra su sesión y necesitará la clave personal para configurarse otra vez.`),
          `/auth/devices/${device.id}/revoke`, null, ui.t('Phone removed.', 'Teléfono quitado.')));
      }
      if (me && person.id !== me.id && person.role !== 'owner') {
        const tools = ui.el('div', undefined, card); tools.className = 'staff-tools';
        button(tools, ui.t('Sign out everywhere and remove phones', 'Cerrar todas las sesiones y quitar teléfonos'), () => act(
          ui.t(`Sign ${person.name} out on every device and remove all their phones? Their personal key keeps working.`,
            `¿Cerrar todas las sesiones de ${person.name} y quitar sus teléfonos? Su clave personal sigue funcionando.`),
          `/auth/actors/${person.id}/remove-access`, null, ui.t('All sign-ins removed.', 'Todas las sesiones quitadas.')));
        const next = !person.active;
        button(tools, next ? ui.t('Reactivate', 'Reactivar') : ui.t('Deactivate', 'Desactivar'), () => act(
          next ? ui.t(`Reactivate ${person.name}? Their personal key works again; phones must be set up again.`, `¿Reactivar a ${person.name}? Su clave personal vuelve a funcionar; los teléfonos se configuran de nuevo.`)
            : ui.t(`Deactivate ${person.name}? Their key and PINs stop working at once. Their past entries stay.`, `¿Desactivar a ${person.name}? Su clave y PIN dejan de funcionar. Sus registros anteriores se mantienen.`),
          `/auth/actors/${person.id}/active`, { active: next }, next ? ui.t('Reactivated.', 'Reactivado.') : ui.t('Deactivated.', 'Desactivado.')));
      }
    }
    ui.relabel();
  }
  // 091: browsers approved for one-tap name selection. Approving needs THIS page's
  // owner sign-in (Michael's own key or PIN) — a tapped name never reaches it.
  function renderBrowsers(host, data) {
    const box = ui.el('section', undefined, host); box.className = 'staff-card'; box.id = 'browsers';
    ui.el('h2', ui.t('Browsers for name selection', 'Navegadores para elegir nombre'), box);
    ui.el('p', ui.t('Approve each browser once. Then anyone using it taps their name. A name identifies who is entering; it does not prove it. Owner decisions always need your own sign-in.',
      'Aprueba cada navegador una vez. Luego quien lo use toca su nombre. El nombre identifica quién registra; no lo prueba. Las decisiones del dueño siempre necesitan tu propio inicio de sesión.'), box);
    if (data.people_problem) ui.el('p', ui.t('The list of people is not set up correctly, so nobody can record by name yet.', 'La lista de personas no está bien configurada; nadie puede registrar por nombre todavía.'), box).className = 'bad';
    else ui.el('p', data.people.map(p => `${p.display_name} (${p.available ? p.role_label : ui.t('not available', 'no disponible')})`).join(' · '), box);
    const form = ui.el('form', undefined, box); form.className = 'staff-tools';
    const codeLabel = ui.el('label', undefined, form); ui.el('span', ui.t('Code shown on the browser', 'Código que muestra el navegador'), codeLabel);
    const code = ui.el('input', undefined, codeLabel); Object.assign(code, { id: 'browser-code', autocomplete: 'off', required: true, maxLength: 20 }); code.setAttribute('autocapitalize', 'characters');
    const nameLabel = ui.el('label', undefined, form); ui.el('span', ui.t('Name this browser', 'Nombre de este navegador'), nameLabel);
    const label = ui.el('input', undefined, nameLabel); Object.assign(label, { id: 'browser-label', required: true, maxLength: 60, placeholder: ui.t('Floor tablet', 'Tableta del piso') });
    const submit = ui.el('button', ui.t('Approve browser', 'Aprobar navegador'), form); submit.type = 'submit'; submit.className = 'primary';
    form.onsubmit = async event => {
      event.preventDefault(); submit.disabled = true;
      try { await ui.request('/auth/browsers/approve', { pairing_code: code.value.trim(), label: label.value.trim() }); code.value = ''; label.value = '';
        ui.notice(ui.t('Browser approved. It now shows the names.', 'Navegador aprobado. Ya muestra los nombres.'), 'good'); await load(); }
      catch (error) { ui.failure(error); } finally { submit.disabled = false; }
    };
    for (const browser of data.browsers.filter(b => b.status === 'approved')) {
      const row = ui.el('div', undefined, box); row.className = 'staff-device';
      ui.el('strong', browser.label, row);
      ui.el('p', ui.t(`Approved ${when(browser.approved_at)} by ${browser.approved_by || '—'} · last used ${when(browser.last_seen_at)}`,
        `Aprobado ${when(browser.approved_at)} por ${browser.approved_by || '—'} · último uso ${when(browser.last_seen_at)}`), row);
      button(row, ui.t('Remove browser', 'Quitar navegador'), () => act(
        ui.t(`Remove ${browser.label}? Nobody can record on it until you approve it again. Past entries stay.`, `¿Quitar ${browser.label}? Nadie podrá registrar en él hasta que lo apruebes otra vez. Los registros anteriores se mantienen.`),
        `/auth/browsers/${browser.id}/revoke`, null, ui.t('Browser removed.', 'Navegador quitado.')));
    }
    const waiting = data.browsers.filter(b => b.status === 'pending').length;
    if (waiting) ui.el('p', ui.t(`${waiting} browser(s) waiting for a code.`, `${waiting} navegador(es) esperando un código.`), box);
  }
  async function load() {
    content.replaceChildren();
    if (!ui.session.actor) { ui.el('p', ui.t('Sign in above as the owner (your own key or PIN).', 'Inicia sesión arriba como dueño (tu propia clave o PIN).'), content); return; }
    if (ui.session.actor.role !== 'owner') { ui.el('p', ui.t('Only the owner can manage staff phones and browsers.', 'Solo el dueño puede administrar teléfonos y navegadores.'), content); return; }
    if (ui.session.pinEnabled) {
      try { render((await ui.request('/auth/devices', undefined, { read: true })).people); }   // replaces the content
      catch (error) { ui.failure(error); }
    } else ui.el('p', ui.t('PIN sign-in is not switched on.', 'El inicio con PIN no está activo.'), content);
    if (root.FLDashboard?.personSelect === true) {
      const holder = document.createElement('div');
      try { renderBrowsers(holder, await ui.request('/auth/browsers', undefined, { read: true })); content.prepend(...holder.childNodes); ui.relabel(); }
      catch (error) { ui.failure(error); }
    }
  }
  window.addEventListener('ledger-auth', load);
  window.addEventListener('ledger-signout', load);
  window.addEventListener('ledger-language', load);
  ui.ready.then(load).catch(() => {});
})();
