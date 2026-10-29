/* Staff & phones (owner only): who has which phone set up for a PIN, and the
   four owner actions: remove a phone, reset its PIN, remove all of a person's
   sign-ins, deactivate/reactivate a person. Roles are never changed here. */
(() => {
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
  async function load() {
    content.replaceChildren();
    if (!ui.session.pinEnabled) { ui.el('p', ui.t('PIN sign-in is not switched on yet.', 'El inicio con PIN aún no está activo.'), content); return; }
    if (!ui.session.actor) { ui.el('p', ui.t('Sign in above as the owner.', 'Inicia sesión arriba como dueño.'), content); return; }
    if (ui.session.actor.role !== 'owner') { ui.el('p', ui.t('Only the owner can manage staff phones.', 'Solo el dueño puede administrar los teléfonos.'), content); return; }
    try { render((await ui.request('/auth/devices', undefined, { read: true })).people); }
    catch (error) { ui.failure(error); }
  }
  window.addEventListener('ledger-auth', load);
  window.addEventListener('ledger-signout', load);
  window.addEventListener('ledger-language', load);
  ui.ready.then(load).catch(() => {});
})();
