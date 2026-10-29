/* My PIN: set up this phone, change the PIN, choose a new one after an owner
   reset, or forget the phone. PINs go straight to FL and are never stored. */
(() => {
  const ui = FLLedger; ui.bind();
  const content = document.getElementById('content');
  const session = ui.session;
  const PIN = /^[0-9]{4}$/;

  function pinInput(host, en, es, id) {
    const label = ui.el('label', undefined, host), text = ui.el('span', ui.t(en, es), label);
    text.className = 'pin-label';
    const input = ui.el('input', undefined, label);
    Object.assign(input, { type: 'password', inputMode: 'numeric', autocomplete: 'off', required: true, maxLength: 4, id });
    input.pattern = '[0-9]{4}';
    return input;
  }
  function pinForm(host, fields, submitText, action) {
    const form = ui.el('form', undefined, host); form.className = 'pin-form';
    const inputs = fields.map(([en, es, id]) => pinInput(form, en, es, id));
    const button = ui.el('button', submitText, form); button.type = 'submit'; button.className = 'primary';
    form.onsubmit = async event => {
      event.preventDefault();
      const values = inputs.map(input => input.value); inputs.forEach(input => { input.value = ''; });
      if (!values.every(value => PIN.test(value))) return ui.notice(ui.t('Use exactly 4 digits.', 'Usa exactamente 4 dígitos.'));
      button.disabled = true;
      // PIN messages are not entries, so no "NOT recorded" prefix here.
      try { await action(values); } catch (error) { ui.notice(error.labels ? ui.t(...error.labels) : error.message); } finally { button.disabled = false; }
    };
    return form;
  }
  const done = message => { ui.notice(message, 'good'); render(); };

  function render() {
    content.replaceChildren();
    if (!session.pinEnabled) {
      ui.el('p', ui.t('PIN sign-in is not switched on yet. Use your personal key.', 'El inicio con PIN aún no está activo. Usa tu clave personal.'), content);
      return ui.relabel();
    }
    const actor = session.actor, device = session.device;
    if (!actor) {
      if (device) {
        ui.el('p', ui.t(`This phone is set up for ${device.actor.name}. Enter the PIN above to sign in for the shift.`,
          `Este teléfono es de ${device.actor.name}. Escribe el PIN arriba para iniciar tu turno.`), content);
        const reset = ui.el('details', undefined, content);
        ui.el('summary', ui.t('The owner reset my PIN', 'El dueño restableció mi PIN'), reset);
        ui.el('p', ui.t('Choose a new PIN within 60 minutes of the reset.', 'Elige un PIN nuevo dentro de 60 minutos.'), reset);
        pinForm(reset, [['New PIN', 'PIN nuevo', 'reset-pin'], ['New PIN again', 'Repite el PIN nuevo', 'reset-pin-2']],
          ui.t('Save new PIN and sign in', 'Guardar PIN e iniciar sesión'), async ([pin, again]) => {
            if (pin !== again) throw ui.problem(ui.t('The two PINs do not match.', 'Los dos PIN no coinciden.'));
            await session.resetPin(pin); done(ui.t('New PIN saved. You are signed in.', 'PIN nuevo guardado. Sesión iniciada.'));
          });
      } else {
        ui.el('p', ui.t('Sign in above with your personal key once. Then choose a 4-digit PIN for this phone.',
          'Inicia sesión arriba con tu clave personal una vez. Luego elige un PIN de 4 dígitos para este teléfono.'), content);
      }
      return ui.relabel();
    }
    if (actor.role === 'owner') {
      const p = ui.el('p', undefined, content), a = ui.el('a', ui.t('Manage staff phones and PINs →', 'Administrar teléfonos y PIN del personal →'), p);
      a.href = 'staff-access.html';
    }
    if (session.phoneSession) {
      ui.el('p', ui.t(`Signed in with your PIN on ${device?.label || 'this phone'}. The sign-in lasts the shift (12 hours).`,
        `Sesión con tu PIN en ${device?.label || 'este teléfono'}. Dura el turno (12 horas).`), content);
      ui.el('h2', ui.t('Change my PIN', 'Cambiar mi PIN'), content);
      pinForm(content, [['Current PIN', 'PIN actual', 'current-pin'], ['New PIN', 'PIN nuevo', 'new-pin'], ['New PIN again', 'Repite el PIN nuevo', 'new-pin-2']],
        ui.t('Change PIN', 'Cambiar PIN'), async ([current, pin, again]) => {
          if (pin !== again) throw ui.problem(ui.t('The two new PINs do not match.', 'Los dos PIN nuevos no coinciden.'));
          await session.changePin(current, pin); done(ui.t('PIN changed.', 'PIN cambiado.'));
        });
      ui.el('h2', ui.t('Lost or handing over this phone?', '¿Perdiste o entregas este teléfono?'), content);
      const forget = ui.el('button', ui.t('Forget this phone', 'Olvidar este teléfono'), content);
      forget.onclick = async () => {
        if (!confirm(ui.t('Remove the PIN from this phone? You will need your personal key to set it up again.',
          '¿Quitar el PIN de este teléfono? Necesitarás tu clave personal para configurarlo otra vez.'))) return;
        try { await session.forgetDevice(); done(ui.t('This phone no longer has a PIN.', 'Este teléfono ya no tiene PIN.')); } catch (error) { ui.notice(error.message); }
      };
      return ui.relabel();
    }
    // Signed in with the personal key: set up (or re-set up) this phone.
    ui.el('h2', ui.t('Set up a PIN on this phone', 'Configurar un PIN en este teléfono'), content);
    if (device && device.actor.id !== actor.id)
      ui.el('p', ui.t(`This phone is set up for ${device.actor.name}. Setting it up for you replaces that on this phone.`,
        `Este teléfono es de ${device.actor.name}. Configurarlo para ti lo reemplaza en este teléfono.`), content);
    ui.el('p', ui.t('Use this only on your own phone. Avoid repeats (111111) and sequences (123456).',
      'Úsalo solo en tu propio teléfono. Evita repeticiones (111111) y secuencias (123456).'), content);
    const nameLabel = ui.el('label', undefined, content); ui.el('span', ui.t('Name of this phone', 'Nombre de este teléfono'), nameLabel);
    const name = ui.el('input', undefined, nameLabel); Object.assign(name, { id: 'phone-name', maxLength: 60, value: `${actor.name}'s phone` });
    const keyLabel = ui.el('label', undefined, content); ui.el('span', ui.t('Your personal key (once more, to confirm it is you)', 'Tu clave personal (otra vez, para confirmar que eres tú)'), keyLabel);
    const key = ui.el('input', undefined, keyLabel); Object.assign(key, { id: 'enroll-key', type: 'password', autocomplete: 'off' });
    pinForm(content, [['Choose a 4-digit PIN', 'Elige un PIN de 4 dígitos', 'enroll-pin'], ['Same PIN again', 'Repite el PIN', 'enroll-pin-2']],
      ui.t('Save PIN for this phone', 'Guardar PIN para este teléfono'), async ([pin, again]) => {
        const personalKey = key.value.trim(); key.value = '';
        if (pin !== again) throw ui.problem(ui.t('The two PINs do not match.', 'Los dos PIN no coinciden.'));
        if (!personalKey) throw ui.problem(ui.t('Enter your personal key once more.', 'Escribe tu clave personal otra vez.'));
        await session.enroll(pin, name.value.trim() || `${actor.name}'s phone`, personalKey);
        done(ui.t('Done. Next time, open Factory Ledger on this phone and enter your PIN.',
          'Listo. La próxima vez, abre Factory Ledger en este teléfono y escribe tu PIN.'));
      });
    ui.relabel();
  }
  for (const event of ['ledger-auth', 'ledger-signout', 'ledger-language']) window.addEventListener(event, render);
  ui.ready.then(render).catch(() => {});
})();
