(async function () {
  'use strict';
  const status = document.getElementById('pin-status'), own = document.getElementById('own-pin'), admin = document.getElementById('admin-pin');
  let actor;
  async function load() {
    const person = await FLSession.requireSession();
    actor = person;
    own.hidden = false; admin.hidden = actor.role !== 'owner';
    if (actor.role !== 'owner') return;
    const response = await FLSession.fetch(FLSession.base + '/actors/pins');
    if (!response.ok) { status.textContent = 'PIN administration is unavailable.'; return; }
    const data = await response.json(), picker = document.getElementById('pin-person');
    if (FLSession.actor()?.id !== person.id) return;
    picker.replaceChildren();
    data.actors.filter(p => p.active).forEach(person => {
      const option = document.createElement('option'); option.value = person.id;
      option.textContent = person.name + ' — ' + (person.pin_set ? 'PIN set' : 'Needs first PIN'); picker.append(option);
    });
    const me = data.actors.find(p => p.id === actor.id);
    const bootstrap = me && !me.pin_set;
    own.elements.current.required = !bootstrap; own.elements.current.parentElement.hidden = bootstrap;
    own.querySelector('h2').textContent = bootstrap ? 'Set Michael’s first PIN' : 'Change my PIN';
    status.textContent = bootstrap ? 'Set your own PIN first using your personal owner sign-in. Then sign in with that PIN to set everyone else’s.' :
      (data.global_locked_until && new Date(data.global_locked_until) > new Date() ? 'PIN sign-in is temporarily locked after repeated attempts. ' : '') +
      data.failures_last_hour + ' failed PIN attempts in the past hour.';
  }
  async function save(event, ownerForm) {
    event.preventDefault(); const form = event.target, button = form.querySelector('button'); button.disabled = true;
    const id = ownerForm ? Number(document.getElementById('pin-person').value) : actor.id;
    const body = {pin:form.elements.pin.value}; if (!ownerForm && form.elements.current.value) body.current_pin = form.elements.current.value;
    form.elements.pin.value = ''; if (!ownerForm) form.elements.current.value = '';
    try {
      const response = await FLSession.fetch(FLSession.base + '/actors/' + id + '/pin', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
      const data = await response.json();
      if (!response.ok) { status.textContent = data.detail?.message || 'PIN change failed.'; return; }
      status.textContent = 'PIN saved. All of that person’s sessions have ended.';
      if (data.sign_in_again) { await FLSession.signOut(); await load(); } else await load();
    } catch (error) { status.textContent = error.message || 'PIN change failed.'; }
    finally { delete body.pin; delete body.current_pin; button.disabled = false; }
  }
  own.onsubmit = event => void save(event, false); admin.onsubmit = event => void save(event, true);
  await load();
  window.addEventListener('fl-session-change', event => {
    own.hidden = admin.hidden = true;
    document.querySelectorAll('input[type=password]').forEach(input => { input.value = ''; });
    document.getElementById('pin-person').replaceChildren();
    if (!event.detail) { status.textContent = 'Signed out. Enter your PIN to continue.'; return; }
    void load().catch(() => { status.textContent = 'PIN settings could not be loaded. Refresh and try again.'; });
  });
})();
