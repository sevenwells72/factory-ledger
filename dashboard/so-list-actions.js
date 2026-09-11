/* Sales Orders list administrative exits. The server owns state and release totals. */
(function () {
  'use strict';

  const REASONS = {
    close: [
      ['shipped_recorded', 'Shipped, recorded'],
      ['shipped_not_recorded', 'Shipped, not recorded'],
      ['short_closed', 'Short-closed'],
    ],
    cancel: [
      ['customer_cancelled', 'Customer cancelled'],
      ['cns_declined', 'CNS declined'],
      ['duplicate', 'Duplicate'],
      ['superseded', 'Superseded'],
      ['other', 'Other'],
    ],
  };
  const TITLES = { close: 'Close', cancel: 'Cancel', reopen: 'Reopen' };
  const PAST = { close: 'closed', cancel: 'cancelled', reopen: 'reopened' };
  let currentDialog = null;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function detailOf(error) {
    const detail = error && (error.detail || (error.payload && error.payload.detail) || error.payload);
    return detail && typeof detail === 'object' && !Array.isArray(detail) ? detail : {};
  }

  function errorMessage(error, context) {
    const detail = detailOf(error);
    const specific = {
      ORDER_NOT_OPEN: 'This order is already closed or cancelled. Refresh the list to see its current state.',
      ORDER_NOT_CLOSED: 'This order is already open. Refresh the list to see its current state.',
      ORDER_ALREADY_SHIPPED: 'Some pounds have already shipped, so this order cannot be cancelled.',
      STATE_NOTE_REQUIRED: 'Add a note explaining why this order is being cancelled.',
      RELATED_SO_REQUIRED: 'Enter the related sales order number.',
      RELATED_SO_INVALID: 'Choose a different sales order as the related SO.',
      RELATED_SO_NOT_FOUND: 'The related sales order was not found. Check its SO number and try again.',
      RELATED_SO_NOT_APPLICABLE: 'The related SO applies only to Duplicate or Superseded. Choose the reason again.',
    };
    if (specific[detail.error_code]) return specific[detail.error_code];
    if (error && error.status === 404) return context === 'related'
      ? 'The related sales order was not found. Check its SO number and try again.'
      : 'This sales order could not be found. Refresh the list and try again.';
    if (error && (error.status === 401 || error.status === 403)) return 'Your connection does not allow this change. Check your access and try again.';
    if (error && error.status === 503) return 'Sales orders are temporarily unavailable. Wait a moment, then try again.';
    if (error && error.status === 409) return 'This order changed while you were working. Refresh the list and try again.';
    if (error && (error.status === 400 || error.status === 422)) return 'Some details could not be accepted. Check the reason, note, and related SO, then preview again.';
    if (error && error.status >= 500) return 'The change could not be confirmed. Refresh the list before trying again.';
    return context === 'related'
      ? 'The related sales order could not be checked. Check your connection and try again.'
      : 'The request could not be confirmed. Check your connection and refresh the list before trying again.';
  }

  function open(order, options) {
    if (!order || !options || typeof options.request !== 'function' || typeof options.format !== 'function') return null;
    if (currentDialog) {
      if (currentDialog.open && currentDialog.getAttribute('aria-busy') === 'true') return currentDialog;
      if (currentDialog.open) currentDialog.close();
      currentDialog.remove();
      currentDialog = null;
    }
    const trigger = options.trigger || document.activeElement;
    let action = options.action || (order.state === 'closed' || order.state === 'cancelled' ? 'reopen' : 'close');
    if (!TITLES[action]) return null;
    let busy = false;
    let committed = false;
    let previewPayload = null;
    let suggestedReason = null;
    const orderKey = order.order_id || order.id || order.order_number;
    const orderLabel = String(order.order_number || 'sales order');

    const dialog = element('dialog', 'so-exit-dialog');
    currentDialog = dialog;
    dialog.setAttribute('aria-labelledby', 'so-exit-title');
    dialog.setAttribute('aria-describedby', 'so-exit-description');
    const form = element('form', 'so-exit-form');
    const header = element('header', 'so-exit-header');
    const title = element('h2');
    title.id = 'so-exit-title';
    const description = element('p', 'so-exit-description');
    description.id = 'so-exit-description';
    header.append(title, description);
    const fields = element('div', 'so-exit-fields');
    const reasonGroup = element('div', 'so-exit-field');
    const reasonLabel = element('label', '', 'Reason');
    reasonLabel.htmlFor = 'so-exit-reason';
    const reason = element('select');
    reason.id = 'so-exit-reason';
    reason.name = 'reason';
    reasonGroup.append(reasonLabel, reason);

    const relatedGroup = element('div', 'so-exit-field');
    const relatedLabel = element('label', '', 'Related SO');
    relatedLabel.htmlFor = 'so-exit-related';
    const related = element('input');
    related.id = 'so-exit-related';
    related.name = 'related_so';
    related.type = 'text';
    related.autocomplete = 'off';
    related.spellcheck = false;
    related.setAttribute('aria-describedby', 'so-exit-related-hint');
    const relatedHint = element('p', 'so-exit-hint', 'Enter the SO number this order duplicates or was replaced by.');
    relatedHint.id = 'so-exit-related-hint';
    relatedGroup.append(relatedLabel, related, relatedHint);

    const noteGroup = element('div', 'so-exit-field');
    const noteLabel = element('label');
    noteLabel.htmlFor = 'so-exit-note';
    const note = element('textarea');
    note.id = 'so-exit-note';
    note.name = 'note';
    note.rows = 3;
    noteGroup.append(noteLabel, note);
    fields.append(reasonGroup, relatedGroup, noteGroup);

    const errorBox = element('div', 'so-exit-error');
    errorBox.setAttribute('role', 'alert');
    errorBox.hidden = true;
    const errorText = element('p');
    const offer = element('button', 'so-exit-button so-exit-secondary');
    offer.type = 'button';
    offer.hidden = true;
    errorBox.append(errorText, offer);
    const preview = element('div', 'so-exit-preview');
    preview.setAttribute('role', 'status');
    preview.setAttribute('aria-live', 'polite');
    preview.tabIndex = -1;
    preview.hidden = true;
    const previewSummary = element('p', 'so-exit-release-summary');
    const previewDetail = element('p', 'so-exit-hint');
    preview.append(previewSummary, previewDetail);

    const footer = element('footer', 'so-exit-footer');
    const dismiss = element('button', 'so-exit-button so-exit-secondary', 'Back');
    dismiss.type = 'button';
    const previewButton = element('button', 'so-exit-button so-exit-primary');
    previewButton.type = 'submit';
    const commitButton = element('button', 'so-exit-button');
    commitButton.type = 'button';
    commitButton.hidden = true;
    footer.append(dismiss, previewButton, commitButton);
    form.append(header, fields, errorBox, preview, footer);
    dialog.append(form);
    document.body.append(dialog);

    function updateLabels() {
      dialog.dataset.action = action;
      title.textContent = `${TITLES[action]} ${orderLabel}`;
      description.textContent = action === 'reopen'
        ? 'Return this order to Open. Previously released reservations are not restored; reserve stock again after reopening.'
        : action === 'close'
          ? 'Remove this order from Open and release its stock reservations. Closing does not record a shipment.'
          : 'Cancel this unshipped order and release its stock reservations.';
      reason.replaceChildren();
      (REASONS[action] || []).forEach(([value, label]) => {
        const option = element('option', '', label);
        option.value = value;
        reason.append(option);
      });
      reasonGroup.hidden = action === 'reopen';
      reason.required = action !== 'reopen';
      reason.disabled = action === 'reopen';
      commitButton.className = `so-exit-button ${action === 'cancel' ? 'so-exit-destructive' : 'so-exit-primary'}`;
      updateConditionalFields();
      updateButtons();
    }

    function updateConditionalFields() {
      const needsRelated = action === 'cancel' && ['duplicate', 'superseded'].includes(reason.value);
      relatedGroup.hidden = !needsRelated;
      related.required = needsRelated;
      related.disabled = !needsRelated || busy;
      note.required = action === 'cancel' && reason.value === 'other';
      noteLabel.textContent = note.required ? 'Note (required)' : 'Note (optional)';
    }

    function updateButtons() {
      previewButton.textContent = busy && !previewPayload ? 'Checking…' : `Preview ${action}`;
      commitButton.textContent = busy && previewPayload ? `${TITLES[action]} in progress…` : `${TITLES[action]} order`;
      previewButton.hidden = !!previewPayload || committed;
      commitButton.hidden = !previewPayload || committed;
      [previewButton, commitButton, dismiss, offer, note].forEach(control => { control.disabled = busy; });
      reason.disabled = busy || action === 'reopen';
      updateConditionalFields();
      dialog.setAttribute('aria-busy', String(busy));
    }

    function invalidatePreview() {
      if (busy || committed) return;
      previewPayload = null;
      preview.hidden = true;
      errorBox.hidden = true;
      offer.hidden = true;
      suggestedReason = null;
      related.setCustomValidity('');
      note.setCustomValidity('');
      updateButtons();
    }

    function showError(error, context) {
      errorText.textContent = errorMessage(error, context);
      const detail = detailOf(error);
      suggestedReason = action === 'cancel' && error.status === 409 && detail.suggested_action === 'close'
        && REASONS.close.some(([value]) => value === detail.suggested_reason) ? detail.suggested_reason : null;
      offer.hidden = !suggestedReason;
      if (suggestedReason) {
        const label = REASONS.close.find(([value]) => value === suggestedReason)[1];
        offer.textContent = `Preview close instead · ${label}`;
      }
      errorBox.hidden = false;
    }

    async function performPreview() {
      if (busy || committed) return;
      related.setCustomValidity(related.required && !related.value.trim() ? 'Enter the related SO number.' : '');
      note.setCustomValidity(note.required && !note.value.trim() ? 'Add a note explaining the cancellation.' : '');
      if (!form.reportValidity()) return;
      previewPayload = null;
      preview.hidden = true;
      errorBox.hidden = true;
      offer.hidden = true;
      busy = true;
      updateButtons();
      let context = 'preview';
      try {
        const payload = { note: note.value.trim() || null };
        if (action !== 'reopen') payload.reason = reason.value;
        if (related.required) {
          context = 'related';
          const target = await options.request(`/sales/orders/${encodeURIComponent(related.value.trim())}`);
          const targetId = Number(target.order_id || target.id);
          if (!Number.isInteger(targetId) || targetId <= 0) throw new Error('Related SO lookup unavailable');
          if (String(targetId) === String(orderKey) || String(target.order_number) === String(order.order_number)) {
            related.setCustomValidity('Choose a different sales order as the related SO.');
            throw { status: 400, detail: { error_code: 'RELATED_SO_INVALID' } };
          }
          payload.related_so_id = targetId;
        }
        context = 'preview';
        const result = await options.request(`/sales/orders/${encodeURIComponent(orderKey)}/${action}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...payload, mode: 'preview' }),
        });
        // A preview must explicitly provide the reservation list; never imply
        // zero releases after a malformed or incomplete server response.
        if (result.mode !== 'preview' || !Array.isArray(result.reservations_to_release)) throw new Error('Preview unavailable');
        const reservations = result.reservations_to_release;
        const pounds = reservations.reduce((sum, row) => sum + Number(row.quantity_lb), 0);
        if (!Number.isFinite(pounds) || pounds < 0) throw new Error('Preview totals unavailable');
        previewPayload = payload;
        previewSummary.textContent = `Will release ${options.format(reservations.length)} reservations (${options.format(pounds, 'pounds')} lb)`;
        previewDetail.textContent = action === 'reopen'
          ? 'This order will return to Open. Previously released reservations will not be restored.'
          : `This order will be ${PAST[action]}. Released stock becomes available for other orders.`;
        preview.hidden = false;
      } catch (error) {
        showError(error, context);
      } finally {
        busy = false;
        updateButtons();
        if (previewPayload) (action === 'cancel' ? preview : commitButton).focus();
        else if (!offer.hidden) offer.focus();
      }
    }

    async function commit() {
      if (busy || committed || !previewPayload) return;
      busy = true;
      errorBox.hidden = true;
      updateButtons();
      try {
        await options.request(`/sales/orders/${encodeURIComponent(orderKey)}/${action}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...previewPayload, mode: 'commit' }),
        });
        committed = true;
        try {
          if (options.onCommitted) await options.onCommitted();
          dialog.close();
        } catch (_) {
          preview.hidden = true;
          errorText.textContent = `Order ${PAST[action]}. The list could not refresh; use Refresh to see the updated order.`;
          errorBox.hidden = false;
          offer.hidden = true;
          fields.hidden = true;
          dismiss.textContent = 'Dismiss';
        }
      } catch (error) {
        // Any failure requires a new preview. Do not blindly resend a commit
        // after a connection failure that may have followed a successful save.
        previewPayload = null;
        preview.hidden = true;
        showError(error, 'commit');
      } finally {
        busy = false;
        updateButtons();
        if (dialog.open && !offer.hidden) offer.focus();
        else if (dialog.open && committed) dismiss.focus();
      }
    }

    fields.addEventListener('input', invalidatePreview);
    fields.addEventListener('change', invalidatePreview);
    form.addEventListener('submit', event => {
      event.preventDefault();
      // Enter on a form field previews only; a commit always needs its button.
      if (!previewPayload) performPreview();
    });
    offer.addEventListener('click', () => {
      if (busy || !suggestedReason) return;
      const nextReason = suggestedReason;
      action = 'close';
      invalidatePreview();
      updateLabels();
      reason.value = nextReason;
      performPreview();
    });
    commitButton.addEventListener('click', commit);
    dismiss.addEventListener('click', () => { if (!busy) dialog.close(); });
    dialog.addEventListener('cancel', event => { if (busy) event.preventDefault(); });
    dialog.addEventListener('close', () => {
      dialog.remove();
      if (currentDialog !== dialog) return;
      currentDialog = null;
      const focusTarget = options.getFocusTarget?.() || (trigger && trigger.isConnected ? trigger
        : document.querySelector('[data-orders-tab][aria-selected="true"]') || document.getElementById('refresh-btn'));
      if (focusTarget && !focusTarget.disabled) focusTarget.focus({ preventScroll: true });
    }, { once: true });
    updateLabels();
    dialog.showModal();
    (action === 'reopen' ? note : reason).focus();
    return dialog;
  }

  window.SOListActions = Object.freeze({ open });
})();
