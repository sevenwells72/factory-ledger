(function() {
  'use strict';

  const MONTH_FMT = new Intl.DateTimeFormat(undefined, { month: 'short', year: 'numeric' });
  const DOW_LABELS = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
  const SHIP_DATES_EVENT = 'factory-ledger:ship-dates';
  const SALES_API_BASE = 'https://fastapi-production-b73a.up.railway.app';
  const SALES_API_KEY = 'ledger-secret-2026-factory';
  const OPEN_ORDER_STATUSES = new Set(['new', 'confirmed', 'in_production', 'ready', 'partial_ship']);

  let shipDateCounts = {};

  function monthStart(date) {
    return new Date(date.getFullYear(), date.getMonth(), 1);
  }

  function addMonths(date, offset) {
    return new Date(date.getFullYear(), date.getMonth() + offset, 1);
  }

  function sameLocalDate(a, b) {
    return a.getFullYear() === b.getFullYear() &&
      a.getMonth() === b.getMonth() &&
      a.getDate() === b.getDate();
  }

  function dateKey(date) {
    return [
      date.getFullYear(),
      String(date.getMonth() + 1).padStart(2, '0'),
      String(date.getDate()).padStart(2, '0')
    ].join('-');
  }

  function renderMonth(date, today) {
    const first = monthStart(date);
    const daysInMonth = new Date(first.getFullYear(), first.getMonth() + 1, 0).getDate();
    const blanks = first.getDay();
    let html = `<div class="mini-calendar-month"><div class="mini-calendar-title">${MONTH_FMT.format(first)}</div>`;
    html += '<div class="mini-calendar-grid">';
    for (const label of DOW_LABELS) {
      html += `<div class="mini-calendar-dow">${label}</div>`;
    }
    for (let i = 0; i < blanks; i += 1) {
      html += '<div class="mini-calendar-day is-outside" aria-hidden="true"></div>';
    }
    for (let day = 1; day <= daysInMonth; day += 1) {
      const cellDate = new Date(first.getFullYear(), first.getMonth(), day);
      const key = dateKey(cellDate);
      const shipCount = shipDateCounts[key] || 0;
      const isToday = sameLocalDate(cellDate, today);
      const classes = [
        'mini-calendar-day',
        isToday ? 'is-today' : '',
        shipCount ? 'has-shipments' : ''
      ].filter(Boolean).join(' ');
      const title = shipCount ? ` title="${shipCount} open Sales Order${shipCount === 1 ? '' : 's'} ship by ${key}"` : '';
      html += `<div class="${classes}" data-date="${key}" data-ship-count="${shipCount}"${title}${isToday ? ' aria-current="date"' : ''}>${day}</div>`;
    }
    html += '</div></div>';
    return html;
  }

  function renderStrip(root, centerDate) {
    const today = new Date();
    const months = [addMonths(centerDate, -1), monthStart(centerDate), addMonths(centerDate, 1)];
    root.innerHTML = `
      <button type="button" class="mini-calendar-nav" data-calendar-nav="-1" aria-label="Previous month">&#8249;</button>
      <div class="mini-calendar-months" aria-label="Three month calendar">
        ${months.map(month => renderMonth(month, today)).join('')}
      </div>
      <button type="button" class="mini-calendar-nav" data-calendar-nav="1" aria-label="Next month">&#8250;</button>
    `;
  }

  function initMiniCalendar(root) {
    let centerDate = monthStart(new Date());
    root.classList.add('mini-calendar-strip');
    renderStrip(root, centerDate);
    root._miniCalendarRender = () => renderStrip(root, centerDate);
    root.addEventListener('click', event => {
      const btn = event.target.closest('[data-calendar-nav]');
      if (!btn) return;
      centerDate = addMonths(centerDate, Number(btn.dataset.calendarNav));
      renderStrip(root, centerDate);
    });
  }

  function setShipDateCounts(counts) {
    shipDateCounts = counts || {};
    document.querySelectorAll('[data-mini-calendar]').forEach(root => {
      if (typeof root._miniCalendarRender === 'function') root._miniCalendarRender();
    });
  }

  function buildShipDateCounts(orders) {
    const counts = {};
    for (const order of orders || []) {
      const shipDate = order && order.requested_ship_date;
      if (!shipDate || !OPEN_ORDER_STATUSES.has(order.status)) continue;
      counts[shipDate] = (counts[shipDate] || 0) + 1;
    }
    return counts;
  }

  async function fetchShipDateCounts() {
    try {
      const res = await fetch(SALES_API_BASE + '/sales/orders?limit=200', {
        headers: { 'X-API-Key': SALES_API_KEY }
      });
      if (!res.ok) return;
      const data = await res.json();
      setShipDateCounts(buildShipDateCounts(data.orders || []));
    } catch {
      // Calendar indicators are informational; leave the calendar usable if the API is unavailable.
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-mini-calendar]').forEach(initMiniCalendar);
    fetchShipDateCounts();
  });

  window.addEventListener(SHIP_DATES_EVENT, event => {
    setShipDateCounts(event.detail && event.detail.counts);
  });

  window.FactoryLedgerMiniCalendar = {
    setShipDateCounts,
    buildShipDateCounts
  };
})();
