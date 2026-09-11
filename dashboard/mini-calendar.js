(function() {
  'use strict';

  const MONTH_FMT = new Intl.DateTimeFormat(undefined, { month: 'short', year: 'numeric' });
  const DOW_LABELS = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
  const SHIP_DATES_EVENT = 'factory-ledger:ship-dates';
  const SALES_API_BASE = 'https://fastapi-production-b73a.up.railway.app';
  const SALES_API_KEY = 'dashboard-key-2026';

  // IMP-069: below 520px the three-month strip collapses to one row — today's
  // date with a disclosure that expands the current month inline, one month
  // at a time, with tappable --hit-min day cells. Above 520px nothing here
  // changes: renderStrip() below is the desktop form, byte for byte.
  const COMPACT_QUERY = typeof window.matchMedia === 'function' ? window.matchMedia('(max-width: 520px)') : null;
  const TODAY_FMT = new Intl.DateTimeFormat(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
  const LONG_DAY_FMT = new Intl.DateTimeFormat(undefined, { weekday: 'long', month: 'long', day: 'numeric' });
  const DOW_LONG = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  let rootCounter = 0;

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

  function isCompact() {
    return Boolean(COMPACT_QUERY && COMPACT_QUERY.matches);
  }

  function shipPhrase(count) {
    if (!count) return 'no open Sales Orders ship by this date';
    return count + ' open Sales Order' + (count === 1 ? '' : 's') + ' ship by this date';
  }

  function renderCompactMonth(date, today, ui) {
    const first = monthStart(date);
    const daysInMonth = new Date(first.getFullYear(), first.getMonth() + 1, 0).getDate();
    const blanks = first.getDay();
    let html = '<div class="mini-calendar-grid" role="group" aria-label="' + MONTH_FMT.format(first) + '">';
    DOW_LABELS.forEach((label, i) => {
      html += `<div class="mini-calendar-dow" aria-hidden="true" title="${DOW_LONG[i]}">${label}</div>`;
    });
    for (let i = 0; i < blanks; i += 1) {
      html += '<div class="mini-calendar-day is-outside" aria-hidden="true"></div>';
    }
    for (let day = 1; day <= daysInMonth; day += 1) {
      const cellDate = new Date(first.getFullYear(), first.getMonth(), day);
      const key = dateKey(cellDate);
      const shipCount = shipDateCounts[key] || 0;
      const isToday = sameLocalDate(cellDate, today);
      const isSelected = ui.selectedKey === key;
      const classes = [
        'mini-calendar-day',
        isToday ? 'is-today' : '',
        shipCount ? 'has-shipments' : ''
      ].filter(Boolean).join(' ');
      // A real button: the desktop cells are display-only, but at 44px a cell
      // that looks tappable has to do something. A tap reads the day's count
      // into the caption below — the title tooltip the desktop cells carry does
      // not exist on touch.
      const label = LONG_DAY_FMT.format(cellDate) + (isToday ? ', today' : '') + ' — ' + shipPhrase(shipCount);
      html += `<button type="button" class="${classes}" data-date="${key}" data-ship-count="${shipCount}" aria-label="${label}" aria-pressed="${isSelected}"${isToday ? ' aria-current="date"' : ''}>${day}</button>`;
    }
    html += '</div>';
    return html;
  }

  function renderCompact(root, centerDate, ui) {
    const today = new Date();
    const todayKey = dateKey(today);
    const todayCount = shipDateCounts[todayKey] || 0;
    const panelId = root.id + '-panel';
    const month = monthStart(centerDate);
    let caption;
    if (ui.selectedKey) {
      const sel = ui.selectedKey.split('-').map(Number);
      const selDate = new Date(sel[0], sel[1] - 1, sel[2]);
      caption = TODAY_FMT.format(selDate) + ' — ' + shipPhrase(shipDateCounts[ui.selectedKey] || 0);
    } else {
      caption = 'Dots mark days with open ship dates. Tap a day for details.';
    }
    root.innerHTML = `
      <button type="button" class="mini-calendar-toggle" aria-expanded="${ui.expanded}" aria-controls="${panelId}">
        <span class="mini-calendar-today"><span class="mini-calendar-today-label">Today</span> ${TODAY_FMT.format(today)}</span>
        ${todayCount ? `<span class="mini-calendar-today-count">${todayCount} SO${todayCount === 1 ? '' : 's'} ship today</span>` : ''}
        <span class="mini-calendar-chevron" aria-hidden="true">&#9662;</span>
      </button>
      <div class="mini-calendar-panel" id="${panelId}"${ui.expanded ? '' : ' hidden'}>
        <div class="mini-calendar-panel-head">
          <button type="button" class="mini-calendar-nav" data-calendar-nav="-1" aria-label="Previous month">&#8249;</button>
          <div class="mini-calendar-title" aria-live="polite">${MONTH_FMT.format(month)}</div>
          <button type="button" class="mini-calendar-nav" data-calendar-nav="1" aria-label="Next month">&#8250;</button>
        </div>
        ${renderCompactMonth(month, today, ui)}
        <p class="mini-calendar-caption" aria-live="polite">${caption}</p>
      </div>
    `;
  }

  function initMiniCalendar(root) {
    let centerDate = monthStart(new Date());
    // Compact-only UI state; survives the re-render a ship-count refresh
    // triggers so an open month does not snap shut under the user's thumb.
    const ui = { expanded: false, selectedKey: null };
    if (!root.id) root.id = 'mini-calendar-' + (++rootCounter);
    root.classList.add('mini-calendar-strip');

    function render() {
      root.classList.toggle('is-compact', isCompact());
      if (isCompact()) renderCompact(root, centerDate, ui);
      else renderStrip(root, centerDate);
    }

    render();
    root._miniCalendarRender = render;

    root.addEventListener('click', event => {
      const nav = event.target.closest('[data-calendar-nav]');
      if (nav) {
        centerDate = addMonths(centerDate, Number(nav.dataset.calendarNav));
        render();
        return;
      }
      if (!isCompact()) return;
      const toggle = event.target.closest('.mini-calendar-toggle');
      if (toggle) {
        ui.expanded = !ui.expanded;
        render();
        // The render replaced the toggle; put focus back on its successor.
        const btn = root.querySelector('.mini-calendar-toggle');
        if (btn) btn.focus();
        return;
      }
      const day = event.target.closest('.mini-calendar-day[data-date]');
      if (day) {
        ui.selectedKey = ui.selectedKey === day.dataset.date ? null : day.dataset.date;
        render();
        const again = root.querySelector(`.mini-calendar-day[data-date="${day.dataset.date}"]`);
        if (again) again.focus();
      }
    });

    if (COMPACT_QUERY) {
      const onChange = () => {
        // Crossing the breakpoint changes the markup, not the month or the
        // open state — those belong to the user.
        render();
      };
      if (typeof COMPACT_QUERY.addEventListener === 'function') COMPACT_QUERY.addEventListener('change', onChange);
      else if (typeof COMPACT_QUERY.addListener === 'function') COMPACT_QUERY.addListener(onChange);
    }
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
      if (!shipDate || order.state !== 'open') continue;
      counts[shipDate] = (counts[shipDate] || 0) + 1;
    }
    return counts;
  }

  async function fetchShipDateCounts() {
    try {
      const res = await FL.fetchWithTimeout(SALES_API_BASE + '/sales/orders?limit=200', {
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
