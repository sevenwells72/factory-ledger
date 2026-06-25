(function() {
  'use strict';

  const MONTH_FMT = new Intl.DateTimeFormat(undefined, { month: 'short', year: 'numeric' });
  const DOW_LABELS = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];

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
      const isToday = sameLocalDate(cellDate, today);
      html += `<div class="mini-calendar-day${isToday ? ' is-today' : ''}"${isToday ? ' aria-current="date"' : ''}>${day}</div>`;
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
    root.addEventListener('click', event => {
      const btn = event.target.closest('[data-calendar-nav]');
      if (!btn) return;
      centerDate = addMonths(centerDate, Number(btn.dataset.calendarNav));
      renderStrip(root, centerDate);
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-mini-calendar]').forEach(initMiniCalendar);
  });
})();
