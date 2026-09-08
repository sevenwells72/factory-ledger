import {CHECK_SOURCE as BASE_CHECK_SOURCE} from '../lib/checks.mjs';
// Include native selects and JS-property/SVG click targets omitted by the older fixture helper.
export const CHECK_SOURCE=BASE_CHECK_SOURCE.replace('function isVisible(el) {', 'function isVisible(el) { if (el.checkVisibility && !el.checkVisibility()) return false;').replace("el.tagName === 'OPTION' || el.closest('select')","el.tagName === 'OPTION'").replace("'.cell', '.copyday'","'.lot-pill', 'g[cursor=pointer]', '[data-audit-click]', '.cell', '.copyday'");
