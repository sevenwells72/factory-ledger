(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.PalletCalculations = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const CASES_PER_PALLET_BY_CASE_SIZE_LB = Object.freeze({
    10: 140,
    25: 60
  });

  function finiteNumber(value) {
    if (value == null || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function inferCaseSizeLb(item) {
    if (!item) return null;
    for (const field of ['case_size_lb', 'case_weight_lb', 'default_case_weight_lb']) {
      const size = finiteNumber(item[field]);
      if (size === 10 || size === 25) return size;
    }

    const text = [
      item.product,
      item.product_name,
      item.name,
      item.description,
      item.uom,
      item.sku,
      item.odoo_code
    ].filter(Boolean).join(' ');
    const match = text.match(/\b(10|25)\s*(?:-|\u2013|\u2014)?\s*lb\b/i);
    return match ? Number(match[1]) : null;
  }

  function inferCasesPerPallet(item) {
    return CASES_PER_PALLET_BY_CASE_SIZE_LB[inferCaseSizeLb(item)] || null;
  }

  function roundedForDisplay(value) {
    return Math.round((value + Number.EPSILON) * 10) / 10;
  }

  function formatNumber(value) {
    const rounded = roundedForDisplay(value);
    return rounded.toLocaleString('en-US', { maximumFractionDigits: 1 });
  }

  function formatPalletCount(value) {
    const rounded = roundedForDisplay(value);
    const label = rounded > 0 && rounded <= 1 ? 'pallet' : 'pallets';
    return `${formatNumber(rounded)} ${label}`;
  }

  function calculateLinePallets(item, caseQuantity) {
    const casesPerPallet = inferCasesPerPallet(item);
    const cases = finiteNumber(caseQuantity);
    if (!casesPerPallet || cases == null || cases < 0) {
      return {
        casesPerPallet: casesPerPallet || null,
        calculatedPallets: null,
        physicalPalletsRoundedUp: null,
        display: '\u2014'
      };
    }

    const calculatedPallets = cases / casesPerPallet;
    return {
      casesPerPallet,
      calculatedPallets,
      physicalPalletsRoundedUp: Math.ceil(calculatedPallets),
      display: formatPalletCount(calculatedPallets)
    };
  }

  function calculateOrderPallets(lines, quantitySelector) {
    const eligibleLines = (lines || []).filter(line => line && !line.is_non_weight && !line.is_service);
    let calculatedPallets = 0;
    let mappedLineCount = 0;
    let unknownLineCount = 0;

    for (const line of eligibleLines) {
      const quantity = typeof quantitySelector === 'function'
        ? quantitySelector(line)
        : line[quantitySelector || 'unit_count'];
      const cases = finiteNumber(quantity);
      if (cases == null) {
        unknownLineCount += 1;
        continue;
      }
      const calculation = calculateLinePallets(line, cases);
      if (calculation.calculatedPallets == null) {
        if (cases > 0) unknownLineCount += 1;
        continue;
      }
      calculatedPallets += calculation.calculatedPallets;
      mappedLineCount += 1;
    }

    if (!mappedLineCount || unknownLineCount) {
      return {
        casesPerPallet: null,
        calculatedPallets: null,
        physicalPalletsRoundedUp: null,
        display: '\u2014',
        mappedLineCount,
        unknownLineCount
      };
    }

    const physicalPalletsRoundedUp = Math.ceil(calculatedPallets);
    const isMixed = mappedLineCount > 1;
    const hasFraction = Math.abs(calculatedPallets - Math.round(calculatedPallets)) > 1e-9;
    let display = formatPalletCount(calculatedPallets);
    if (isMixed && hasFraction) {
      display += ` / ${physicalPalletsRoundedUp} physical mixed ${physicalPalletsRoundedUp === 1 ? 'pallet' : 'pallets'}`;
    }

    return {
      casesPerPallet: null,
      calculatedPallets,
      physicalPalletsRoundedUp,
      display,
      mappedLineCount,
      unknownLineCount
    };
  }

  return {
    CASES_PER_PALLET_BY_CASE_SIZE_LB,
    inferCaseSizeLb,
    inferCasesPerPallet,
    calculateLinePallets,
    calculateOrderPallets,
    formatPalletCount
  };
}));
