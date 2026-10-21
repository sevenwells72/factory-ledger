// Product family (coconut vs granola) for the Ingredients relationship map.
//
// Factory Ledger has no reliable family flag on products: products.product_category
// is null on every batch row, and the backend itself classifies by name
// (main.py _production_family_from_name / _matrix_family). So this list is a
// PROPOSAL. The relationship map places nothing until Michael approves it and
// `approved` is set to true. Until then every ingredient sits in the Unplaced tray.
//
// How the proposal was made (2026-10-21, staging GET /bom/products?product_type=batch,
// 25 active batch products, STG test products excluded):
//   name contains "coconut"  -> coconut
//   name contains "granola"  -> granola
//   anything else            -> not classified (its use counts as "unclassified")
// Keyed by product id (stable), with the name for review.
window.FL_FAMILIES = {
  approved: false,
  proposedOn: '2026-10-21',
  products: {
    125: ['coconut', 'Batch Coconut Sweetened Fancy'],
    126: ['coconut', 'Batch Coconut Sweetened Flake'],
    127: ['coconut', 'Batch Coconut Sweetened Medium'],
    128: ['coconut', 'Batch Coconut Toasted Sweetened Flake'],
    123: ['granola', 'Batch BS Almond Butter Granola 350'],
    121: ['granola', 'Batch BS Dark Chocolate Granola 350'],
    124: ['granola', 'Batch BS Hazelnut Butter Granola 350'],
    122: ['granola', 'Batch BS Peanut Butter Banana Granola'],
    108: ['granola', 'Batch Classic Chocolate Chip Granola #9'],
    284: ['granola', 'Batch Classic Chocolate Chip Granola #9 — EXTRA-KOSHER (SS)'],
    107: ['granola', 'Batch Classic Granola #9'],
    283: ['granola', 'Batch Classic Granola #9 — EXTRA-KOSHER (SS)'],
    113: ['granola', 'Batch Granola Vanilla Almond 380 lb'],
    109: ['granola', 'Batch Setton Cinnamon Almond Granola #14'],
    111: ['granola', 'Batch Setton Cocoa Crunch Granola #13'],
    110: ['granola', 'Batch Setton Coffee Granola #12'],
    114: ['granola', 'Batch SS Chocolate Chip Granola #2'],
    115: ['granola', 'Batch SS Chocolate Chip Granola #5'],
    118: ['granola', 'Batch SS Cranberry Granola #3'],
    119: ['granola', 'Batch SS Low Carb Chocolate Chip Granola #8'],
    120: ['granola', 'Batch SS Low Carb Original Granola #7'],
    116: ['granola', 'Batch SS Original Granola #1'],
    117: ['granola', 'Batch SS Original Granola #4'],
    112: ['granola', 'Batch Vanilla Crisp Granola #16(no almonds)'],
    179: ['granola', 'Granola Fruit Nut Batch']
  }
};
