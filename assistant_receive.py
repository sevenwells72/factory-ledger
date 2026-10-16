"""Keep explicitly supplied receiving lot identities in their separate fields."""
import re

from assistant_voice import normalize_voice_text


_LABEL = re.compile(
    r'(?P<internal>\b(?:FL|internal|system)\s+(?:lot|lote)\b(?:\s+(?:code|number))?'
    r'|\blote\s+(?:FL|interno|del\s+sistema)(?:\s+(?:n[uú]mero|c[oó]digo))?)'
    r'|(?P<supplier>\b(?:supplier|vendor|manufacturer)\s+lot\b(?:\s+(?:code|number))?'
    r'|\blote\s+(?:del?\s+proveedor)(?:\s+(?:n[uú]mero|c[oó]digo))?'
    r'|\b(?:lot|lote)\b(?:\s+(?:number|code|n[uú]mero|c[oó]digo))?)', re.I)
_NOT_CODES = {'unknown', 'unidentified', 'none', 'missing', 'not', 'given', 'available',
              'na', 'n/a', 'n.a', 'sin', 'ninguno', 'desconocido', 'pendiente',
              'no', 'the', 'el', 'la', 'bol', 'pol', 'number', 'code', 'numero',
              'número', 'codigo', 'código', 'is', 'es', 'supplier', 'proveedor',
              'on', 'if', 'check', 'needs', 'bag', 'label', 'tag', 'en', 'bolsa', 'etiqueta',
              'identity', 'checking', 'checked', 'verify', 'verification', 'identidad',
              'verificar', 'revisar', 'and', 'y', 'or', 'o', 'please', 'por', 'favor'}
_CLAUSE = re.compile(
    r'(?:^|\s+)(?:(?:and|y)\s+)?(?:BOL\b|POL\b|from\b|with\b|con\b|de\b|'
    r'(?:supplier|vendor|manufacturer|internal|system|FL)\s+lot\b|'
    r'lote\s+(?:del?\s+proveedor|interno|del\s+sistema|FL)\b)', re.I)
_PUNCTUATION = re.compile(r'[,;\n!?]|\.(?=\s|$)')


def _complete_code(text):
    """Accept a complete identifier span, never a prefix of prose or a list."""
    text = text.strip()
    if not text or len(text) > 200 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/ -]*', text):
        return None
    words = text.split()
    if any(word.casefold().rstrip('.') in _NOT_CODES for word in words):
        return None
    # A digit or explicit code punctuation distinguishes mixed-case IDs from
    # narrative. All-capital supplier codes remain valid without digits.
    if not (any(char.isdigit() for char in text) or re.search(r'[-_/]', text)
            or all(word.isupper() for word in words)):
        return None
    return text


def _code_after(text):
    text = re.sub(r'^\s*(?:[:#=]\s*)?(?:(?:is|es)\s+)?', '', text, flags=re.I)
    # Reuse the voice code parser with a synthetic label so Spanish supplier
    # phrases such as "lote del proveedor P uno cinco ocho cinco" also parse.
    normalized = normalize_voice_text('lot ' + text)[4:]
    punctuation = _PUNCTUATION.search(normalized)
    span = normalized[:punctuation.start()] if punctuation else normalized
    span = _CLAUSE.split(span, maxsplit=1)[0]
    code = _complete_code(span)
    if not code:
        return None
    if punctuation and punctuation[0] in ',;':
        # A comma before another bare identifier denotes multiple codes, not
        # permission to silently bind the first one. Named following fields
        # such as BOL or internal lot are separate, explicit clause boundaries.
        following = normalized[punctuation.end():].strip()
        next_span = _PUNCTUATION.split(following, maxsplit=1)[0]
        if _complete_code(_CLAUSE.split(next_span, maxsplit=1)[0]):
            return None
    return code


def lot_facts_from_text(text):
    """Read human lot labels; an ordinary receiving lot is the supplier's lot.

    Explicit FL/internal/system labels alone identify the internal lot field.
    Several supplier codes are left to the existing commingled receiving flow;
    this helper never chooses one of them or manufactures an internal code.
    """
    values = {'supplier_lot_code': [], 'lot_code': []}
    for label in _LABEL.finditer(text or ''):
        code = _code_after(text[label.end():])
        if code:
            field = 'lot_code' if label.group('internal') else 'supplier_lot_code'
            if code not in values[field]:
                values[field].append(code)
    return {field: codes[0] for field, codes in values.items() if len(codes) == 1}


def normalize_receive_arguments(arguments, facts):
    """Bind model arguments to the worker's explicit, separately labelled lots."""
    result = dict(arguments)
    supplier = facts.get('supplier_lot_code')
    internal = facts.get('lot_code')
    if supplier:
        result['supplier_lot_code'] = supplier
        if not internal:
            result.pop('lot_code', None)
    if internal:
        result['lot_code'] = internal
    return result
