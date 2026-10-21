"""Keep explicitly supplied receiving lot identities in their separate fields."""
import re
from decimal import Decimal

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
_AMBIGUOUS = object()
_ABSENT_AFTER = re.compile(
    r'^\s*(?:[:#=]\s*)?(?:(?:is|es)\s+)?(?:unknown|unidentified|missing|none|N/A|NA|'
    r'not\s+(?:given|provided|available|known)|desconocid[oa]|ninguno|'
    r'sin\s+(?:n[uú]mero|c[oó]digo)|no\s+(?:se\s+(?:indic[oó]|dio|proporcion[oó])|disponible))'
    r'(?![A-Za-z0-9_/-])', re.I)
_NO_LOT_ON_LABEL = re.compile(
    r'\b(?:no\s+(?:(?:supplier|vendor|manufacturer)\s+)?lot(?:\s+(?:code|number))?'
    r'\s+on\s+(?:the\s+)?label'
    r'|(?:the\s+)?label\s+has\s+no\s+(?:(?:supplier|vendor|manufacturer)\s+)?lot(?:\s+(?:code|number))?'
    r'|(?:sin|no\s+hay)\s+(?:(?:n[uú]mero|c[oó]digo)\s+de\s+)?lote'
    r'(?:\s+del\s+proveedor)?\s+en\s+(?:la\s+)?etiqueta)\b', re.I)
_CLAUSE_BOUNDARY = re.compile(r'[;.!?\n]')
_LABEL_ASSERTION_PREFIX_GUARD = re.compile(
    r'\b(?:not|never|no|nunca|jam[aá]s|if|unless|whether|maybe|perhaps|assuming|suppose|'
    r'si|quiz[aá]s?|supongamos|means?|meaning|option|flag|checkbox|phrase|wording|'
    r'button|example|quote|instruction|asking|ask|select|choose|check|significa|'
    r'opci[oó]n|casilla|frase|texto|bot[oó]n|ejemplo|pregunt[ao]|seleccion[ae])\b'
    r'|\b\w+n[\'’]t\b', re.I)
_LABEL_SIMPLE_PREFIX = re.compile(
    r'\s*(?:(?:actually|in\s+fact|en\s+realidad|de\s+hecho)\s*[,:-]?\s*)?'
    r'(?:there\s+(?:is|was)\s+)?["\'“]?\s*', re.I)
_LABEL_RECEIVE_PREFIX = re.compile(
    r'\s*(?:(?:we|I)\s+)?(?:received|recibimos|recib[ií])\b[^;.!?\n]*,\s*'
    r'(?:there\s+(?:is|was)\s+)?["\'“]?\s*', re.I)
_LABEL_ASSERTION_END = re.compile(r'\s*["\'”]?\s*')
_LABEL_DENIAL_END = re.compile(
    r'\s*["\'”]?\s*(?:is\s+(?:not\s+(?:true|correct)|false|incorrect|wrong)'
    r'|isn[\'’]t\s+(?:true|correct)|es\s+(?:fals[oa]|incorrect[oa])'
    r'|no\s+es\s+(?:cierto|verdad|correct[oa]))\s*', re.I)


def _explicit_no_lot_on_label(text):
    """Read an exact label assertion or its explicit denial; other text is unknown."""
    result = None
    for phrase in _NO_LOT_ON_LABEL.finditer(text):
        before = _CLAUSE_BOUNDARY.split(text[:phrase.start()])[-1]
        tail = text[phrase.end():]
        boundary = _CLAUSE_BOUNDARY.search(tail)
        after = tail[:boundary.start()] if boundary else tail
        if '¿' in before or boundary and boundary[0] == '?':
            continue
        simple = _LABEL_SIMPLE_PREFIX.fullmatch(before)
        receiving = (_LABEL_RECEIVE_PREFIX.fullmatch(before)
                     and not _LABEL_ASSERTION_PREFIX_GUARD.search(before))
        if not (simple or receiving):
            continue
        if _LABEL_DENIAL_END.fullmatch(after):
            result = False
        elif _LABEL_ASSERTION_END.fullmatch(after):
            result = True
    return result


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
        choices = re.split(r'\s+(?:and|y|or|o|&)\s+', span, flags=re.I)
        codes = [_complete_code(choice) for choice in choices]
        if len(codes) < 2 or not all(codes):
            return None
        if len(set(codes)) != 1:
            return _AMBIGUOUS
        code = codes[0]
    if punctuation:
        # Any punctuation followed by another bare identifier is ambiguous,
        # including a period from voice transcription. A named BOL or other
        # field starts a separate clause, whose values do not belong to this lot.
        for following in _PUNCTUATION.split(normalized[punctuation.end():]):
            following = following.strip()
            if not following:
                continue
            clause = _CLAUSE.search(following)
            if clause and clause.start() == 0:
                break
            next_code = _complete_code(_CLAUSE.split(following, maxsplit=1)[0])
            if next_code and next_code != code:
                return _AMBIGUOUS
    return code


def _parsed_lot_facts(text):
    values = {'supplier_lot_code': [], 'lot_code': []}
    ambiguous = set()
    labels = list(_LABEL.finditer(text or ''))
    for index, label in enumerate(labels):
        field = 'lot_code' if label.group('internal') else 'supplier_lot_code'
        end = labels[index + 1].start() if index + 1 < len(labels) else len(text)
        span = text[label.end():end]
        if re.search(r'\b(?:no|sin)\s*$', text[:label.start()], re.I) or _ABSENT_AFTER.match(span):
            ambiguous.add(field)
            continue
        if index + 1 < len(labels):
            span = span.rstrip(' \t\r\n,;.!?')
            span = re.sub(r'\s+(?:and|y)$', '', span, flags=re.I).rstrip(' \t\r\n,;.!?')
        code = _code_after(span)
        if code is _AMBIGUOUS:
            ambiguous.add(field)
        elif code:
            if code not in values[field]:
                values[field].append(code)
    ambiguous.update(field for field, codes in values.items() if len(codes) > 1)
    return ({field: codes[0] for field, codes in values.items() if len(codes) == 1 and field not in ambiguous},
            ambiguous)


def lot_facts_from_text(text):
    """Read unambiguous human lot labels without choosing among supplier codes."""
    return _parsed_lot_facts(text)[0]


def merge_receive_lot_facts(previous_facts, text):
    """Persist unresolved-field tombstones; button turns preserve all facts."""
    no_label_lot = _explicit_no_lot_on_label(text or '')
    # The absence phrase is its own fact. Removing it before code parsing lets
    # an actual supplier code from the same message take precedence.
    facts, ambiguous = _parsed_lot_facts(_NO_LOT_ON_LABEL.sub(' ', text or ''))
    result = dict(previous_facts)
    for field in ambiguous:
        result[field] = None
    result.update(facts)
    if facts.get('supplier_lot_code'):
        result.pop('no_supplier_lot', None)
    elif no_label_lot is not None:
        # Only this explicit physical-label statement permits unidentified
        # receiving. Missing/unknown/unreadable is not the worker's assertion.
        result['supplier_lot_code'] = None
        if no_label_lot:
            result['no_supplier_lot'] = True
        else:
            result.pop('no_supplier_lot', None)
    return result


def normalize_receive_arguments(arguments, facts):
    """Bind model arguments to the worker's explicit, separately labelled lots."""
    result = dict(arguments)
    # The model may not invent the physical-label exception. Persist only the
    # worker's explicit statement, including through supplier-choice buttons.
    result.pop('no_supplier_lot', None)
    if facts.get('no_supplier_lot') is True and not facts.get('supplier_lot_code'):
        result['no_supplier_lot'] = True
    supplier = facts.get('supplier_lot_code')
    internal = facts.get('lot_code')
    for field in ('supplier_lot_code', 'lot_code'):
        if field in facts and facts[field] is None:
            result.pop(field, None)
    if supplier:
        result['supplier_lot_code'] = supplier
    if 'supplier_lot_code' in facts and not internal:
        result.pop('lot_code', None)
    if internal:
        result['lot_code'] = internal
    for field in ('cases', 'case_size_lb'):
        if facts.get(field) is not None:
            result[field] = facts[field]
    return result


def package_facts(text):
    """Multiply explicitly stated bag/case size, never mistake it for a total."""
    text = normalize_voice_text(text)
    count = re.search(r'(?<![\w.+-])(\d+)\s+(?:bags?|bolsas?|sacos?|cases?|cajas?)\b', text, re.I)
    if not count:
        return {}
    for weight in re.finditer(r'(?<![\w.+-])(\d+(?:\.\d+)?)\s*(?:lb|lbs|pounds?|libras?)\b', text, re.I):
        if weight.start() < count.end():
            continue
        before, after = text[count.end():weight.start()], text[weight.end():]
        if re.search(r'\btotal\s*[:=]?\s*$', before, re.I) or re.match(r'\s*(?:(?:in|en)\s+)?total\b', after, re.I):
            continue
        per_package = (re.match(r'\s*(?:each|cada\b|(?:per|por|/)\s*(?:bag|bolsa|saco|case|caja)|bags?\b|bolsas?\b|cases?\b)', after, re.I)
                       or re.search(r'\bde\s*$', before, re.I))
        if per_package:
            cases, size = int(count[1]), Decimal(weight[1])
            totals = re.findall(r'(?<![\w.])(\d+(?:\.\d+)?)\s*(?:lb|lbs|pounds?|libras?)\s*(?:(?:in|en)\s+)?total\b', text, re.I)
            return {'cases': cases, 'case_size_lb': float(size),
                    'quantity_lb': float(totals[0]) if len(totals) == 1 else float(cases * size)}
    return {}


def split_receive_items(text):
    """Retain every explicitly enumerated receive item, with shared delivery facts.

    Only split conjunctions introducing a new quantity and package unit. Product
    names such as Fruit and Nut remain intact; no catalog identity is inferred.
    """
    text = normalize_voice_text(text)
    parts = re.split(r'\s+(?:and|y|plus|m[aá]s)\s+(?=\d+\s+(?:cases?|cajas?|bags?|bolsas?|sacos?)\b)', text, flags=re.I)
    if len(parts) < 2:
        return []
    first = re.search(r'\d+\s+(?:cases?|cajas?|bags?|bolsas?|sacos?)\b', parts[0], re.I)
    if not first:
        return []
    prefix = parts[0][:first.start()]
    if not re.search(r'\b(?:receive[ds]?|recibimos|recib[ií]|recibir)\b', prefix, re.I):
        if prefix.strip():
            return []
        prefix = 'Received '
        parts[0] = prefix + parts[0].lstrip()
    # The final item usually carries shared supplier/BOL/package-weight facts.
    last = parts[-1]
    shared = re.search(r'\s+(?:from|del?\s+proveedor)\s+|,\s*(?=\d+(?:\.\d+)?\s*(?:lb|libras?)\b|BOL\b|supplier\s+lot\b|lote\b)', last, re.I)
    # Spanish supplier proper names following the product: "avena de Grain Supply".
    spanish_supplier = re.search(r'\s+de\s+(?=[A-ZÁÉÍÓÚ][a-záéíóú]+\s+[A-ZÁÉÍÓÚ])', last)
    if spanish_supplier and (not shared or spanish_supplier.start() < shared.start()):
        shared = spanish_supplier
    suffix = last[shared.start():] if shared else ''
    parts[-1] = last[:shared.start()] if shared else last
    # A lot stated after the final item belongs only to that item unless the
    # worker explicitly supplies it for another item. Do not share trace IDs.
    lot_label = _LABEL.search(suffix)
    common = suffix[:lot_label.start()].rstrip(' ,;') if lot_label else suffix
    return [parts[0].rstrip(' ,.') + common] + [
        prefix + part.strip(' ,.') + (suffix if index == len(parts)-1 else common)
        for index, part in enumerate(parts[1:], 1)]
