"""Conservative, catalog-independent normalization of phone dictation.

Only explicit product numbers, quantities, document labels and spoken lot codes
change. This never selects a catalog item, supplies a missing lot, or confirms a
lot tag; those decisions remain with the assistant's normal resolution flow.
"""
from __future__ import annotations

import re
import unicodedata


_EN_SMALL = dict(zip(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(),
    range(20),
))
_EN_TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
_ES_SMALL = dict(zip(
    'cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciseis diecisiete dieciocho diecinueve veinte veintiuno veintidos veintitres veinticuatro veinticinco veintiseis veintisiete veintiocho veintinueve'.split(),
    range(30),
))
_ES_SMALL.update(un=1, una=1)
_ES_TENS = dict(zip('treinta cuarenta cincuenta sesenta setenta ochenta noventa'.split(), range(30, 100, 10)))
_ES_HUNDREDS = dict(zip('doscientos trescientos cuatrocientos quinientos seiscientos setecientos ochocientos novecientos'.split(), range(200, 1000, 100)))
# Include accented spellings in the recognizer while comparing folded tokens.
_NUMBER_WORDS = set(_EN_SMALL) | set(_EN_TENS) | set(_ES_SMALL) | set(_ES_TENS) | set(_ES_HUNDREDS) | {
    'hundred', 'thousand', 'million', 'cien', 'ciento', 'mil', 'millon', 'millones',
    'dieciséis', 'veintiún', 'veintiun', 'veintidós', 'veintitrés', 'veintiséis',
    'millón',
}
_JOINERS = {'and', 'y', 'point', 'punto', 'coma'}
_WORD = '(?:' + '|'.join(sorted(_NUMBER_WORDS, key=lambda w: (-len(w), w))) + ')'
_PART = '(?:' + '|'.join(sorted(_NUMBER_WORDS | _JOINERS, key=lambda w: (-len(w), w))) + ')'
_DIGIT_QUANTITY = r'(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)'
_NUMBER_PHRASE = rf'(?:{_DIGIT_QUANTITY}|{_WORD}(?:[\s-]+{_PART})*)'
_CODE_PHRASE = rf'(?:\d+|{_WORD}|oh)(?:[\s-]+(?:\d+|{_PART}|oh))*'


def _fold(value: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', value.lower()) if not unicodedata.combining(c))


def _tokens(value: str) -> list[str]:
    return _fold(value).replace('-', ' ').split()


def _under_hundred(tokens: list[str], language: str) -> int | None:
    small, tens = (_EN_SMALL, _EN_TENS) if language == 'en' else (_ES_SMALL, _ES_TENS)
    if len(tokens) == 1:
        token = tokens[0]
        if token == 'veintiun' and language == 'es':
            return 21
        return small.get(token, tens.get(token))
    if language == 'es' and len(tokens) == 3 and tokens[1] == 'y':
        tokens = [tokens[0], tokens[2]]
    if len(tokens) == 2 and tokens[0] in tens and 0 < small.get(tokens[1], 100) < 10:
        return tens[tokens[0]] + small[tokens[1]]
    return None


def _under_thousand(tokens: list[str], language: str) -> int | None:
    if not tokens:
        return 0
    if language == 'en' and len(tokens) >= 2 and tokens[1] == 'hundred':
        hundreds = _EN_SMALL.get(tokens[0], 0)
        if not 0 < hundreds < 10:
            return None
        tail = tokens[2:]
        if tail[:1] == ['and']:
            tail = tail[1:]
            if not tail:
                return None
        remainder = _under_hundred(tail, language) if tail else 0
        return hundreds * 100 + remainder if remainder is not None else None
    if language == 'es' and tokens[0] in {'cien', 'ciento', *_ES_HUNDREDS}:
        if tokens[0] == 'cien' and len(tokens) > 1:
            return None
        hundreds = _ES_HUNDREDS.get(tokens[0], 100)
        remainder = _under_hundred(tokens[1:], language) if len(tokens) > 1 else 0
        return hundreds + remainder if remainder is not None else None
    return _under_hundred(tokens, language)


def _cardinal(tokens: list[str]) -> int | None:
    if len(tokens) == 1 and tokens[0].isdigit():
        return int(tokens[0])
    for language, thousand in [('en', 'thousand'), ('es', 'mil')]:
        if thousand in tokens:
            if tokens.count(thousand) != 1:
                continue
            position = tokens.index(thousand)
            head = _under_thousand(tokens[:position], language) if position else (1 if language == 'es' else None)
            tail = tokens[position + 1:]
            if language == 'en' and tail[:1] == ['and']:
                tail = tail[1:]
                if not tail:
                    continue
            remainder = _under_thousand(tail, language)
            if head is not None and head > 0 and remainder is not None:
                return head * 1000 + remainder
        else:
            value = _under_thousand(tokens, language)
            if value is not None:
                return value
    return None


def _quantity(value: str) -> str | None:
    if re.fullmatch(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?', value):
        return value.replace(',', '')
    if re.fullmatch(r'\d+(?:[.,]\d+)?', value):
        return value.replace(',', '.')
    tokens = _tokens(value)
    separators = [i for i, token in enumerate(tokens) if token in {'point', 'punto', 'coma'}]
    if separators:
        if len(separators) != 1:
            return None
        position = separators[0]
        whole = _cardinal(tokens[:position])
        decimal = _code_number(tokens[position + 1:])
        if whole is None or decimal is None:
            return None
        return f'{whole}.{decimal}'
    value = _cardinal(tokens)
    return str(value) if value is not None else None


def _code_number(tokens: list[str]) -> str | None:
    """Join explicitly spoken groups: one/two -> 12; fifteen/eighty-five -> 1585."""
    if not tokens:
        return None
    pieces = []
    while tokens:
        if tokens[0].isdigit():
            pieces.append(tokens[0])  # Preserve significant leading zeroes.
            tokens = tokens[1:]
            continue
        if tokens[0] == 'oh':
            pieces.append('0')
            tokens = tokens[1:]
            continue
        for end in range(len(tokens), 0, -1):
            value = _cardinal(tokens[:end])
            if value is not None:
                pieces.append(str(value))
                tokens = tokens[end:]
                break
        else:
            return None
    return ''.join(pieces)


_UNITS = {
    **dict.fromkeys(('lb', 'lbs', 'pound', 'pounds', 'libra', 'libras'), 'lb'),
    **dict.fromkeys(('case', 'cases', 'caja', 'cajas'), 'cases'),
    **dict.fromkeys(('bag', 'bags', 'bolsa', 'bolsas', 'saco', 'sacos'), 'bags'),
    **dict.fromkeys(('batch', 'batches', 'tanda', 'tandas'), 'batches'),
}
_QUANTITY_RE = re.compile(rf'(?<![\w.,])(?P<number>{_NUMBER_PHRASE})\s*(?P<unit>' + '|'.join(sorted(_UNITS, key=lambda w: -len(w))) + r')\b', re.I)
_PRODUCT_RE = re.compile(
    r'\b(?P<family>classic|cl[aá]sic[oa]|s\.?\s*s\.?|sunshine)\s*'
    r'(?:(?:original)\s+)?(?:(?:number|n[uú]mero|no\.?|n[uú]m\.?)\s*)?(?:#\s*)?'
    r'(?P<number>one|two|three|nine|uno|un|dos|tres|nueve|[1239])\b'
    rf'(?![\s-]+(?:{_WORD}|point|punto|coma)\b)', re.I,
)
_BOL_RE = re.compile(
    r'\b(?:BOL|POL|B\s*\.\s*O\s*\.\s*L\.?|B\s+O\s+L|bowl)\b\.?'
    rf'(?=\s+(?:[a-z]\s+)?(?:\d+|{_WORD})\b)', re.I,
)
_CODE_RE = re.compile(
    rf'\b(?P<label>lot(?:\s+number)?|lote(?:\s+n[uú]mero)?|BOL)\s+(?:#\s*)?'
    rf'(?P<prefix>[a-z]\s+)?(?P<number>{_CODE_PHRASE})\b', re.I,
)
_LETTER_CODE_RE = re.compile(rf'\b(?P<prefix>[a-z])\s+(?P<number>{_CODE_PHRASE})\b', re.I)



def normalize_order_code(value: str) -> str:
    """Canonicalize only a complete explicit SO spelling, retaining its zeros."""
    words = r'(?:[0-9]+|zero|oh|one|two|three|four|five|six|seven|eight|nine|cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)'
    # A greedy word pattern cannot determine the date/sequence boundary in an
    # all-digit dictation without dash; parse the complete explicit SO span.
    whole = re.compile(r'(?<!\w)(?:s\s*\.?\s*o\.?|ese\s+o)[\s-]+'
        r'(?P<body>(?:' + words + r'|dash|guion|guión)(?:[\s-]+(?:' + words + r'|dash|guion|guión))*)(?!\w)', re.I)
    digits = {**{key:str(v) for key,v in _EN_SMALL.items() if v < 10},
              **{key:str(v) for key,v in _ES_SMALL.items() if v < 10}, 'oh':'0'}
    def convert(match):
        # Typed separators are opaque. Only the documented six-digit SO date
        # and sequence may retain their existing hyphen in this helper.
        if '-' in match['body'] and not re.search(r'[a-z]', match['body'], re.I):
            if not re.fullmatch(r'\d{6}-\d{3,}', match['body']):
                return match[0]
        tokens = _tokens(match['body'])
        separators=[i for i,t in enumerate(tokens) if t in {'dash','guion'}]
        if separators:
            if len(separators)!=1: return match[0]
            at=separators[0]
            groups=[tokens[:at],tokens[at+1:]]
        elif len(tokens)==2 and all(t.isdigit() for t in tokens):
            groups=[[tokens[0]],[tokens[1]]]
        else:
            raw=''.join(t if t.isdigit() else digits.get(t,'?') for t in tokens)
            if '?' in raw or len(raw)<9: return match[0]
            groups=[[raw[:6]],[raw[6:]]]
        parts=[''.join(t if t.isdigit() else digits.get(t,'?') for t in group) for group in groups]
        if '?' in ''.join(parts) or len(parts[0])!=6 or len(parts[1])<3: return match[0]
        return 'SO-' + parts[0] + '-' + parts[1]
    # Keep a following shipment quantity outside the identifier. Look for
    # parseable quantity/unit spans at every word boundary so a spoken code's
    # number words cannot greedily hide a later 'ten cases' / 'diez cajas'.
    quantities = []
    end = 0
    for start in re.finditer(r'(?<!\w)(?=\w)', value):
        match = _QUANTITY_RE.match(value, start.start())
        if match and match.start() >= end and _quantity(match['number']) is not None:
            quantities.append((match.start(), match.end(), match[0]))
            end = match.end()
    for index, (start, end, original) in reversed(list(enumerate(quantities))):
        value = value[:start] + f'\ue010{chr(0xe200 + index)}\ue011' + value[end:]
    value = whole.sub(convert, value)
    for index, (_, _, original) in enumerate(quantities):
        value = value.replace(f'\ue010{chr(0xe200 + index)}\ue011', original)
    return value


def normalize_voice_text(text: str, *, protected_codes=()) -> str:
    """Return the same utterance with explicit voice spellings canonicalized.

    The original text should still be retained for the visible transcript/audit
    history. This value is intended for parsing and tool queries, not permission
    or traceability decisions.
    """
    # Typed identifiers are opaque: separators and leading zeroes are identity.
    # Protect even near-code spellings before quantity/product normalization.
    protected = []
    def protect(match):
        protected.append(match[0])
        return f"\ue000{chr(0xe100 + len(protected) - 1)}\ue001"
    for literal in sorted(set(protected_codes), key=len, reverse=True):
        if literal:
            text = re.sub(re.escape(literal), protect, text, flags=re.I)
    text = re.sub(r'(?<!\w)(?=[\w/-]*\d)[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)+(?!\w)', protect, text)

    def product(match: re.Match) -> str:
        family = _fold(match['family']).replace('.', '').replace(' ', '')
        number = _cardinal(_tokens(match['number']))
        if family in {'classic', 'clasico', 'clasica'} and number == 9:
            return 'Classic #9'
        if family in {'ss', 'sunshine'} and number in {1, 2, 3}:
            return f'SS #{number}'
        return match[0]

    def quantity(match: re.Match) -> str:
        if re.match(r'\s*(?:,\s*)?(?:suppl(?:y|ies)|inc(?:orporated)?|llc|company|bakery|depot)\b', match.string[match.end():], re.I):
            return match[0]
        number = _quantity(match['number'])
        if number is None:
            return match[0]
        unit = _UNITS[match['unit'].lower()]
        if number == '1':
            unit = {'bags': 'bag', 'cases': 'case', 'batches': 'batch'}.get(unit, unit)
        return f'{number} {unit}'

    def code(match: re.Match) -> str:
        prefix = (match['prefix'] or '').strip().upper()
        # Outside a lot/BOL phrase, articles and conjunctions are not code letters.
        if 'label' not in match.re.groupindex and prefix in {'A', 'I', 'Y', 'O', 'E'}:
            return match[0]
        spoken = match['number']
        if not any(token in _NUMBER_WORDS or token == 'oh' for token in _tokens(spoken)):
            return match[0]
        suffix = ''
        # A following clause is not part of the tag: retain its conjunction.
        trailing = re.search(r'\s+(?:and|y)$', spoken, re.I)
        if trailing:
            suffix, spoken = trailing[0], spoken[:trailing.start()]
        number = _code_number(_tokens(spoken))
        if number is None and re.match(r'\s*(?:' + '|'.join(_UNITS) + r')\b', match.string[match.end():], re.I):
            # 'lot one two three four and five bags' names a tag and a
            # separate quantity. A conjunction inside 'hundred and two' or
            # 'ochenta y cinco' is kept because the full code already parses.
            for joiner in reversed(list(re.finditer(r'\s+(?:and|y)\s+', spoken, re.I))):
                if _quantity(spoken[joiner.end():]) is not None:
                    head = _code_number(_tokens(spoken[:joiner.start()]))
                    if head is not None:
                        number, suffix = head, spoken[joiner.start():] + suffix
                        break
        if number is None:
            return match[0]
        label = match.groupdict().get('label')
        return (label + ' ' if label else '') + prefix + number + suffix

    text = normalize_order_code(text)
    text = _PRODUCT_RE.sub(product, text)
    text = _BOL_RE.sub('BOL', text)
    text = _CODE_RE.sub(code, text)
    # A party name can contain a letter and number word ('B One Bakery').
    # Outside explicit lot/BOL spans, normalize only an isolated code query.
    isolated_code = _LETTER_CODE_RE.fullmatch(text.strip())
    if isolated_code:
        start = len(text) - len(text.lstrip())
        text = text[:start] + code(isolated_code) + text[start + len(isolated_code[0]):]
    text = _QUANTITY_RE.sub(quantity, text)
    # An isolated product-number answer may use grouped digits ("two eighty
    # three" = 283), just as a labelled lot does. Never rewrite prose or names.
    isolated_number = re.fullmatch(_CODE_PHRASE, text.strip(), re.I)
    if isolated_number and any(word in _NUMBER_WORDS for word in _tokens(text)):
        value = _code_number(_tokens(text))
        if value:
            text = value
    text = re.sub(r'\b(product(?:\s+number)?|producto(?:\s+n[uú]mero)?|SKU)\s+(' + _CODE_PHRASE + r')\b',
        lambda m: m[1] + ' ' + (_code_number(_tokens(m[2])) or m[2]), text, flags=re.I)
    for index, literal in enumerate(protected):
        text = text.replace(f"\ue000{chr(0xe100 + index)}\ue001", literal)
    return text
