"""Round C phone dictation: explicit speech becomes safe parsing text."""
import pytest

from assistant_voice import normalize_voice_text


@pytest.mark.parametrize('text, expected', [
    ('How much classic number nine do we have?', 'How much Classic #9 do we have?'),
    ('Made two batches of classic number nine.', 'Made 2 batches of Classic #9.'),
    ('Made one batch of SS number one.', 'Made 1 batch of SS #1.'),
    ('We received four bags of sunflower seeds, 50 pounds each, from Green Supply, POL 5521.',
     'We received 4 bags of sunflower seeds, 50 lb each, from Green Supply, BOL 5521.'),
    ('Packed 40 cases of fruit and nut granola from batch lot one two three four.',
     'Packed 40 cases of fruit and nut granola from batch lot 1234.'),
    ('New order from fake bakery. Five cases of rainbow sprinkles.',
     'New order from fake bakery. 5 cases of rainbow sprinkles.'),
    ("Open today's entries. Is everything there correct?", "Open today's entries. Is everything there correct?"),
    ('¿Cuánto classic número nueve tenemos?', '¿Cuánto Classic #9 tenemos?'),
    ('Hicimos dos tandas de classic número nueve.', 'Hicimos 2 batches de Classic #9.'),
    ('Hicimos una tanda de SS número uno.', 'Hicimos 1 batch de SS #1.'),
    ('Recibimos cuatro bolsas de semillas de girasol, cincuenta libras cada una, de Green Supply, B O L 5521.',
     'Recibimos 4 bags de semillas de girasol, 50 lb cada una, de Green Supply, BOL 5521.'),
    ('Empacamos cuarenta cajas de granola de frutas y nueces del lote uno dos tres cuatro.',
     'Empacamos 40 cases de granola de frutas y nueces del lote 1234.'),
    ('Pedido nuevo de fake bakery. Cinco cajas de rainbow sprinkles.',
     'Pedido nuevo de fake bakery. 5 cases de rainbow sprinkles.'),
    ('¿Qué registré hoy?', '¿Qué registré hoy?'),
])
def test_real_phone_transcripts_and_spanish_equivalents(text, expected):
    assert normalize_voice_text(text) == expected


@pytest.mark.parametrize('phrase, expected', [
    ('classic number nine', 'Classic #9'), ('classic no. 9', 'Classic #9'),
    ('classic no 9', 'Classic #9'), ('classic nine', 'Classic #9'),
    ('Classic #9', 'Classic #9'), ('Classic número nueve', 'Classic #9'),
    ('Clásico nueve', 'Classic #9'),
    ('granola clásica número nueve', 'granola Classic #9'),
    ('granola clásica nueve', 'granola Classic #9'), ('SS number one', 'SS #1'),
    ('SS number two', 'SS #2'), ('SS number three', 'SS #3'),
    ('Sunshine number one', 'SS #1'), ('Sunshine número dos', 'SS #2'),
    ('SS número uno', 'SS #1'), ('SS número tres', 'SS #3'),
    ('S. S. number two', 'SS #2'), ('SS no. 1', 'SS #1'),
])
def test_supported_product_number_names(phrase, expected):
    assert normalize_voice_text(phrase) == expected


@pytest.mark.parametrize('phrase, expected', [
    ('four bags', '4 bags'), ('fifty pounds', '50 lb'), ('two batches', '2 batches'),
    ('forty cases', '40 cases'), ('four lbs', '4 lb'), ('50 pounds', '50 lb'), ('50lbs', '50 lb'),
    ('one hundred and twenty-five pounds', '125 lb'), ('two thousand five cases', '2005 cases'),
    ('twenty-one cases', '21 cases'), ('two point five pounds', '2.5 lb'),
    ('cuatro bolsas', '4 bags'), ('cincuenta libras', '50 lb'), ('cuarenta cajas', '40 cases'),
    ('veintidós cajas', '22 cases'), ('veintitrés cajas', '23 cases'),
    ('ciento cincuenta libras', '150 lb'), ('doscientos cuarenta y cinco cajas', '245 cases'),
    ('mil quinientos libras', '1500 lb'), ('dos mil cuarenta y cinco cajas', '2045 cases'),
    ('dos coma cinco libras', '2.5 lb'), ('1,5 libras', '1.5 lb'),
    ('1,000 pounds', '1000 lb'), ('1,000,000 pounds', '1000000 lb'),
    ('1,234,567.89 pounds', '1234567.89 lb'), ('10lb', '10 lb'),
    ('una bolsa', '1 bag'), ('dos tandas', '2 batches'), ('cuatro sacos', '4 bags'),
    ('four bags of seeds from Five Pound Supply', '4 bags of seeds from Five Pound Supply'),
    ('cuatro bolsas de semillas de Five Pound Supply', '4 bags de semillas de Five Pound Supply'),
])
def test_spoken_quantities_and_units(phrase, expected):
    assert normalize_voice_text(phrase) == expected


@pytest.mark.parametrize('phrase, expected', [
    ('lot one two three four', 'lot 1234'), ('lote uno dos tres cuatro', 'lote 1234'),
    ('lot zero zero one two', 'lot 0012'), ('lote cero cero uno dos', 'lote 0012'),
    ('lot oh one two three', 'lot 0123'), ('lot 0012', 'lot 0012'),
    ('lot number one two three four', 'lot number 1234'),
    ('lote número uno dos tres cuatro', 'lote número 1234'),
    ('P fifteen eighty-five', 'P1585'), ('P one five eight five', 'P1585'),
    ('P quince ochenta y cinco', 'P1585'), ('P uno cinco ocho cinco', 'P1585'),
    ('lot P fifteen eighty-five', 'lot P1585'),
    ('lote P quince ochenta y cinco', 'lote P1585'),
    ('lot P zero one five eight five', 'lot P01585'),
    ('lot one thousand two hundred thirty-four', 'lot 1234'),
    ('lote mil doscientos treinta y cuatro', 'lote 1234'),
    ('lot A one two three', 'lot A123'),
    ('lot one hundred and two', 'lot 102'),
    (' P one five eight five ', ' P1585 '),
    ('lot one two three four and show the entries', 'lot 1234 and show the entries'),
    ('lote uno dos tres cuatro y listo', 'lote 1234 y listo'),
    ('lot one two three four and five bags', 'lot 1234 and 5 bags'),
    ('lote uno dos tres cuatro y cinco bolsas', 'lote 1234 y 5 bags'), ('lote A uno dos tres', 'lote A123'),
])
def test_spoken_lot_codes_preserve_explicit_digits(phrase, expected):
    assert normalize_voice_text(phrase) == expected


@pytest.mark.parametrize('label', ['POL', 'B.O.L.', 'B. O. L.', 'bowl', 'B O L', 'bol'])
@pytest.mark.parametrize('number', ['5521', 'five five two one', 'cinco cinco dos uno'])
def test_bol_spellings_and_spoken_codes(label, number):
    assert normalize_voice_text(f'{label} {number}') == 'BOL 5521'


@pytest.mark.parametrize('text', [
    'Green Supply', 'Grain Supply', 'Pound Supply', 'Pol Supply', 'Bowl Supply', 'fake bakery', 'Classic four', 'SS number nine',
    'Sunshine Bakery', 'B One Bakery', 'New order from B One Bakery.',
    'Five Pound Supply', 'Five Pound Inc.', 'Five Pound LLC', 'Five Pound Company',
    'Five Pound Bakery', 'Five Pound Depot', 'Five Pound, Inc.', 'SS number one hundred', 'Classic ninety', 'Please confirm the lot tag', 'No lot was supplied',
    'There are one or two ways to do this', 'A five year plan',
    'P1585', 'UNVERIFIED-1234', 'x10lb', 'abc50lbs', '1,00,000 pounds', 'two batches from lot XYZ',
])
def test_unrelated_prose_names_and_existing_codes_do_not_gain_facts(text):
    expected = '2 batches from lot XYZ' if text == 'two batches from lot XYZ' else text
    assert normalize_voice_text(text) == expected


@pytest.mark.parametrize('text', [
    'Classic #9 and SS #1, 4 bags, 50 lb, lot 1234, BOL 5521.',
    'Hicimos dos tandas de SS número uno del lote P quince ochenta y cinco.',
    'Packed forty cases from lot one two three four.',
])
def test_normalization_is_idempotent(text):
    normalized = normalize_voice_text(text)
    assert normalize_voice_text(normalized) == normalized
