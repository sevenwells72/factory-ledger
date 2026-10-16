"""Deterministic navigation for supported Office work outside chat recording."""
import re
import unicodedata
from urllib.parse import urlencode

from assistant_voice import normalize_order_code


def plain_words(text):
    return ''.join(char for char in unicodedata.normalize('NFKD', text.lower())
                   if not unicodedata.combining(char))


def customer_creation(text):
    """Recognize creation intent after the model has identified a customer name."""
    text = plain_words(text)
    creation = r'(?:add|create|register|agrega(?:r|lo|la)?|agregue|anad[ea](?:lo|la)?|anadir|crea(?:r)?|cree|registra(?:r)?)'
    explicit = re.search(r'\b(?:new customer|nuevo cliente|cliente nuevo)\b', text)
    explicit = explicit or re.search(r'\b' + creation + r'\s+(?:(?:a|an|un|el)\s+)?(?:customer|cliente)\b', text)
    return bool(explicit or customer_only(text) and re.search(r'\b' + creation + r'\b', text))


def customer_only(text):
    """A customer-only choice can finish locally; requested orders must continue."""
    return not re.search(r'\b(?:orders?|pedidos?|orden(?:es)?|cases?|cajas?|bags?|bolsas?|lb|lbs|pounds?|libras?)\b', plain_words(text))


def office_handoff(text, language='en'):
    es = language == 'es'
    text = normalize_order_code(text)
    change = re.search(r'\b(change|update|modify|edit|cancel|cancela|cancelar|cancele|cambia|cambiar|cambie|modifica|modificar)\b', text, re.I)
    order = re.search(r'\bSO-\d{6}-\d{3,}\b', text, re.I)
    if change and (order or re.search(r'\b(?:order|pedido|orden)\b', text, re.I)):
        code = order[0].upper() if order else None
        action = 'cancel_order' if re.match(r'cancel', change[0], re.I) else 'update_order_line'
        params = {'action': action}
        if code:
            params['order'] = code
        message = ('Los pedidos se cambian o cancelan en la pantalla Oficina' if es else
                   'Changing or cancelling an order is done on the Office screen')
        message += (' — abra ' if es else ' — open ') + code + '.' if code else '.'
        return {'kind': 'handoff', 'message': message, 'actions': [{
            'href': '/dashboard/office.html?' + urlencode(params),
            'label': ('Abrir ' if es else 'Open ') + (code or ('Oficina' if es else 'Office'))}]}
    words = plain_words(text)
    expected = re.search(r'\b(?:expected deliver(?:y|ies)|expected receipt|entrega esperada|entregas esperadas)\b', words)
    future = re.search(r'\b(?:is sending|will send|will deliver|is delivering|esta enviando|va a enviar|enviara|va a entregar|entregara|va a mandar|mandara)\b', words)
    goods = re.search(r'\b(?:cases?|cajas?|bags?|bolsas?|lb|lbs|pounds?|libras?|pallets?|tarimas?)\b', words)
    create = re.search(r'\b(?:create|add|new|crea(?:r|lo|la)?|agrega(?:r|lo|la)?|anad[ea](?:lo|la)?|anadir|nueva|nuevo|registra(?:r|lo|la)?)\b', words)
    if (expected and create) or (future and goods):
        return {'kind': 'handoff', 'message': ('Cree la entrega esperada en la pantalla Oficina.' if es else
            'Create the expected delivery on the Office screen.'), 'actions': [{
            'href': '/dashboard/office.html?action=create_expected_receipt',
            'label': 'Crear entrega esperada' if es else 'Create expected delivery'}]}
    return None


def is_receive_continuation(text):
    return text.strip().lower().rstrip('.').replace('ú', 'u') in {
        'continue the next queued receive', 'continua la siguiente recepción pendiente',
        'continua la siguiente recepcion pendiente'}
