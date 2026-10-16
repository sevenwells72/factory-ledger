"""Deterministic navigation for supported Office work outside chat recording."""
import re
from urllib.parse import urlencode

from assistant_voice import normalize_order_code


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
    expected = re.search(r'\b(?:expected deliver(?:y|ies)|expected receipt|entrega esperada|entregas esperadas)\b', text, re.I)
    future = re.search(r'\b(?:is sending|will send|will deliver|is delivering|est[aá] enviando|va a enviar|enviar[aá]|va a entregar|entregar[aá])\b', text, re.I)
    goods = re.search(r'\b(?:cases?|cajas?|bags?|bolsas?|lb|lbs|pounds?|libras?|pallets?|tarimas?)\b', text, re.I)
    if (expected and re.search(r'\b(?:create|add|new|crear|crea|agrega|nueva|nuevo|registra)\b', text, re.I)) or (future and goods):
        return {'kind': 'handoff', 'message': ('Cree la entrega esperada en la pantalla Oficina.' if es else
            'Create the expected delivery on the Office screen.'), 'actions': [{
            'href': '/dashboard/office.html?action=create_expected_receipt',
            'label': 'Crear entrega esperada' if es else 'Create expected delivery'}]}
    return None


def is_receive_continuation(text):
    return text.strip().lower().rstrip('.').replace('ú', 'u') in {
        'continue the next queued receive', 'continua la siguiente recepción pendiente',
        'continua la siguiente recepcion pendiente'}
