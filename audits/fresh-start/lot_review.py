"""Pure local lot-date and similarity helpers. Suggestions never change identity."""
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from difflib import SequenceMatcher
from itertools import combinations
import re


def dt(value):
    return datetime.fromisoformat(str(value).replace('Z', '+00:00')) if value else None


def code_date(code):
    """Only obvious calendar dates; never guess a encoded supplier/batch date."""
    text = code.upper().strip()
    patterns = [(r'(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)', '%Y-%m-%d'),
                (r'(?<!\d)(\d{1,2}/\d{1,2}/\d{4})(?!\d)', '%m/%d/%Y'),
                (r'\b([A-Z]{3,9}\s+\d{1,2}[, ]+\d{4})(?!\d)', '%b %d %Y')]
    for pattern, fmt in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        token = re.sub(r'[, ]+', ' ', match.group(1))
        formats = (fmt, '%B %d %Y') if fmt.startswith('%b') else (fmt,)
        for f in formats:
            try:
                return datetime.strptime(token, f).date()
            except ValueError:
                continue
    return None


def stripped_code(code):
    return re.sub(r'(?i)[\s_\-/]*(?:lote|lot)\s*$', '', code).strip()


def normalized(code):
    return re.sub(r'[^a-z0-9]', '', stripped_code(code).casefold())


def ends_lot(code):
    return bool(re.search(r'(?i)(?:lote|lot)\s*$', code))


def near_reason(a, b):
    if a.casefold() == b.casefold():
        return 'Same code except letter case, or duplicate code.'
    if normalized(a) == normalized(b):
        return 'Same code after spacing/punctuation or a trailing Lot/Lote word is removed.'
    da, db = code_date(a), code_date(b)
    if da and db:
        return 'Same calendar date; formatting or a run suffix differs.' if da == db else None
    na, nb = normalized(a), normalized(b)
    base_a = re.sub(r'[-_ ](?:[A-Za-z]|\d{1,2})$', '', stripped_code(a)).casefold()
    base_b = re.sub(r'[-_ ](?:[A-Za-z]|\d{1,2})$', '', stripped_code(b)).casefold()
    if base_a == base_b and len(base_a) >= 5:
        return 'Same base code with a possible run suffix.'
    if min(len(na), len(nb)) >= 5 and SequenceMatcher(None, na, nb).ratio() >= 0.92:
        return 'Very similar text (at least 92% similarity); check for a transcription error.'
    return None


def lot_suggestions(code, lots, limit=3):
    scored = []
    for lot in lots:
        other = lot.get('lot_code') or ''
        if not other:
            continue
        score = SequenceMatcher(None, code.casefold(), other.casefold()).ratio()
        if code_date(code) and code_date(code) == code_date(other):
            score = max(score, 0.96)
        if normalized(code) == normalized(other):
            score = 1.0
        scored.append((score, other, lot['id']))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [dict(lot_code=c, lot_id=i, similarity=round(score, 3)) for score,c,i in scored[:limit]]


def facts(snapshot):
    by_lot = defaultdict(list)
    for line in snapshot['lines']:
        by_lot[(line['product_id'], line['lot_id'])].append(line)
    now = dt(snapshot['snapshot_at']); since = now - timedelta(days=60)
    result = {}
    for lot in snapshot['lots']:
        lines = by_lot[(lot['product_id'], lot['id'])]
        posted = [x for x in lines if x['effective_status'] == 'posted']
        balance = sum((Decimal(str(x['quantity_lb'])) for x in posted), Decimal(0))
        physical = [dt(x['occurred_at']) for x in lines if x.get('occurred_at')]
        credits = [dt(x['occurred_at']) for x in posted if x.get('occurred_at') and Decimal(str(x['quantity_lb'])) > 0]
        made = [dt(x['occurred_at']) for x in posted if x.get('occurred_at') and Decimal(str(x['quantity_lb'])) > 0 and x['type'] in ('make','pack','receive')]
        real_activity = list(physical)
        for x in lines:
            for field,source in [('line_created_at','line_created_at_source'),('transaction_created_at','transaction_created_at_source')]:
                if x.get(field) and x.get(source) == 'database':
                    real_activity.append(dt(x[field]))
            for field in ('line_correction_at','transaction_correction_at'):
                if x.get(field):
                    real_activity.append(dt(x[field]))
        if made:
            date,source = min(made), 'first posted make/pack/receive physical time'
        elif lot.get('received_at'):
            date,source = dt(lot['received_at']), 'lot received_at'
        elif credits:
            date,source = min(credits), 'first posted credit physical time'
        elif physical:
            date,source = min(physical), 'first recorded physical activity'
        elif lot.get('lot_created_at') and lot.get('lot_created_at_source') == 'database':
            date,source = dt(lot['lot_created_at']), 'lot creation time (not production date)'
        else:
            date,source = None, 'date unavailable; no date inferred from code'
        recent = any(since <= x <= now for x in real_activity)
        result[lot['id']] = dict(lot, balance_lb=balance, lot_date=date.isoformat() if date else '',
                                date_source=source, first_physical=min(physical).isoformat() if physical else '',
                                last_physical=max(physical).isoformat() if physical else '',
                                last_activity=max(real_activity).isoformat() if real_activity else '',
                                recent=recent, listed=bool(balance != 0 or recent))
    return result


def lookalike_pairs(snapshot):
    product_lots = defaultdict(list)
    for lot in snapshot['lots']:
        product_lots[lot['product_id']].append(lot)
    pairs, suffix_only = [], []
    for pid,lots in sorted(product_lots.items()):
        paired = set()
        for a,b in combinations(lots, 2):
            reason = near_reason(a.get('lot_code') or '', b.get('lot_code') or '')
            if reason:
                pairs.append((pid,a,b,reason));paired.update((a['id'],b['id']))
        for a in lots:
            if ends_lot(a.get('lot_code') or '') and a['id'] not in paired:
                suffix_only.append((pid,a))
    return pairs,suffix_only
