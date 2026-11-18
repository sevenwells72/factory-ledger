"""A tiny plant, a fake receiving Form and a fake Google for the 097 tests
(tests/test_receiving_form.py, tests/test_trial_recall.py). Not a test file."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import google_evidence
import receiving_form

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (ROOT / 'migrations/097_trial_recall.sql').read_text()
DOWN = (ROOT / 'migrations/down/097_trial_recall_down.sql').read_text()
NEW_TABLES = ('receiving_form_sync_runs', 'receiving_form_rows', 'receipt_bol_links', 'trial_recall_runs')
ET = receiving_form.PLANT_TIMEZONE
CUTOFF = date(2026, 11, 11)
PHOTO = '1a7X-ZpzzF4PPfR6qbwNaLYHOkMUNn_P6'
PHOTO_2 = '1O7QaIsBPsUyudfkIILnpRxfslrVor-JN'
# The real tab's heading row, in its real order.
HEADER = ['$0.00', 'Was lot# entered into Odoo?', 'Timestamp', 'Name:',
          'Did you place new lot# stickers on the pallets received?', 'What lot# did you use?',
          'Truck Condition: clean, there is no damage odor or pests/ Estado del ...',
          ' Approved Supplier?/Para recibir: ¿Proveedor aprobado?', ' BOL#', 'CARRIER/TRANSPORTADOR', 'PICTURE OF BOL',
          *['Name of Product: Case Size: Number of Cases (example:coconut-50lbs-32)'] * 6]


def day(offset=0):
    """A plant date: 0 = the cutoff (2026-11-11), 1 = the day after, -1 = the day before."""
    return CUTOFF + timedelta(days=offset)


def stamp(on, hour=11, minute=47):
    """A Form timestamp as the sheet shows it."""
    return f'{on.month}/{on.day}/{on.year} {hour}:{minute:02d}:46'


def form_row(on, lot, products, *, photo=PHOTO, bol='632054', carrier='Jack’s Eggs', name='Arturo', hour=11):
    link = ', '.join(f'https://drive.google.com/open?id={p}' for p in ([photo] if isinstance(photo, str) else photo or []))
    return ['', '', stamp(on, hour) if on else '', name, 'yes', lot, 'Yes', 'YES', bol, carrier, link, *products]


def sheet(*rows):
    return {'sheet_id': receiving_form.SHEET_ID, 'tab': receiving_form.TAB, 'gid': 1234,
            'time_zone': 'America/New_York', 'values': [HEADER, *rows]}


class FakeGoogle:
    """Stands in for google_evidence.Client: a sheet and some Drive files. Records every call."""

    def __init__(self, values=None, files=None):
        self.values, self.calls, self.refused_media = values or [HEADER], [], set()
        self.files = {PHOTO: {'mime_type': 'image/jpeg', 'size': 1000}} if files is None else files

    def sheet(self, sheet_id, tab):
        self.calls.append(('sheet', sheet_id, tab))
        return {'time_zone': 'America/New_York', 'gid': 1234, 'values': self.values}

    def file_metadata(self, file_id):
        self.calls.append(('metadata', file_id))
        if file_id not in self.files:
            raise google_evidence.GoogleError('NOT_FOUND', 'Google has no such file or sheet.')
        meta = self.files[file_id]
        return {'id': file_id, 'name': 'bol.jpg', 'mime_type': meta['mime_type'], 'size': meta.get('size'),
                'parents': meta.get('parents', []), 'trashed': meta.get('trashed', False), 'web_view_link': None}

    def open_media(self, file_id, max_bytes=None):
        self.calls.append(('media', file_id))
        if file_id in self.refused_media:       # Drive shows the file's details but refuses its contents
            raise google_evidence.GoogleError('FORBIDDEN', 'Google did not allow this account to read it.')
        return iter([b'photo-', b'bytes'])

    def download(self, file_id, max_bytes=None):
        return b''.join(self.open_media(file_id))


class Plant:
    """Products, lots and entries posted the way the app leaves them in the books."""

    def __init__(self, cur):
        self.cur, self.lots, self.txs, self.count = cur, {}, set(), 0
        self.tag = 'TR' + uuid4().hex[:6].upper()
        cur.execute(MIGRATION)
        cur.execute("INSERT INTO actors (name, role, key_hash) VALUES (%s, 'floor', %s) RETURNING id",
                    (f'{self.tag} Worker', uuid4().hex * 2))
        self.actor = cur.fetchone()['id']
        cur.execute('INSERT INTO suppliers (name) VALUES (%s) RETURNING id', (f'{self.tag} Sugar Supply',))
        self.supplier = cur.fetchone()['id']
        self.supplier_name = f'{self.tag} Sugar Supply'

    def name(self, words):
        return f'{self.tag} {words}'

    def product(self, words, type_='ingredient', uom='lb'):
        self.cur.execute('INSERT INTO products (name, type, uom, active) VALUES (%s, %s, %s, true) RETURNING id',
                         (self.name(words), type_, uom))
        return self.cur.fetchone()['id']

    def lot(self, product, code=None, supplier_lot='SUP-1', **columns):
        self.count += 1
        code = code or f'26-11-12-{self.tag}-{self.count:03d}'
        names = ''.join(f', {name}' for name in columns)
        self.cur.execute(f'INSERT INTO lots (product_id, lot_code, supplier_lot_code{names}) '
                         f'VALUES (%s, %s, %s{", %s" * len(columns)}) RETURNING id',
                         (product, code, supplier_lot, *columns.values()))
        lot = self.cur.fetchone()['id']
        self.lots[lot] = code
        return lot

    def post(self, type_, lines, on, *, named=True, hour=10, **columns):
        """One posted entry. `lines` = [(lot id, quantity), ...]; extra keywords are transaction columns."""
        when = datetime(on.year, on.month, on.day, hour, tzinfo=ET)
        self.count += 1
        columns.setdefault('receipt_number', f'{type_[:3].upper()}-{self.tag}-{self.count:03d}')
        names = ''.join(f', {name}' for name in columns)
        self.cur.execute(f"""INSERT INTO transactions (type, timestamp, status, occurred_at, business_date, operator_id,
                                                       entered_by_actor_id{names})
                             VALUES (%s, %s, 'posted', %s, %s, %s, %s{', %s' * len(columns)}) RETURNING id""",
                         (type_, when.astimezone(timezone.utc).replace(tzinfo=None),      # the ledger stores UTC
                          when, on, f'{self.tag} Worker' if named else 'legacy-shared-key',
                          self.actor if named else None, *columns.values()))
        tx = self.cur.fetchone()['id']
        self.txs.add(tx)
        for lot, quantity in lines:
            self.cur.execute('SELECT product_id FROM lots WHERE id = %s', (lot,))
            self.cur.execute('INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) '
                             'VALUES (%s, %s, %s, %s)', (tx, self.cur.fetchone()['product_id'], lot, quantity))
        return tx

    def receive(self, product, quantity, on, *, code=None, supplier_lot='SUP-1', bol='632054', **columns):
        lot = self.lot(product, code, supplier_lot)
        columns.setdefault('supplier_id', self.supplier)
        columns.setdefault('shipper_name', self.supplier_name)
        return lot, self.post('receive', [(lot, quantity)], on, bol_reference=bol, **columns)

    def ship(self, lot, quantity, on, customer, *, bol='OUT-1', **columns):
        tx = self.post('ship', [(lot, -quantity)], on, customer_name=customer, bol_reference=bol, **columns)
        self.cur.execute('INSERT INTO shipments (transaction_id, shipped_at, bol_reference) VALUES (%s, %s, %s) '
                         'RETURNING id', (tx, datetime(on.year, on.month, on.day, 10, tzinfo=ET), bol))
        return tx

    def void(self, tx):
        self.cur.execute("""INSERT INTO ledger_corrections (target_table, target_id, event_type, previous_values,
                                                            replacement_values, reason, operator_id)
                            VALUES ('transactions', %s, 'void', '{"status": "posted"}', '{"status": "voided"}',
                                    'Entered by mistake', 'legacy-shared-key')""", (tx,))

    def chain(self, on=None, *, customer=None, received=11250, shipped=300):
        """Sugar received -> a batch made -> cases packed -> shipped. Returns the ids by name."""
        on = on or day(1)
        made = on + timedelta(days=1)
        sugar = self.product('Sugar Brown')
        batch = self.product('Batch Granola', 'batch')
        cases = self.product('Granola 25 LB', 'finished')
        sugar_lot, receipt = self.receive(sugar, received, on)
        batch_lot = self.lot(batch, supplier_lot=None)
        make = self.post('make', [(batch_lot, 400), (sugar_lot, -400)], made)
        case_lot = self.lot(cases, supplier_lot=None)
        pack = self.post('pack', [(case_lot, 400), (batch_lot, -400)], made, hour=13)
        ship = self.ship(case_lot, shipped, made + timedelta(days=1), customer or self.name('Market'))
        return {'sugar': sugar, 'batch': batch, 'cases': cases, 'sugar_lot': sugar_lot, 'batch_lot': batch_lot,
                'case_lot': case_lot, 'receipt': receipt, 'make': make, 'pack': pack, 'ship': ship, 'on': on}

    def form_row(self, on, lot, products, **options):
        """A Form row for one of this plant's lots (products are words, e.g. 'Sugar brown-50lbs-225')."""
        return form_row(on, self.lots[lot] if isinstance(lot, int) else lot,
                        [self.name(p) for p in products], **options)


def fingerprint(cur, tables):
    """Row count and content hash of each table."""
    out = {}
    for table in tables:
        cur.execute(f'SELECT count(*) AS n, md5(COALESCE(string_agg(t::text, \'|\' ORDER BY t::text), \'\')) AS h '
                    f'FROM {table} t')
        out[table] = tuple(cur.fetchone().values())
    return out


def existing_tables(cur):
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename <> ALL(%s) ORDER BY 1",
                (list(NEW_TABLES),))
    return [row['tablename'] for row in cur.fetchall()]
