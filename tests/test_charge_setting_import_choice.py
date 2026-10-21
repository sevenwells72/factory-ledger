"""Direct catalog imports must choose Product or Charge for each new item."""
from pathlib import Path
from uuid import uuid4

import psycopg2
import pytest


ROOT = Path(__file__).resolve().parents[1]
UP = (ROOT / 'migrations/086_charge_setting_check.sql').read_text()
DOWN = (ROOT / 'migrations/down/086_charge_setting_check_down.sql').read_text()


def _name():
    return 'Imported catalog item ' + uuid4().hex


@pytest.mark.db
@pytest.mark.parametrize('choice_sql', ['', ',is_service'])
def test_new_import_requires_an_explicit_nonnull_choice(db_cursor, choice_sql):
    values = '%s' if not choice_sql else '%s,NULL'
    name = _name()
    db_cursor.execute('SAVEPOINT import_without_choice')
    with pytest.raises(psycopg2.errors.NotNullViolation) as failure:
        db_cursor.execute(f'INSERT INTO products(name{choice_sql}) VALUES ({values})', (name,))
    assert failure.value.diag.column_name == 'is_service'
    db_cursor.execute('ROLLBACK TO SAVEPOINT import_without_choice')
    db_cursor.execute('SELECT count(*) AS n FROM products WHERE name=%s', (name,))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.db
@pytest.mark.parametrize('kind,choice', [('batch', False), ('finished', False), ('finished', True)])
def test_new_import_saves_the_explicit_choice_and_existing_import_edits_keep_it(db_cursor, kind, choice):
    name = _name()
    db_cursor.execute('''INSERT INTO products(name,type,is_service) VALUES (%s,%s,%s)
                        RETURNING id,is_service''', (name, kind, choice))
    imported = db_cursor.fetchone()
    assert imported['is_service'] is choice
    # Updating an existing item does not require choosing again or reset it.
    db_cursor.execute('''UPDATE products SET name=%s,odoo_code=%s WHERE id=%s
                        RETURNING is_service''', (name + ' renamed', uuid4().hex, imported['id']))
    assert db_cursor.fetchone()['is_service'] is choice


@pytest.mark.db
@pytest.mark.parametrize('operation', ['insert_batch_charge', 'mark_batch_charge', 'change_charge_to_batch'])
def test_sql_imports_cannot_create_or_change_a_batch_to_charge(db_cursor, operation):
    if operation == 'insert_batch_charge':
        statement = "INSERT INTO products(name,type,is_service) VALUES (%s,'batch',true)"
        params = (_name(),)
    else:
        kind, choice = ('batch', False) if operation == 'mark_batch_charge' else ('finished', True)
        db_cursor.execute('INSERT INTO products(name,type,is_service) VALUES (%s,%s,%s) RETURNING id',
                          (_name(), kind, choice))
        product_id = db_cursor.fetchone()['id']
        statement = 'UPDATE products SET is_service=true WHERE id=%s' if operation == 'mark_batch_charge' else \
            "UPDATE products SET type='batch' WHERE id=%s"
        params = (product_id,)
    db_cursor.execute('SAVEPOINT batch_must_be_product')
    with pytest.raises(psycopg2.errors.CheckViolation) as failure:
        db_cursor.execute(statement, params)
    assert failure.value.diag.constraint_name == 'products_batch_is_product_check'
    db_cursor.execute('ROLLBACK TO SAVEPOINT batch_must_be_product')
    if operation != 'insert_batch_charge':
        db_cursor.execute('SELECT type,is_service FROM products WHERE id=%s', (product_id,))
        assert db_cursor.fetchone() == {'type': kind, 'is_service': choice}


@pytest.mark.db
def test_batch_guard_preserves_existing_values_on_apply_and_disappears_on_rollback(db_cursor):
    db_cursor.execute('SET LOCAL search_path TO public')
    db_cursor.execute(DOWN)
    # An invalid legacy item is preserved as evidence, never silently corrected.
    db_cursor.execute("INSERT INTO products(name,type,is_service) VALUES (%s,'batch',true) RETURNING id", (_name(),))
    legacy_id = db_cursor.fetchone()['id']
    db_cursor.execute(UP)
    db_cursor.execute(UP)
    db_cursor.execute('SELECT type,is_service FROM products WHERE id=%s', (legacy_id,))
    assert db_cursor.fetchone() == {'type': 'batch', 'is_service': True}
    db_cursor.execute("SELECT convalidated FROM pg_constraint WHERE conrelid='public.products'::regclass "
                      "AND conname='products_batch_is_product_check'")
    assert db_cursor.fetchone()['convalidated'] is False
    # An explicit correction can restore the invariant; future edits are safe.
    db_cursor.execute('UPDATE products SET is_service=false WHERE id=%s', (legacy_id,))
    db_cursor.execute('UPDATE products SET name=%s WHERE id=%s RETURNING is_service', (_name(), legacy_id))
    assert db_cursor.fetchone()['is_service'] is False
    db_cursor.execute(DOWN)
    db_cursor.execute("SELECT count(*) AS n FROM pg_constraint WHERE conrelid='public.products'::regclass "
                      "AND conname='products_batch_is_product_check'")
    assert db_cursor.fetchone()['n'] == 0
    db_cursor.execute(UP)


@pytest.mark.db
def test_migration_removes_only_the_import_default_and_rollback_restores_it(db_cursor):
    db_cursor.execute('SET LOCAL search_path TO public')
    db_cursor.execute(DOWN)
    db_cursor.execute('''INSERT INTO products(name,type,is_service) VALUES
                        (%s,'finished',false),(%s,'finished',true) RETURNING id,is_service''',
                      (_name(), _name()))
    before = db_cursor.fetchall()
    # Model a pre-migration import that relied on the old Product default.
    db_cursor.execute("INSERT INTO products(name,type) VALUES (%s,'finished') RETURNING id,is_service", (_name(),))
    legacy = db_cursor.fetchone()
    assert legacy['is_service'] is False
    db_cursor.execute(UP)
    db_cursor.execute(UP)
    db_cursor.execute('''SELECT column_default,is_nullable FROM information_schema.columns
                        WHERE table_schema='public' AND table_name='products' AND column_name='is_service' ''')
    column = db_cursor.fetchone()
    assert column['column_default'] is None and column['is_nullable'] == 'NO'
    db_cursor.execute('SELECT id,is_service FROM products WHERE id = ANY(%s) ORDER BY id',
                      ([row['id'] for row in before] + [legacy['id']],))
    assert db_cursor.fetchall() == sorted(before + [legacy], key=lambda row: row['id'])
    db_cursor.execute(DOWN)
    db_cursor.execute("INSERT INTO products(name,type) VALUES (%s,'finished') RETURNING is_service", (_name(),))
    assert db_cursor.fetchone()['is_service'] is False
    db_cursor.execute(UP)
