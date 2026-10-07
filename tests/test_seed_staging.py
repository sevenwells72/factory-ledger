"""Seed integrity and the production copy's SQL allowlist, using a local test DB."""
import psycopg2
from unittest.mock import patch

import pytest
from scripts import seed_staging as seed


@pytest.fixture
def tuple_cursor(db_cursor):
    with db_cursor.connection.cursor(cursor_factory=psycopg2.extensions.cursor) as cur:
        yield cur


def test_exact_copy_allowlist():
    assert seed.MASTER_TABLES == (
        "customers", "suppliers", "products",
        "customer_product_aliases", "supplier_product_aliases",
    )


def test_secret_permissions(tmp_path):
    path = tmp_path/"url"
    path.write_text("synthetic")
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="600"):
        seed.secret_file(path)


def test_production_destination_rejected_before_connection(tmp_path):
    path = tmp_path/"url"
    path.write_text("postgresql://postgres:synthetic@production.example/postgres")
    path.chmod(0o600)
    args = type("Args", (), {"db_url_file":path, "production_host":"production.example",
                            "copy_master_data":False})()
    with patch.object(seed.psycopg2,"connect") as connect:
        with pytest.raises(RuntimeError, match="matches production"):
            seed.run(args)
        connect.assert_not_called()


@pytest.mark.db
def test_seed_is_idempotent_and_uses_no_known_actor_key(tuple_cursor):
    assert seed.seed_fixtures(tuple_cursor) is True
    assert seed.seed_fixtures(tuple_cursor) is False
    seed.sync_sequences(tuple_cursor)
    tuple_cursor.execute("SELECT count(*) FROM products WHERE id >= %s", (seed.FIXTURE_BASE,))
    assert tuple_cursor.fetchone()[0] == 3
    tuple_cursor.execute("SELECT active FROM actors WHERE id=%s", (seed.FIXTURE_BASE+1,))
    assert tuple_cursor.fetchone()[0] is False
    tuple_cursor.execute("""SELECT sum(quantity_lb) FROM transaction_lines
                         WHERE transaction_id=%s""", (seed.FIXTURE_BASE+1,))
    assert tuple_cursor.fetchone()[0] == 500
    tuple_cursor.execute("SELECT count(*) FROM sales_order_lines WHERE sales_order_id=%s",
                      (seed.FIXTURE_BASE+1,))
    assert tuple_cursor.fetchone()[0] == 3


class ReadOnlySource:
    """An instrumented source; any SQL beyond the allowlist fails the test."""
    def __init__(self, columns):
        self.columns = columns
        self.statements = []
        self.closed = False
        self.table = None
    def cursor(self):
        return self
    def __enter__(self):
        return self
    def __exit__(self,*args):
        pass
    def close(self):
        self.closed = True
    def execute(self, statement):
        if isinstance(statement,str):
            assert statement in (
                "BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY", "ROLLBACK")
            self.statements.append(statement)
            return
        # Composed SELECT contains the literal SELECT, column identifiers and table.
        table = [part for part in statement._wrapped
                 if isinstance(part, seed.sql.Identifier)][-1].string
        assert table in seed.MASTER_TABLES
        assert statement._wrapped[0].string == "SELECT "
        self.table = table
        self.statements.append("SELECT "+table)
    def fetchall(self):
        data = {
            "customers": {"id":720001,"name":"Copy Test Customer","active":True},
            "suppliers": {"id":720001,"name":"Copy Test Supplier","active":True},
            "products": {"id":720001,"name":"Copy Test Product","type":"ingredient",
                         "active":True,"uom":"lb","is_service":False,"is_copack":False,
                         "no_production":False},
            "customer_product_aliases": {
                "id":720001,"customer_id":720001,"product_id":720001,
                "customer_item_code":"COPY-TEST","case_size_lb":25},
            "supplier_product_aliases": {
                "id":720001,"supplier_id":720001,"product_id":720001,
                "vendor_description":"COPY-TEST","lb_per_unit":25,"unit":"case"},
        }
        # Use explicit defaults for nullable fixture fields; timestamps NOT NULL.
        from datetime import datetime,timezone
        row = data[self.table]
        for col in ("created_at","updated_at"):
            if col in self.columns[self.table]:
                row[col]=datetime.now(timezone.utc)
        return [tuple(row.get(col) for col in self.columns[self.table])]


@pytest.mark.db
def test_copy_reads_only_master_tables_and_excludes_generated_alias(tuple_cursor):
    columns={}
    for table in seed.MASTER_TABLES:
        tuple_cursor.execute("""SELECT column_name FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s AND is_generated='NEVER'
            ORDER BY ordinal_position""", (table,))
        columns[table]=[r[0] for r in tuple_cursor.fetchall()]
    assert "alias_key" not in columns["customer_product_aliases"]
    source=ReadOnlySource(columns)
    with patch.object(seed.psycopg2,"connect",return_value=source) as connect:
        counts=seed.copy_master_data(tuple_cursor,
            "postgresql://test:synthetic@source.example/postgres",
            "postgresql://test:synthetic@staging.example/postgres")
        assert counts == {table:1 for table in seed.MASTER_TABLES}
        assert seed.copy_master_data(tuple_cursor,
            "postgresql://test:synthetic@source.example/postgres",
            "postgresql://test:synthetic@staging.example/postgres") == {"already_copied":True}
        assert connect.call_count == 1
    assert source.closed
    assert source.statements == [
        "BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY",
        *("SELECT "+t for t in seed.MASTER_TABLES), "ROLLBACK"]
    tuple_cursor.execute("SELECT uom FROM products WHERE id=720001")
    assert tuple_cursor.fetchone()[0] == "lb"
    tuple_cursor.execute("SELECT alias_key FROM customer_product_aliases WHERE id=720001")
    assert tuple_cursor.fetchone()[0] == "copy-test"
