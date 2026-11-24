"""Explicit test-fixture choices for legacy contracts versus current stock rules."""
def enabled(cur):
    cur.execute("SELECT 1 FROM migration_markers WHERE name='098_no_stock_by_hand'")
    return cur.fetchone() is not None


def manufactured_count(cur, items):
    """Generic count/approval tests use a batch; purchased-ingredient limits have their own proofs."""
    if enabled(cur):
        cur.execute("UPDATE products SET type='batch' WHERE id=%s",(items['ingredient']['id'],))
