-- FR-15: named-actor attribution for writes with no inventory transaction.
-- Apply after 052, before deploying named-actor metadata writes. No backfill.
-- Legacy requests never insert here. Each insert shares the business write's
-- transaction; an audit failure must roll back that write. No key material.
-- No BEGIN/COMMIT: compatible with the SQL editor and test savepoints.
CREATE TABLE IF NOT EXISTS public.actor_write_audit (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    actor_id integer NOT NULL REFERENCES public.actors(id),
    operator_id text NOT NULL CHECK (btrim(operator_id) <> ''),
    method text NOT NULL CHECK (method IN ('POST', 'PATCH')),
    route text NOT NULL,
    target_table text NOT NULL CHECK (
        target_table IN ('customers', 'lots', 'sales_orders', 'sales_order_lines')
    ),
    target_id bigint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX IF NOT EXISTS actor_write_audit_target_idx
    ON public.actor_write_audit (target_table, target_id, id);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger
                   WHERE tgrelid = 'public.actor_write_audit'::regclass
                     AND tgname = 'actor_write_audit_append_only') THEN
        CREATE TRIGGER actor_write_audit_append_only
            BEFORE UPDATE OR DELETE ON public.actor_write_audit
            FOR EACH ROW EXECUTE FUNCTION public.ledger_block_append_only_change();
    END IF;
END $$;

-- Supabase Data API hardening (owner-approved, PR #66 review F7). No policies:
-- anon/authenticated get nothing; the backend connects as the table owner,
-- which bypasses RLS (not FORCEd). Idempotent on rerun.
ALTER TABLE public.actor_write_audit ENABLE ROW LEVEL SECURITY;

INSERT INTO public.migration_markers (name) VALUES ('056_actor_write_audit')
ON CONFLICT (name) DO NOTHING;
