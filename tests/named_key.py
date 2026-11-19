"""Named owner credential for local HTTP fixtures; never the shared API_KEY."""
from hashlib import sha256
KEY = 'test-named-owner-key'
NAME = 'Test named owner'
SEED_SQL = "INSERT INTO actors(name,role,key_hash,active) SELECT 'Test named owner','owner','" + sha256(KEY.encode()).hexdigest() + "',true WHERE NOT EXISTS (SELECT 1 FROM actors WHERE key_hash='" + sha256(KEY.encode()).hexdigest() + "');"
