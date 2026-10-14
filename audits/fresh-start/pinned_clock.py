"""Pin the wall clock for the offline count-tool tests (FOLLOWUPS P1.25).

The tool compares the signed approval date with the New York plant date, so fixtures that use the
UTC date fail between 20:00 and 24:00 ET. This helper freezes datetime.now() in the count-tool and
test modules so the suite can be proven at any plant time:

    python3 pinned_clock.py 2026-10-13T23:59 test_apply_reset test_fresh_start_v3

A naive time is read as America/New_York. No live access; it only patches module attributes.
"""
import contextlib,sys,unittest
from datetime import datetime as _Real
from unittest.mock import patch
from zoneinfo import ZoneInfo

PLANT=ZoneInfo('America/New_York')
MODULES=('fresh_start_common','fresh_start_v3','apply_reset','test_apply_reset','test_fresh_start_v3')

def at(text):
 t=_Real.fromisoformat(text)
 return t if t.tzinfo else t.replace(tzinfo=PLANT)

class _Meta(type):
 # Fixture values built by arithmetic are plain datetimes; keep isinstance(x, datetime) true for them.
 def __instancecheck__(cls,obj):return isinstance(obj,_Real)

def frozen_datetime(moment):
 class FrozenDatetime(_Real,metaclass=_Meta):
  _pinned_clock=True
  @classmethod
  def now(cls,tz=None):
   return _Real.fromtimestamp(moment.timestamp(),tz)
  @classmethod
  def utcnow(cls):
   return _Real.fromtimestamp(moment.timestamp(),ZoneInfo('UTC')).replace(tzinfo=None)
 return FrozenDatetime

@contextlib.contextmanager
def pinned(moment):
 """Freeze datetime.now() in every loaded count-tool module at `moment` (aware or New York naive)."""
 if isinstance(moment,str):moment=at(moment)
 fake=frozen_datetime(moment)
 with contextlib.ExitStack() as stack:
  for name in MODULES:
   mod=sys.modules.get(name)
   now=getattr(mod,'datetime',None) if mod is not None else None
   if now is _Real or getattr(now,'_pinned_clock',False):
    stack.enter_context(patch.object(mod,'datetime',fake))
  yield moment

def main(argv):
 if len(argv)<2:
  print(__doc__);return 2
 moment=at(argv[0])
 suite=unittest.defaultTestLoader.loadTestsFromNames(argv[1:])
 with pinned(moment):
  print(f'Clock pinned at {moment.astimezone(PLANT).isoformat()} (New York)',flush=True)
  result=unittest.TextTestRunner(verbosity=1).run(suite)
 return 0 if result.wasSuccessful() else 1

if __name__=='__main__':
 sys.exit(main(sys.argv[1:]))
