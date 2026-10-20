"""FL must run as exactly one server process per Railway replica.

The FL Assistant frees a failed turn's chat lock in this process's memory
(fl_assistant: released_leases) when the database is too busy to clear the row
at once. A second worker process would not see that and would answer
"chat busy" until the two-minute lock runs out (PR #141 review). Procfile and
the staging launcher pin one worker; this check makes any launch that
bypasses them loud.

It only logs: refusing to start would take FL down (a uvicorn 0.24 supervisor
keeps the port open with no live workers and does not restart them).
"""
import multiprocessing
import os
import sys


def configured_workers(argv=None, environ=None):
    """Worker count the uvicorn CLI / uvicorn.run would use; None if unreadable.

    An explicit --workers wins; without it both honour WEB_CONCURRENCY."""
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    value = environ.get('WEB_CONCURRENCY') or '1'
    for index, argument in enumerate(argv):
        if argument == '--workers' and index + 1 < len(argv):
            value = argv[index + 1]
        elif argument.startswith('--workers='):
            value = argument.split('=', 1)[1]
    try:
        return int(value)
    except ValueError:
        return None


def multiple_workers_problem(argv=None, environ=None, parent=None):
    """Describe why this may not be the only server process, or return None."""
    argv = sys.argv if argv is None else argv
    workers = configured_workers(argv, environ)
    if workers is None or workers > 1:
        return f'{workers or "an unreadable number of"} server workers configured (--workers / WEB_CONCURRENCY)'
    parent = multiprocessing.parent_process() if parent is None else parent
    # uvicorn runs one worker in-process; workers > 1 (or --reload, local
    # development only) run the app in spawned child processes.
    if parent and '--reload' not in argv:
        return 'the app runs in a child process of a multi-worker supervisor'
    return None
