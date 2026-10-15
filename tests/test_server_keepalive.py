"""Exercise Railway's keep-alive settings against the pinned Uvicorn server."""

import ast
import asyncio
from pathlib import Path
import shlex
import socket
import time

import pytest
import uvicorn


ROOT = Path(__file__).resolve().parents[1]
SLOW_SECONDS = 6


def _configured_timeout(service):
    """Use each Railway launch path, including Uvicorn's old default on main."""
    default = uvicorn.Config(lambda: None).timeout_keep_alive
    if service == "production":
        procfile = ROOT / "Procfile"
        web = next(
            (line.partition(":")[2] for line in procfile.read_text().splitlines()
             if line.startswith("web:")),
            "",
        ) if procfile.exists() else ""
        command = shlex.split(web)
        for index, argument in enumerate(command):
            if argument == "--timeout-keep-alive":
                return int(command[index + 1])
            if argument.startswith("--timeout-keep-alive="):
                return int(argument.split("=", 1)[1])
        return default

    # The guarded staging launcher embeds this Python source in its command.
    tree = ast.parse((ROOT / "scripts/staging_start_command.py").read_text())
    launch = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "LAUNCH" for target in node.targets)
    )
    for node in ast.walk(ast.parse(launch)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "uvicorn"
            and node.func.attr == "run"
        ):
            return next(
                (ast.literal_eval(argument.value) for argument in node.keywords
                 if argument.arg == "timeout_keep_alive"),
                default,
            )
    raise AssertionError("Staging launch command must run Uvicorn")


async def _read_response(reader):
    try:
        headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
    except asyncio.IncompleteReadError:
        pytest.fail("Uvicorn cut off the second request on the reused keep-alive connection")
    lines = headers.split(b"\r\n")
    assert lines[0] == b"HTTP/1.1 200 OK"
    content_length = next(
        int(line.split(b":", 1)[1])
        for line in lines[1:]
        if line.lower().startswith(b"content-length:")
    )
    return await asyncio.wait_for(reader.readexactly(content_length), timeout=10)


async def _back_to_back_requests(timeout):
    requests = []
    slow_started = asyncio.Event()

    async def app(scope, receive, send):
        requests.append((scope["path"], scope["client"]))
        if scope["path"] == "/slow":
            slow_started.set()
            await asyncio.sleep(SLOW_SECONDS)
        else:
            # Let both back-to-back requests arrive before completing the first.
            await asyncio.sleep(0.05)
        body = scope["path"].encode()
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})

    server = uvicorn.Server(uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        http="h11",
        loop="asyncio",
        lifespan="off",
        timeout_keep_alive=timeout,
        access_log=False,
        log_level="error",
    ))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        server_task = asyncio.create_task(server.serve(sockets=[listener]))
        writer = None
        try:
            startup_deadline = time.monotonic() + 5
            while not server.started:
                if server_task.done():
                    await server_task
                assert time.monotonic() < startup_deadline, "Uvicorn did not start"
                await asyncio.sleep(0.01)
            reader, writer = await asyncio.open_connection(*listener.getsockname())
            started = time.monotonic()
            # One TCP connection and one write deterministically exercise the race:
            # 0.24 schedules an idle timer after /fast, then starts the buffered
            # /slow without the data_received call that would cancel that timer.
            writer.write(
                b"GET /fast HTTP/1.1\r\nHost: localhost\r\n\r\n"
                b"GET /slow HTTP/1.1\r\nHost: localhost\r\n\r\n"
            )
            await writer.drain()
            assert await _read_response(reader) == b"/fast"
            await asyncio.wait_for(slow_started.wait(), timeout=1)
            assert [path for path, _ in requests] == ["/fast", "/slow"]
            assert requests[0][1] == requests[1][1]
            assert await _read_response(reader) == b"/slow"
            assert time.monotonic() - started >= SLOW_SECONDS
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            server.should_exit = True
            await asyncio.wait_for(server_task, timeout=10)


@pytest.mark.parametrize("service", ["production", "staging"])
def test_slow_second_request_survives_railway_keepalive(service):
    asyncio.run(_back_to_back_requests(_configured_timeout(service)))
