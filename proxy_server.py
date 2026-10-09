"""Explicit uvicorn proxy configuration; opt in per service, never in railway.json."""
import ipaddress
import os
import uvicorn


class RailwayEdgeHeaders:
    """Normalize Railway's overwritten X-Real-IP before uvicorn handles XFF.

    The public HTTP edge owns X-Real-IP; client-supplied XFF is not authoritative.
    This outer adapter runs before uvicorn rewrites scope.client, so the trust
    check uses the actual socket peer. Only enable behind Railway's HTTP edge.
    """
    def __init__(self, app, trusted_hosts):
        self.app = app
        self.trusted = {host.strip() for host in trusted_hosts.split(',')}

    async def __call__(self, scope, receive, send):
        peer = (scope.get('client') or ('', 0))[0]
        if scope['type'] in ('http', 'websocket') and ('*' in self.trusted or peer in self.trusted):
            headers = scope.get('headers', [])
            real = [value for key, value in headers if key.lower() == b'x-real-ip']
            forwarded = None
            if len(real) == 1:
                try:
                    forwarded = str(ipaddress.ip_address(real[0].decode('ascii'))).encode('ascii')
                except (ValueError, UnicodeError):
                    pass
            # Missing/malformed edge identity falls back to the peer, never to
            # a client-controlled XFF that could manufacture new PIN buckets.
            headers = [(key, value) for key, value in headers if key.lower() != b'x-forwarded-for']
            if forwarded is not None:
                headers.append((b'x-forwarded-for', forwarded))
            scope = dict(scope, headers=headers)
        await self.app(scope, receive, send)


def config():
    trusted = os.environ.get('FORWARDED_ALLOW_IPS', '127.0.0.1')
    result = uvicorn.Config('main:app', host='0.0.0.0', port=int(os.environ.get('PORT', '8000')),
                            proxy_headers=True, forwarded_allow_ips=trusted)
    result.load()
    if os.environ.get('RAILWAY_ENVIRONMENT_ID'):
        result.loaded_app = RailwayEdgeHeaders(result.loaded_app, trusted)
    return result


def run():
    uvicorn.Server(config()).run()


if __name__ == '__main__':
    run()
