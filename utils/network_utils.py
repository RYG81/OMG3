"""Network safety helpers for user-influenced public web downloads."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit


def validate_public_http_url(url: str) -> str:
    """Require an HTTP(S) URL whose resolved addresses are publicly routable."""

    raw = str(url or "").strip()
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public http:// and https:// URLs are allowed")
    if parsed.username or parsed.password:
        raise ValueError("Credentials are not allowed in web download URLs")

    try:
        default_port = 443 if parsed.scheme == "https" else 80
        addresses = socket.getaddrinfo(
            parsed.hostname, parsed.port or default_port, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve web host {parsed.hostname!r}") from exc
    if not addresses:
        raise ValueError(f"Web host {parsed.hostname!r} did not resolve")

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError(f"Web downloads cannot access non-public address {ip}")
    return raw
