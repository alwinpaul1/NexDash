"""OAuth 2.1 for the NexDash MCP server — secure, stateless, bring-your-own-key.

Why this exists
---------------
Claude **Desktop / web's** custom-connector UI only speaks OAuth (it can't send a
custom header), so a non-technical user can't paste an API key. With OAuth they
just click **Connect**, a consent page asks for their TomTom key, and they're in.

Security model
--------------
* The consent page collects the user's TomTom key and validates it with one live
  TomTom call before issuing anything.
* That key is **AEAD-encrypted (Fernet) INTO the access token**. The server stores
  **no** user keys — each key lives only inside that user's own opaque bearer
  token and is decrypted per request. Tokens are signed+encrypted, so a client
  can neither read nor forge them. Nothing to leak at rest.
* Auth codes and dynamically-registered clients are kept in-process and are
  short-lived / single-use (losing them on restart just means a re-connect).
* The Fernet key is derived from ``MCP_TOKEN_SECRET`` so tokens survive deploys;
  if it's unset a random one is generated (tokens then reset on restart).

The SDK (``FastMCP(auth_server_provider=..., auth=AuthSettings(...))``) mounts the
discovery / registration / authorize / token endpoints and enforces the token on
every request; this module supplies the provider logic + the consent page.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from typing import Optional
from urllib.parse import urlencode

from cryptography.fernet import Fernet, InvalidToken
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
)
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

SCOPE = "nexdash"
_AUTH_CODE_TTL = 300  # 5 min
_TOKEN_TTL = 8 * 3600  # 8 h


def _fernet() -> Fernet:
    """Fernet built from ``MCP_TOKEN_SECRET`` (sha256 → urlsafe-b64 key), or a
    process-random key if unset (tokens then don't survive a restart)."""
    secret = os.environ.get("MCP_TOKEN_SECRET")
    if not secret:
        secret = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


_FERNET = _fernet()


def encode_token(tomtom_key: str, client_id: str) -> tuple[str, int]:
    """Encrypt the user's TomTom key into an opaque access token. Returns
    ``(token, expires_at_epoch)``."""
    exp = int(time.time()) + _TOKEN_TTL
    payload = json.dumps({"k": tomtom_key, "cid": client_id, "exp": exp}).encode()
    return _FERNET.encrypt(payload).decode(), exp


def decode_token(token: str) -> Optional[dict]:
    """Decrypt + validate a token. Returns ``{k, cid, exp}`` or ``None``."""
    try:
        data = json.loads(_FERNET.decrypt(token.encode()).decode())
    except (InvalidToken, ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not data.get("k"):
        return None
    if int(data.get("exp", 0)) < int(time.time()):
        return None
    return data


class NexDashOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    """In-process OAuth AS. Clients + auth codes are in memory; access tokens are
    stateless (encrypted). Each auth code carries the user's TomTom key."""

    def __init__(self) -> None:
        self._clients: dict[str, OAuthClientInformationFull] = {}
        # code -> (AuthorizationCode, tomtom_key)
        self._codes: dict[str, tuple[AuthorizationCode, str]] = {}
        # txn -> (client_id, AuthorizationParams) for the consent round-trip
        self._pending: dict[str, tuple[str, AuthorizationParams]] = {}

    # -- dynamic client registration ------------------------------------- #
    async def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info

    # -- authorize: stash params, send the user to the consent page ------- #
    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        txn = secrets.token_urlsafe(24)
        self._pending[txn] = (client.client_id, params)
        return f"/consent?{urlencode({'txn': txn})}"

    # Called by the consent POST handler once the user supplies their key.
    def complete_consent(self, txn: str, tomtom_key: str) -> Optional[str]:
        """Mint an auth code for a consented transaction; return the client
        redirect URL (code + state), or ``None`` if the txn is unknown/expired."""
        entry = self._pending.pop(txn, None)
        if entry is None:
            return None
        client_id, params = entry
        code_str = secrets.token_urlsafe(32)
        self._codes[code_str] = (
            AuthorizationCode(
                code=code_str,
                scopes=params.scopes or [SCOPE],
                expires_at=time.time() + _AUTH_CODE_TTL,
                client_id=client_id,
                code_challenge=params.code_challenge,
                redirect_uri=params.redirect_uri,
                redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
                resource=params.resource,
            ),
            tomtom_key,
        )
        q = {"code": code_str}
        if params.state:
            q["state"] = params.state
        return f"{params.redirect_uri}?{urlencode(q)}"

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> Optional[AuthorizationCode]:
        entry = self._codes.get(authorization_code)
        if entry is None:
            return None
        code, _ = entry
        if code.client_id != client.client_id or code.expires_at < time.time():
            return None
        return code

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        entry = self._codes.pop(authorization_code.code, None)  # single-use
        if entry is None:
            raise ValueError("invalid authorization code")
        _, tomtom_key = entry
        token, exp = encode_token(tomtom_key, client.client_id)
        return OAuthToken(
            access_token=token,
            token_type="Bearer",
            expires_in=max(1, exp - int(time.time())),
            scope=SCOPE,
        )

    # -- token verification (stateless) ---------------------------------- #
    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        data = decode_token(token)
        if data is None:
            return None
        return AccessToken(
            token=token,
            client_id=str(data.get("cid", "")),
            scopes=[SCOPE],
            expires_at=int(data["exp"]),
            claims={"tomtom_key": data["k"]},
        )

    # -- refresh tokens: not issued -------------------------------------- #
    async def load_refresh_token(self, client, refresh_token):  # noqa: ANN001
        return None

    async def exchange_refresh_token(self, client, refresh_token, scopes):  # noqa: ANN001
        raise ValueError("refresh tokens are not supported")

    async def revoke_token(self, token):  # noqa: ANN001
        return None


# --------------------------------------------------------------------------- #
# Consent page + AuthSettings wiring
# --------------------------------------------------------------------------- #
_CONSENT_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Connect NexDash</title><style>
body{{font-family:Inter,system-ui,sans-serif;background:#0b0f0e;color:#e6edf0;display:flex;
min-height:100vh;align-items:center;justify-content:center;margin:0}}
.card{{background:#0e1413;border:1px solid #28322f;border-radius:16px;padding:28px;max-width:420px;width:90%}}
h1{{font-size:18px;margin:0 0 6px}} p{{color:#9ca3af;font-size:13px;line-height:1.5}}
label{{display:block;font-size:12px;color:#9ca3af;margin:16px 0 6px}}
input{{width:100%;box-sizing:border-box;padding:10px 12px;border-radius:8px;border:1px solid #28322f;
background:#0b0f0e;color:#e6edf0;font-size:14px}}
button{{margin-top:18px;width:100%;padding:11px;border:0;border-radius:999px;background:#10b981;
color:#04140e;font-weight:600;font-size:14px;cursor:pointer}}
a{{color:#34d399}} .err{{color:#f87171;font-size:12px;margin-top:10px}}
.hint{{font-size:12px;color:#7a8a86;line-height:1.5;margin-top:12px}}
</style></head><body><div class="card">
<h1>Connect NexDash MCP</h1>
<p>The NexDash MCP gives your AI tool EV-truck route &amp; range intelligence, running on
<b>your</b> TomTom key. It's validated, then sealed inside your own access token and
<b>never stored</b> on our server. Paste your key to finish.</p>
<p class="hint"><b>No key?</b> <a href="https://developer.tomtom.com" target="_blank">Sign up free at TomTom</a>.
In your dashboard, open <b>Keys &rarr; API &amp; SDK Keys</b> in the left sidebar and copy the value in the
<b>Key</b> column (a key is created with your account; use <b>Create Key</b> for a new one). Paste it above.</p>
<form method="post" action="/consent">
<input type="hidden" name="txn" value="{txn}">
<label>TomTom API key</label>
<input name="tomtom_key" type="password" autocomplete="off" placeholder="your TomTom key" required autofocus>
{err}
<button type="submit">Authorize</button>
</form></div></body></html>"""


def register_consent_routes(mcp, provider: NexDashOAuthProvider) -> None:
    """Add the GET (form) + POST (validate key → mint code → redirect) consent
    routes to the FastMCP app."""
    from nexdash import tomtom

    @mcp.custom_route("/consent", methods=["GET"])
    async def consent_get(request: Request):  # noqa: ANN202
        txn = request.query_params.get("txn", "")
        return HTMLResponse(_CONSENT_HTML.format(txn=txn, err=""))

    @mcp.custom_route("/consent", methods=["POST"])
    async def consent_post(request: Request):  # noqa: ANN202
        form = await request.form()
        txn = str(form.get("txn", ""))
        key = str(form.get("tomtom_key", "")).strip()

        def _page(msg: str):
            return HTMLResponse(
                _CONSENT_HTML.format(txn=txn, err=f'<p class="err">{msg}</p>'),
                status_code=400,
            )

        if not txn:
            return _page("Session expired — start the connection again from your client.")
        if not key:
            return _page("Enter your TomTom API key.")
        # Validate the key with one live TomTom call before issuing a token.
        tok = tomtom.set_request_api_key(key)
        try:
            tomtom.geocode("Berlin")
        except Exception:  # noqa: BLE001
            return _page("That TomTom key didn't work (geocode failed). Check the key and try again.")
        finally:
            tomtom.reset_request_api_key(tok)
        redirect = provider.complete_consent(txn, key)
        if redirect is None:
            return _page("Session expired — start the connection again from your client.")
        return RedirectResponse(redirect, status_code=302)


def build_auth(public_url: str):
    """Return ``(provider, AuthSettings)`` for a combined AS+RS at ``public_url``."""
    provider = NexDashOAuthProvider()
    settings = AuthSettings(
        issuer_url=public_url,
        resource_server_url=public_url,
        client_registration_options=ClientRegistrationOptions(
            enabled=True, valid_scopes=[SCOPE], default_scopes=[SCOPE]
        ),
        required_scopes=[SCOPE],
    )
    return provider, settings
