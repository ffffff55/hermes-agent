"""Stdio MCP SDK load must not pull the ASGI server stack.

Regression for #149388. ``mcp`` 2.x package init imports starlette, uvicorn
and sse_starlette. A stdio-only ``_ensure_mcp_sdk()`` has to leave that
stack unloaded, while still binding a usable ClientSession and stdio
transport. HTTP/SSE clients load on their own gate.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_ASGI = ("starlette", "uvicorn", "sse_starlette")
_REPO = str(Path(__file__).resolve().parents[2])


def _script(body: str) -> str:
    """Child source. ``ASGI_ROOTS`` is replaced so the script stays a plain string."""
    return body.replace("ASGI_ROOTS", repr(_ASGI))


def _require_mcp():
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("mcp SDK not installed")


def _fresh(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=_REPO,
    )


def test_stdio_sdk_load_does_not_import_asgi_server_stack():
    """Fresh interpreter: stdio ensure binds the client and leaves ASGI unloaded.

    A follow-up ``from mcp import ClientSession`` (how computer-use opens its
    stdio session) must keep the server stack unloaded too — that import is
    what used to execute the package init.
    """
    _require_mcp()
    proc = _fresh(_script(
        "import sys\n"
        "from tools import mcp_tool\n"
        "def asgi():\n"
        "    return sorted({m.split('.')[0] for m in sys.modules if m.split('.')[0] in ASGI_ROOTS})\n"
        "assert not asgi(), asgi()\n"
        "assert mcp_tool._ensure_mcp_sdk() is True\n"
        "assert mcp_tool._ensure_mcp_sdk() is True\n"
        "assert not asgi(), asgi()\n"
        "assert mcp_tool.ClientSession.__module__ == 'mcp.client.session'\n"
        "assert callable(getattr(mcp_tool.ClientSession, 'initialize'))\n"
        "assert mcp_tool.stdio_client.__module__ == 'mcp.client.stdio'\n"
        "assert callable(mcp_tool.stdio_client)\n"
        "params = mcp_tool.StdioServerParameters(command='true')\n"
        "assert params.command == 'true'\n"
        "import mcp.client.session as session\n"
        "import mcp.types as types\n"
        "assert mcp_tool.LATEST_HANDSHAKE_VERSION == session.LATEST_HANDSHAKE_VERSION\n"
        "assert mcp_tool.LATEST_PROTOCOL_VERSION == types.LATEST_PROTOCOL_VERSION\n"
        "assert 'mcp.client.streamable_http' not in sys.modules\n"
        "assert 'mcp.client.sse' not in sys.modules\n"
        "assert 'mcp.server' not in sys.modules\n"
        "from mcp import ClientSession, StdioServerParameters\n"
        "assert ClientSession is mcp_tool.ClientSession\n"
        "assert StdioServerParameters is mcp_tool.StdioServerParameters\n"
        "from mcp.client.stdio import stdio_client\n"
        "assert stdio_client is mcp_tool.stdio_client\n"
        "assert not asgi(), asgi()\n"
        "print('ok')\n"
    ))
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_http_client_load_is_separate_from_stdio():
    """Streamable HTTP / SSE clients appear only after the HTTP gate runs."""
    _require_mcp()
    proc = _fresh(_script(
        "import sys\n"
        "from tools import mcp_tool\n"
        "def asgi():\n"
        "    return sorted({m.split('.')[0] for m in sys.modules if m.split('.')[0] in ASGI_ROOTS})\n"
        "assert mcp_tool._ensure_mcp_sdk() is True\n"
        "assert 'mcp.client.streamable_http' not in sys.modules\n"
        "assert 'mcp.client.sse' not in sys.modules\n"
        "assert mcp_tool._MCP_HTTP_AVAILABLE is False\n"
        "assert mcp_tool._ensure_mcp_http_sdk() is True\n"
        "assert callable(mcp_tool.streamable_http_client)\n"
        "assert mcp_tool.streamable_http_client.__module__ == 'mcp.client.streamable_http'\n"
        "assert callable(mcp_tool.sse_client)\n"
        "assert mcp_tool.sse_client.__module__ == 'mcp.client.sse'\n"
        "assert mcp_tool._MCP_HTTP_AVAILABLE is True\n"
        "first = mcp_tool.streamable_http_client\n"
        "assert mcp_tool._ensure_mcp_http_sdk() is True\n"
        "assert mcp_tool.streamable_http_client is first\n"
        "# The client transports themselves do not import the server stack.\n"
        "assert not asgi(), asgi()\n"
        "print('ok')\n"
    ))
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_unavailable_gate_imports_nothing():
    """A patched ``_MCP_AVAILABLE=False`` must not import the SDK at all."""
    proc = _fresh(
        "import sys\n"
        "from tools import mcp_tool\n"
        "mcp_tool._MCP_AVAILABLE = False\n"
        "mcp_tool._MCP_SDK_IMPORT_ATTEMPTED = False\n"
        "mcp_tool._MCP_HTTP_IMPORT_ATTEMPTED = False\n"
        "mcp_tool.ClientSession = None\n"
        "assert mcp_tool._ensure_mcp_sdk() is False\n"
        "assert mcp_tool._ensure_mcp_http_sdk() is False\n"
        "assert 'mcp.client.session' not in sys.modules\n"
        "assert 'mcp.client.stdio' not in sys.modules\n"
        "assert 'mcp.client.streamable_http' not in sys.modules\n"
        "assert 'starlette' not in sys.modules\n"
        "print('ok')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
