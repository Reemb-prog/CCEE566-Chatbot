"""Start Streamlit headless on a free local port, probe its health endpoint and main page, then stop it.

Usage:  python scripts/smoke_app.py
Exit code 0 only if /_stcore/health returns "ok" and the app script ran without an exception.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read().decode("utf-8", "replace")


def main() -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("SKIP: streamlit is not installed in this interpreter")
        return 2
    # 1) run the script once with Streamlit's AppTest (executes app.py, surfaces exceptions)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    if at.exception:
        print("FAIL: app.py raised:", [e.value for e in at.exception])
        return 1
    at.chat_input[0].set_value("How many credits is CCEE 566?").run()
    if at.exception:
        print("FAIL: chat turn raised:", [e.value for e in at.exception])
        return 1
    texts = " ".join(m.value for m in at.markdown)
    ok_turn = "3-credit" in texts and "CCEE 566" in texts
    print(f"AppTest: script ran, chat turn answered={ok_turn}")
    if not ok_turn:
        return 1

    # 2) real server + HTTP health
    port = free_port()
    env = {**os.environ, "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
           "--server.port", str(port), "--server.address", "127.0.0.1"]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                status, body = get(f"http://127.0.0.1:{port}/_stcore/health")
                if status == 200 and body.strip() == "ok":
                    s2, page = get(f"http://127.0.0.1:{port}/")
                    print(f"HTTP: /_stcore/health -> {body.strip()}; / -> {s2} ({len(page)} bytes)")
                    print("PASS: Streamlit server healthy on port", port)
                    return 0
            except OSError:
                pass
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        print("FAIL: server did not become healthy")
        return 1
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        if "Traceback" in (out or ""):
            print("Server log contained a traceback:\n", out[-2000:])


if __name__ == "__main__":
    raise SystemExit(main())
