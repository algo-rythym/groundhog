"""
Ralph-Style Coding Tool for OpenWebUI — Docker Edition
=======================================================
A self-retrying code generation tool that:
  1. Plans complex tasks into small subtasks
  2. Writes each subtask as isolated Python
  3. Syntax-checks with ast (free, instant, no LLM)
  4. Spins up a Docker container, pip installs detected deps, runs the code
  5. Retries with the full error traceback fed back into the prompt
  6. Assembles all passing chunks into one final file
  7. Streams live status updates to OpenWebUI throughout

Docker container per run:
  - python:3.11-slim image
  - --rm (auto-deleted after run)
  - Memory and CPU capped
  - Network ON (so pip install works inside)
  - Auto-pulls image on first use if not present

Setup (run once on Jetson):
  docker pull python:3.11-slim
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
from typing import Callable, Awaitable

# ── CONFIG ────────────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL        = "llama3.2"        # ← change to your Ollama model name
MAX_RETRIES  = 6                 # retry attempts per subtask before giving up
EXEC_TIMEOUT = 30                # seconds before killing the container
DOCKER_IMAGE = "python:3.11-slim"
DOCKER_MEM   = "256m"            # RAM cap per container
DOCKER_CPUS  = "0.5"             # CPU cap per container

# stdlib modules — won't be pip-installed
STDLIB_MODULES = {
    "os", "sys", "re", "json", "math", "time", "datetime", "random",
    "collections", "itertools", "functools", "pathlib", "typing",
    "string", "io", "copy", "enum", "dataclasses", "abc", "contextlib",
    "threading", "multiprocessing", "subprocess", "tempfile", "shutil",
    "ast", "inspect", "logging", "unittest", "hashlib", "base64",
    "urllib", "http", "email", "csv", "sqlite3", "struct", "socket",
    "traceback", "textwrap", "argparse", "pprint", "queue", "heapq",
    "bisect", "array", "weakref", "gc", "platform", "signal", "glob",
}

# ── DOCKER HELPERS ────────────────────────────────────────────────────────────

def _image_exists() -> bool:
    """Check if the Docker image is already pulled locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", DOCKER_IMAGE],
        capture_output=True
    )
    return result.returncode == 0


def _pull_image() -> tuple[bool, str]:
    """Pull the Docker image. Returns (success, error_message)."""
    result = subprocess.run(
        ["docker", "pull", DOCKER_IMAGE],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode == 0:
        return True, ""
    return False, result.stderr[:500]


def _extract_imports(code: str) -> list[str]:
    """
    Walk the AST to find third-party imports that need pip installing.
    Skips anything in stdlib.
    """
    pkgs = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pkg = alias.name.split(".")[0]
                    if pkg and pkg not in STDLIB_MODULES:
                        pkgs.append(pkg)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    pkg = node.module.split(".")[0]
                    if pkg and pkg not in STDLIB_MODULES:
                        pkgs.append(pkg)
    except SyntaxError:
        pass
    return list(set(pkgs))


def _build_runner_script(code: str, packages: list[str]) -> str:
    """
    Build a self-contained shell script that:
      1. pip installs detected packages
      2. runs the code
    This runs inside the container as a single entrypoint.
    """
    pip_lines = ""
    if packages:
        pkg_list = " ".join(packages)
        pip_lines = f"pip install --no-cache-dir -q {pkg_list}\n"

    return f"""#!/bin/sh
set -e
{pip_lines}
python3 /code.py
"""


def _run_in_docker(code: str, timeout: int = EXEC_TIMEOUT) -> dict:
    """
    Write code to a temp file, mount it into a Docker container,
    pip install detected deps, execute, capture output, auto-delete container.

    Returns:
        success  bool
        output   str   stdout
        error    str   stderr or exception
        packages list  packages that were detected for install
    """
    packages = _extract_imports(code)

    # Write code and runner script to temp files
    tmp_dir = tempfile.mkdtemp(prefix="ralph_")
    code_path   = os.path.join(tmp_dir, "code.py")
    runner_path = os.path.join(tmp_dir, "runner.sh")

    try:
        with open(code_path, "w") as f:
            f.write(code)

        runner = _build_runner_script(code, packages)
        with open(runner_path, "w") as f:
            f.write(runner)
        os.chmod(runner_path, 0o755)

        result = subprocess.run([
            "docker", "run",
            "--rm",                              # auto-delete when done
            f"--memory={DOCKER_MEM}",            # RAM cap
            f"--cpus={DOCKER_CPUS}",             # CPU cap
            "-v", f"{code_path}:/code.py:ro",    # mount code read-only
            "-v", f"{runner_path}:/runner.sh:ro",# mount runner read-only
            DOCKER_IMAGE,
            "/bin/sh", "/runner.sh"
        ], capture_output=True, text=True, timeout=timeout)

        return {
            "success": result.returncode == 0,
            "output":  result.stdout[:2000],
            "error":   result.stderr[:2000],
            "packages": packages
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output":  "",
            "error":   f"Container timed out after {timeout}s",
            "packages": packages
        }

    except Exception as e:
        return {
            "success": False,
            "output":  "",
            "error":   str(e),
            "packages": packages
        }

    finally:
        # Clean up temp files from host
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── CODE HELPERS ──────────────────────────────────────────────────────────────

def _call_model(prompt: str) -> str:
    """Call local Ollama synchronously."""
    try:
        import urllib.request
        data = json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }).encode()
        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception as e:
        return f"ERROR calling model: {e}"


def _strip_fences(code: str) -> str:
    """Remove ```python / ``` markdown fences if the model wraps output."""
    lines = code.split("\n")
    return "\n".join(
        l for l in lines if not l.strip().startswith("```")
    ).strip()


def _syntax_check(code: str) -> tuple[bool, str]:
    """
    Parse with ast. Returns (passed, clean_code_or_error).
    Free, instant — catches ~60% of failures before Docker spin-up.
    """
    clean = _strip_fences(code)
    try:
        ast.parse(clean)
        return True, clean
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"


# ── MAIN TOOL CLASS ───────────────────────────────────────────────────────────

class Tools:
    def __init__(self):
        self.max_retries  = MAX_RETRIES
        self._image_ready = False   # cached after first pull check

    # ── Emitter ───────────────────────────────────────────────────────────

    async def _emit(
        self,
        emitter: Callable[[dict], Awaitable[None]],
        msg: str,
        done: bool = False
    ):
        if emitter:
            await emitter({
                "type": "status",
                "data": {"description": msg, "done": done}
            })

    # ── Docker readiness ──────────────────────────────────────────────────

    async def _ensure_image(self, emitter):
        """Pull Docker image if not present. Shows status in UI."""
        if self._image_ready:
            return True

        if _image_exists():
            self._image_ready = True
            return True

        await self._emit(emitter, f"📦 Docker image not found locally. Pulling {DOCKER_IMAGE}...")
        success, err = _pull_image()

        if success:
            self._image_ready = True
            await self._emit(emitter, f"✅ Docker image ready.")
            return True
        else:
            await self._emit(emitter, f"❌ Failed to pull Docker image: {err}", done=True)
            return False

    # ── LLM calls ─────────────────────────────────────────────────────────

    def _plan(self, task: str) -> list[str]:
        """Break task into small ordered subtasks — one function/class each."""
        prompt = f"""You are a software architect planning a Python project.

Break this task into small ordered subtasks. Each subtask = ONE function or ONE class only.
Small enough to implement in isolation without seeing the other parts.

Task: {task}

Reply with a valid JSON array of strings ONLY. No explanation. No markdown.
Example: ["Write parse_input(text) function", "Write validate_data(data) function"]

JSON array:"""

        raw = _call_model(prompt)
        try:
            start = raw.index("[")
            end   = raw.rindex("]") + 1
            result = json.loads(raw[start:end])
            if isinstance(result, list) and result:
                return result
        except Exception:
            pass
        return [task]  # fallback

    def _write_chunk(self, task: str, subtask: str, context: str, error_context: str) -> str:
        """Ask the model to write ONE function or class for a subtask."""
        error_section = ""
        if error_context:
            error_section = f"\nPREVIOUS ATTEMPT FAILED WITH THIS ERROR — fix it:\n{error_context}\n"

        prompt = f"""You are a Python developer. Write ONLY the Python code for the task below.

Rules:
- Pure Python only. No markdown. No explanation. No ``` fences.
- Write only what is asked — one function or one class.
- Include all necessary imports at the top of your snippet.
- If you need third-party packages, import them normally — they will be installed automatically.

Overall goal: {task}
Already built: {context if context else "Nothing yet — this is the first piece."}
Your task now: {subtask}{error_section}

Python code:"""

        return _call_model(prompt)

    def _judge(self, subtask: str, code: str, output: str) -> tuple[bool, str]:
        """Quick LLM quality check — kept tiny for 3B model reliability."""
        prompt = f"""Does this Python code correctly implement the task?

Task: {subtask}
Code:
{code[:800]}
Execution output: {output[:400] if output else "(no output — code ran silently)"}

Reply PASS or FAIL and one short reason.
Example: "PASS - function returns correct result"
"""
        verdict = _call_model(prompt)
        passed  = "PASS" in verdict.upper()
        reason  = verdict.strip()[:200]
        return passed, reason

    def _assemble(self, task: str, chunks: list[str]) -> str:
        """Stitch all chunks into one clean final Python file."""
        separator  = "\n" + ("=" * 50) + "\n"
        all_chunks = separator.join(chunks)

        prompt = f"""You are a Python developer. Combine these code chunks into one complete clean Python file.

Rules:
- Pure Python only. No markdown. No ``` fences. No explanation.
- Remove duplicate imports — keep one copy of each.
- Correct ordering: imports first, helpers next, main logic last.
- Add a main() function and if __name__ == "__main__": block if appropriate.

Original goal: {task}

Chunks:
{all_chunks}

Final combined Python file:"""

        return _call_model(prompt)

    # ── Public tool entry point ────────────────────────────────────────────

    async def build_code(
        self,
        task: str,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> str:
        """
        Build complex Python code using a plan → chunk → test → retry loop.

        Each chunk is executed inside a Docker container. Pip packages are
        detected automatically from imports and installed inside the container.
        The container is deleted after every run.

        Args:
            task: Describe what you want the code to do. More detail = better results.
        """
        emit = lambda msg, done=False: self._emit(__event_emitter__, msg, done)

        # ── Docker check ──────────────────────────────────────────────────
        image_ok = await self._ensure_image(__event_emitter__)
        if not image_ok:
            return "❌ Cannot proceed — Docker image unavailable. Check that Docker is running."

        # ── PHASE 1: PLAN ─────────────────────────────────────────────────
        await emit("🧠 Planning subtasks...")
        subtasks = self._plan(task)
        await emit(f"📋 {len(subtasks)} subtask(s) planned")

        # ── PHASE 2: BUILD EACH CHUNK ─────────────────────────────────────
        built_chunks   = []
        context_summary = ""

        for i, subtask in enumerate(subtasks):
            label = (subtask[:65] + "...") if len(subtask) > 65 else subtask
            await emit(f"🔨 [{i+1}/{len(subtasks)}] {label}")

            attempt       = 0
            success       = False
            error_context = ""

            while attempt < self.max_retries and not success:
                attempt += 1
                await emit(f"  ✏️  Writing — attempt {attempt}/{self.max_retries}...")

                # Write
                code = self._write_chunk(task, subtask, context_summary, error_context)

                # Syntax check — free, instant, no Docker needed
                syntax_ok, code_or_err = _syntax_check(code)
                if not syntax_ok:
                    error_context = code_or_err
                    await emit(f"  ⚠️  Syntax error: {error_context[:100]}")
                    continue

                clean_code = code_or_err

                # Show what packages will be installed if any
                pkgs = _extract_imports(clean_code)
                if pkgs:
                    await emit(f"  📦 Detected packages: {', '.join(pkgs)}")

                # Spin up Docker container
                await emit(f"  🐳 Creating Docker container...")
                run = _run_in_docker(clean_code)

                if pkgs:
                    await emit(f"  ⬇️  Installing packages inside container...")

                if not run["success"]:
                    error_context = (run["error"] or run["output"])
                    short_err     = error_context[:150].replace("\n", " ")
                    await emit(f"  ❌ Attempt {attempt} failed: {short_err}")
                    await emit(f"  🗑️  Container removed")
                    continue

                await emit(f"  🗑️  Container removed")

                # Judge
                await emit(f"  🔍 Judging output...")
                passed, reason = self._judge(subtask, clean_code, run["output"])

                if passed:
                    built_chunks.append(clean_code)
                    context_summary += f"\n- {subtask}: done"
                    await emit(f"  ✅ Passed on attempt {attempt}: {reason[:80]}")
                    success = True
                else:
                    error_context = f"Code ran but judge said FAIL: {reason}"
                    await emit(f"  ❌ Attempt {attempt}: {reason[:100]}")

            if not success:
                await emit(f"  🛑 Gave up on subtask {i+1} after {self.max_retries} attempts")
                built_chunks.append(
                    f"# TODO: subtask {i+1} failed after {self.max_retries} attempts\n"
                    f"# Task: {subtask}\n"
                    f"# Last error: {error_context[:200]}\n"
                    "pass\n"
                )

        # ── PHASE 3: ASSEMBLE ─────────────────────────────────────────────
        await emit("🔗 Assembling final file...")

        final_code = self._assemble(task, built_chunks) if len(built_chunks) > 1 else built_chunks[0]

        syntax_ok, result = _syntax_check(final_code)

        if not syntax_ok:
            await emit("⚠️ Assembly has syntax errors — returning chunks separately.", done=True)
            joined = "\n\n# ── NEXT CHUNK ──\n\n".join(built_chunks)
            return f"⚠️ Final assembly failed. Returning chunks:\n\n```python\n{joined}\n```"

        # Final validation run
        pkgs = _extract_imports(result)
        if pkgs:
            await emit(f"  📦 Final packages: {', '.join(pkgs)}")
        await emit("🐳 Creating Docker container for final validation...")
        final_run = _run_in_docker(result)
        await emit("🗑️  Container removed")

        if final_run["success"]:
            await emit("✅ All done! Final code validated successfully.", done=True)
            output_note = ""
            if final_run["output"].strip():
                preview = final_run["output"].strip()[:300]
                output_note = f"\n\n# --- Test run output ---\n# " + preview.replace("\n", "\n# ")
            return f"```python\n{result}{output_note}\n```"

        else:
            short_err = (final_run["error"] or "unknown error")[:300]
            await emit(f"⚠️ Final code has runtime issues: {short_err[:80]}", done=True)
            return (
                f"⚠️ Code assembled but final run had issues:\n"
                f"```\n{short_err}\n```\n\n"
                f"Code (may need minor fixes):\n```python\n{result}\n```"
            )
