"""
Ralph-style Coding Tool for OpenWebUI
======================================
A self-retrying code generation tool that:
  1. Plans complex tasks into small subtasks
  2. Writes each subtask as isolated Python
  3. Syntax-checks with ast (free, instant)
  4. Executes in a throwaway venv and captures real errors
  5. Retries with full error context until it passes or gives up
  6. Assembles all chunks into one final file
  7. Streams live status updates to OpenWebUI via event emitter

Tuned for fast 3B models on Jetson Orin Nano:
  - Single model for both worker and judge (no memory thrash)
  - Clone-based venv (fast copy vs slow creation)
  - Aggressive pip --no-cache-dir (saves storage)
  - Tiny judge prompts (fits 3B context easily)

Setup (run once before first use):
  python3 -m venv ~/.owui_base_venv
  ~/.owui_base_venv/bin/pip install --no-cache-dir \
      numpy pandas requests matplotlib scipy pillow pydantic
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Awaitable

# ── CONFIG ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"          # change to your 3B model name in Ollama
MAX_RETRIES = 6             # retries per subtask
EXEC_TIMEOUT = 15           # seconds before killing a run
BASE_VENV = os.path.expanduser("~/.owui_base_venv")  # pre-warmed venv to clone

# stdlib modules — these won't be pip-installed
STDLIB_MODULES = {
    "os", "sys", "re", "json", "math", "time", "datetime", "random",
    "collections", "itertools", "functools", "pathlib", "typing",
    "string", "io", "copy", "enum", "dataclasses", "abc", "contextlib",
    "threading", "multiprocessing", "subprocess", "tempfile", "shutil",
    "ast", "inspect", "logging", "unittest", "hashlib", "base64",
    "urllib", "http", "email", "csv", "sqlite3", "struct", "socket",
    "traceback", "textwrap", "argparse", "pprint", "queue", "heapq",
    "bisect", "array", "weakref", "gc", "platform", "signal", "stat",
    "glob", "fnmatch", "linecache", "tokenize", "keyword", "dis",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _call_model(prompt: str) -> str:
    """Call local Ollama synchronously. Returns response text."""
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
            result = json.loads(resp.read())
            return result.get("response", "").strip()
    except Exception as e:
        return f"ERROR calling model: {e}"


def _strip_fences(code: str) -> str:
    """Remove ```python / ``` markdown fences if model wraps output."""
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _syntax_check(code: str) -> tuple[bool, str]:
    """
    Parse code with ast. Returns (passed, clean_code_or_error_string).
    Free, instant, no LLM needed. Catches ~60% of failures before exec.
    """
    clean = _strip_fences(code)
    try:
        ast.parse(clean)
        return True, clean
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"


def _extract_imports(code: str) -> list[str]:
    """
    Walk AST to find third-party imports that need pip installing.
    Best-effort — catches standard import/from patterns.
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


def _ensure_base_venv():
    """
    Create the base venv if it doesn't exist yet.
    Called lazily on first tool use — takes ~30s but only runs once.
    """
    if not os.path.exists(BASE_VENV):
        subprocess.run(
            [sys.executable, "-m", "venv", BASE_VENV],
            check=True
        )
        pip = os.path.join(BASE_VENV, "bin", "pip")
        subprocess.run([
            pip, "install", "--no-cache-dir", "-q",
            "numpy", "pandas", "requests", "matplotlib",
            "scipy", "pillow", "pydantic"
        ])


def _run_in_venv(code: str, timeout: int = EXEC_TIMEOUT) -> dict:
    """
    Clone the base venv into a temp dir, run code, delete everything after.

    Returns dict:
        success         bool
        output          str   (stdout)
        error           str   (stderr or exception message)
        imports_tried   list  (packages we attempted to pip install)
    """
    _ensure_base_venv()

    tmp_dir = tempfile.mkdtemp(prefix="owui_run_")
    venv_dir = os.path.join(tmp_dir, "venv")
    code_file = os.path.join(tmp_dir, "code.py")

    try:
        # ── Clone base venv (fast file copy, ~0.5s vs 4-8s fresh create) ──
        shutil.copytree(BASE_VENV, venv_dir, symlinks=True)

        venv_python = os.path.join(venv_dir, "bin", "python3")
        venv_pip = os.path.join(venv_dir, "bin", "pip")

        # ── Install any third-party imports detected in the code ──────────
        third_party = _extract_imports(code)
        installed = []
        for pkg in third_party:
            result = subprocess.run(
                [venv_pip, "install", "--no-cache-dir", "-q", pkg],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                installed.append(pkg)

        # ── Write code file ───────────────────────────────────────────────
        with open(code_file, "w") as f:
            f.write(code)

        # ── Execute ───────────────────────────────────────────────────────
        run_result = subprocess.run(
            [venv_python, code_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmp_dir       # isolated working dir, not your host home
        )

        return {
            "success": run_result.returncode == 0,
            "output": run_result.stdout[:2000],   # cap to avoid flooding context
            "error": run_result.stderr[:2000],
            "imports_tried": installed
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Execution timed out after {timeout}s",
            "imports_tried": []
        }

    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "imports_tried": []
        }

    finally:
        # Always delete the temp dir — venv, code, any generated files
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── MAIN TOOL CLASS ───────────────────────────────────────────────────────────

class Tools:
    def __init__(self):
        self.max_retries = MAX_RETRIES

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _emit(
        self,
        emitter: Callable[[dict], Awaitable[None]],
        msg: str,
        done: bool = False
    ):
        """Send a status update to the OpenWebUI event stream."""
        if emitter:
            await emitter({
                "type": "status",
                "data": {
                    "description": msg,
                    "done": done
                }
            })

    def _plan(self, task: str) -> list[str]:
        """
        Ask the model to break the task into small, ordered subtasks.
        Each subtask should be ONE function or ONE class — small enough
        for a 3B model to write reliably in isolation.
        """
        prompt = f"""You are a software architect planning a Python project.

Break this task into small, ordered subtasks. Each subtask = ONE function or ONE class only.
Keep them small enough to implement in isolation without needing to see the other parts.

Task: {task}

Reply with a valid JSON array of strings ONLY. No explanation, no markdown.
Example: ["Write parse_input(text) function", "Write validate_data(data) function"]

JSON array:"""

        raw = _call_model(prompt)

        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            subtasks = json.loads(raw[start:end])
            if isinstance(subtasks, list) and len(subtasks) > 0:
                return subtasks
        except Exception:
            pass

        # Fallback: treat as single task
        return [task]

    def _write_chunk(self, task: str, subtask: str, context: str, error_context: str) -> str:
        """Ask the model to write ONE piece of code for a subtask."""
        prompt = f"""You are a Python developer. Write ONLY the Python code for the task below.

Rules:
- Pure Python only. No markdown. No explanation. No ``` fences.
- Write only what is asked — one function or one class.
- Include necessary imports at the top of your snippet.

Overall goal: {task}
Already built: {context if context else "Nothing yet — this is the first piece."}
Your task now: {subtask}
{f"PREVIOUS ATTEMPT FAILED: {error_context}" if error_context else ""}

Python code:"""

        return _call_model(prompt)

    def _judge(self, subtask: str, code: str, output: str) -> tuple[bool, str]:
        """
        Ask the model to judge if the code correctly implements the subtask.
        Kept tiny so a 3B model can reason about it reliably.
        Returns (passed, reason).
        """
        prompt = f"""Does this Python code correctly implement the task?

Task: {subtask}
Code:
{code[:800]}
Execution output: {output[:400] if output else "(no output)"}

Reply with PASS or FAIL followed by one sentence reason. Example: "PASS - function returns correct values"
"""
        verdict = _call_model(prompt)
        passed = "PASS" in verdict.upper()
        reason = verdict.strip()[:200]
        return passed, reason

    def _assemble(self, task: str, chunks: list[str]) -> str:
        """Ask the model to stitch all chunks into one clean final file."""
        separator = "\n" + ("=" * 50) + "\n"
        all_chunks = separator.join(chunks)

        prompt = f"""You are a Python developer. Combine these code chunks into one complete, clean Python file.

Rules:
- Pure Python only. No markdown. No ``` fences. No explanation.
- Remove duplicate imports. Keep one copy of each.
- Ensure correct ordering (imports first, then helpers, then main logic).
- Add a main() function and if __name__ == "__main__": block if appropriate.

Original goal: {task}

Code chunks:
{all_chunks}

Final combined Python file:"""

        return _call_model(prompt)

    # ── Public tool function ───────────────────────────────────────────────

    async def build_code(
        self,
        task: str,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> str:
        """
        Build complex Python code using a plan-chunk-test-retry loop.

        Args:
            task: What you want the code to do. Be as detailed as possible.

        Returns the final assembled Python code, or the best attempt if assembly fails.
        """
        emit = lambda msg, done=False: self._emit(__event_emitter__, msg, done)

        # ── PHASE 1: PLAN ─────────────────────────────────────────────────
        await emit("🧠 Planning subtasks...")
        subtasks = self._plan(task)
        await emit(f"📋 {len(subtasks)} subtask(s) planned")

        # ── PHASE 2: BUILD EACH CHUNK ─────────────────────────────────────
        built_chunks = []
        context_summary = ""   # short running summary of what's done

        for i, subtask in enumerate(subtasks):
            label = subtask[:70] + "..." if len(subtask) > 70 else subtask
            await emit(f"🔨 [{i+1}/{len(subtasks)}] {label}")

            attempt = 0
            success = False
            error_context = ""

            while attempt < self.max_retries and not success:
                attempt += 1
                await emit(f"  ✏️  Writing — attempt {attempt}/{self.max_retries}...")

                # Write
                code = self._write_chunk(task, subtask, context_summary, error_context)

                # Syntax check (free, instant)
                syntax_ok, code_or_err = _syntax_check(code)
                if not syntax_ok:
                    error_context = f"Syntax error: {code_or_err}"
                    await emit(f"  ⚠️  Attempt {attempt}: {error_context[:100]}")
                    continue

                clean_code = code_or_err

                # Execute in throwaway venv
                await emit(f"  🚀 Running in venv...")
                run = _run_in_venv(clean_code)

                if not run["success"]:
                    error_context = run["error"] or run["output"]
                    short_err = error_context[:150].replace("\n", " ")
                    await emit(f"  ❌ Attempt {attempt} runtime error: {short_err}")
                    continue

                # Judge (LLM quality check)
                await emit(f"  🔍 Judging output...")
                passed, reason = self._judge(subtask, clean_code, run["output"])

                if passed:
                    built_chunks.append(clean_code)
                    context_summary += f"\n- {subtask}: implemented"
                    await emit(f"  ✅ Passed on attempt {attempt}: {reason[:80]}")
                    success = True
                else:
                    error_context = f"Judge said FAIL: {reason}"
                    await emit(f"  ❌ Attempt {attempt}: {reason[:100]}")

            if not success:
                await emit(f"  🛑 Giving up on subtask {i+1} after {self.max_retries} attempts")
                built_chunks.append(
                    f"# TODO: subtask {i+1} failed after {self.max_retries} attempts\n"
                    f"# Task was: {subtask}\n"
                    f"# Last error: {error_context[:200]}\n"
                    "pass\n"
                )

        # ── PHASE 3: ASSEMBLE ─────────────────────────────────────────────
        await emit("🔗 Assembling final file...")

        if len(built_chunks) == 1:
            final_code = built_chunks[0]
        else:
            final_code = self._assemble(task, built_chunks)

        # Final syntax check on assembled output
        syntax_ok, result = _syntax_check(final_code)

        if syntax_ok:
            # One final execution to confirm it runs
            await emit("🚀 Final validation run...")
            run = _run_in_venv(result)
            if run["success"]:
                await emit("✅ Done! All checks passed.", done=True)
                output_note = f"\n# Output during test run:\n# {run['output'][:300]}" if run["output"] else ""
                return f"```python\n{result}{output_note}\n```"
            else:
                short_err = (run["error"] or "unknown error")[:200]
                await emit(f"⚠️ Assembled code has runtime issues: {short_err}", done=True)
                return (
                    f"⚠️ Code assembled but has runtime issues. Error:\n```\n{short_err}\n```\n\n"
                    f"Code (may need manual fixes):\n```python\n{result}\n```"
                )
        else:
            await emit("⚠️ Assembly has syntax errors — returning chunks separately.", done=True)
            joined = "\n\n# ── NEXT CHUNK ──\n\n".join(built_chunks)
            return (
                f"⚠️ Final assembly failed syntax check. Returning chunks:\n\n"
                f"```python\n{joined}\n```"
            )
