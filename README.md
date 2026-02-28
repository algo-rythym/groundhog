# groundhog
# Ralph-Style Coding Tool — Full Setup Guide
## Jetson Orin Nano + OpenWebUI + Ollama

---

## What This Does

This tool lets OpenWebUI generate complex Python code using a self-retrying loop:

1. **Plans** your task into small, manageable subtasks
2. **Writes** each subtask as isolated Python code
3. **Syntax-checks** instantly using Python's `ast` module (no LLM needed)
4. **Executes** the code in a throwaway Python venv, capturing real errors
5. **Retries** with full error context fed back into the prompt
6. **Assembles** all passing chunks into one final file
7. **Streams live status** to OpenWebUI as it works

---

## Prerequisites

Before installing the tool, confirm you have these running:

```bash
# Check Docker
docker ps | grep open-webui

# Check Ollama
ollama list

# Check your 3B model is pulled
ollama pull llama3.2
```

---

## Step 1: Free Up Space (Do This First)

### 1a. See what's using space
```bash
df -h
docker system df
du -sh ~/.ollama/models
```

### 1b. Remove unused Docker images (everything except OpenWebUI)
```bash
# Stop all containers first
docker stop $(docker ps -aq)

# Restart only OpenWebUI
docker start open-webui

# Remove all unused images, containers, networks
docker system prune -a -f
```

> ⚠️ Do NOT use `--volumes` if OpenWebUI stores data in a Docker volume.
> Check first: `docker inspect open-webui | grep -A5 Mounts`

### 1c. Clear pip and apt caches
```bash
pip cache purge
sudo apt-get clean
sudo apt-get autoremove -y
```

### 1d. Trim system logs
```bash
sudo journalctl --vacuum-size=100M
```

### 1e. After cleanup, confirm space
```bash
df -h
# You want at least 2GB free before continuing
```

---

## Step 2: Create the Base Venv (One Time Only)

This pre-warmed venv is cloned for each code execution run.
Cloning takes ~0.5 seconds vs 4-8 seconds for a fresh venv — critical on Jetson.

```bash
# Create the venv
python3 -m venv ~/.owui_base_venv

# Install common packages
# --no-cache-dir saves storage (important on Jetson)
~/.owui_base_venv/bin/pip install --no-cache-dir \
    numpy pandas requests matplotlib scipy pillow pydantic
```

This uses approximately **400-600MB**. Add or remove packages based on what
you expect the model to generate — you can always re-run this to add more.

### To add packages later:
```bash
~/.owui_base_venv/bin/pip install --no-cache-dir <package_name>
```

### To see what's installed:
```bash
~/.owui_base_venv/bin/pip list
```

---

## Step 3: Install the Tool in OpenWebUI

### 3a. Open OpenWebUI
Navigate to `http://localhost:3000` in your browser.

### 3b. Go to Tools
- Click your profile icon (top right)
- Go to **Settings → Tools**
- Click **"+"** or **"Create Tool"**

### 3c. Paste the tool code
- Copy the entire contents of `ralph_coder_tool.py`
- Paste into the tool editor
- Name it: `Ralph Coder`
- Description: `Self-retrying Python code generator with venv execution`

### 3d. Configure the model name
At the top of the file, find this line and change it to match your Ollama model:
```python
MODEL = "llama3.2"    # ← change this to whatever `ollama list` shows
```

### 3e. Save the tool
Click Save. OpenWebUI will validate the Python syntax automatically.

---

## Step 4: Enable the Tool in a Chat

1. Open a new chat
2. Click the **"Tools"** icon (wrench/plug icon near the input box)
3. Toggle **"Ralph Coder"** on
4. You should see it appear as active

---

## Step 5: Use It

Type your task naturally. Be specific — more detail = better plans = better code.

### Good prompts:
```
Build a Python script that reads a CSV file, removes duplicate rows,
calculates the mean of each numeric column, and saves a summary report to a new CSV.

Write a Python web scraper that fetches the title and first paragraph from
a list of URLs, handles timeouts and 404 errors gracefully, and saves results to JSON.

Create a Python class for a simple task queue that supports adding tasks,
processing them in order, retrying failed tasks up to 3 times, and logging results.
```

### What you'll see as it runs:
```
🧠 Planning subtasks...
📋 4 subtask(s) planned
🔨 [1/4] Write read_csv(filepath) function...
  ✏️  Writing — attempt 1/6...
  🚀 Running in venv...
  🔍 Judging output...
  ✅ Passed on attempt 1
🔨 [2/4] Write remove_duplicates(df) function...
  ✏️  Writing — attempt 1/6...
  🚀 Running in venv...
  ❌ Attempt 1 runtime error: ModuleNotFoundError: No module named 'pandas'
  ✏️  Writing — attempt 2/6...
  🚀 Running in venv...
  ✅ Passed on attempt 2
...
🔗 Assembling final file...
🚀 Final validation run...
✅ Done! All checks passed.
```

---

## Configuration Options

All config is at the top of `ralph_coder_tool.py`:

| Variable | Default | What it does |
|---|---|---|
| `MODEL` | `"llama3.2"` | Your Ollama model name |
| `MAX_RETRIES` | `6` | Retry attempts per subtask before giving up |
| `EXEC_TIMEOUT` | `15` | Seconds before killing a running script |
| `BASE_VENV` | `~/.owui_base_venv` | Location of the pre-warmed base venv |

### Tuning for your 3B model:

If the model frequently writes bad plans, increase the detail in your task prompt.
The planner is the most important step — a bad plan leads to bad chunks.

If it keeps hitting judge failures even when code looks correct, lower your
expectations by editing the judge prompt in `_judge()` to be more lenient.

If execution is too slow, reduce `MAX_RETRIES` to 4.

---

## Troubleshooting

### "ERROR calling model"
Ollama isn't running or the model name is wrong.
```bash
ollama list              # check model names
systemctl status ollama  # check if Ollama service is running
ollama serve             # start it manually if needed
```

### Tool doesn't appear in OpenWebUI
Check the OpenWebUI logs:
```bash
docker logs open-webui --tail 50
```
Syntax errors in the tool file will prevent it from loading.

### Venv clone fails
The base venv doesn't exist yet:
```bash
ls ~/.owui_base_venv    # should exist
# If not, run Step 2 again
```

### Code always times out
The script is running an infinite loop or doing heavy computation.
Increase `EXEC_TIMEOUT` or add print statements to debug.

### Running out of disk space
Check if temp dirs are being cleaned up:
```bash
ls /tmp/owui_run_*   # should be empty between runs
```
If dirs are accumulating, the `finally` block may not be running.
Force clean: `rm -rf /tmp/owui_run_*`

Also check pip cache isn't growing:
```bash
du -sh ~/.cache/pip
pip cache purge
```

---

## Storage Budget Summary

| Item | Size |
|---|---|
| Base venv (`~/.owui_base_venv`) | ~500MB |
| Temp venv per run (auto-deleted) | ~500MB peak, 0MB after |
| Tool file itself | <50KB |
| pip cache (with `--no-cache-dir`) | ~0MB |
| **Total permanent cost** | **~500MB** |

You need at least **1.5GB free** at all times for the OS to breathe and
temp venvs to exist during execution. Aim for 2GB+ free after setup.

---

## How It Works (Technical Summary)

```
User prompt
    ↓
[_plan()] — LLM breaks task into subtasks (JSON array)
    ↓
For each subtask:
    [_write_chunk()] — LLM writes one function/class
         ↓
    [_syntax_check()] — ast.parse(), instant, free
         ↓ (if passes)
    [_run_in_venv()] — clone venv, pip install detected deps,
                        execute, capture stdout/stderr, delete venv
         ↓ (if success)
    [_judge()] — LLM checks if output matches intent
         ↓ (if PASS)
    chunk added to built_chunks[]
    ↓
[_assemble()] — LLM stitches all chunks into one file
    ↓
[_syntax_check()] — final syntax validation
    ↓
[_run_in_venv()] — final execution to confirm it works
    ↓
Return code to user
```

The key insight borrowed from Ralph: **each retry gets the full error context**
from the previous attempt. The model sees exactly what went wrong — the actual
Python traceback — not just "it failed." This is what makes a 3B model able to
produce complex, working code reliably.
