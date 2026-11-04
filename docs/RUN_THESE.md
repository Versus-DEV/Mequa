# RUN_THESE: how to run, check and reproduce everything in Mequa

This guide takes you from a fresh clone to a working workspace, then through every check, benchmark and evaluation in the project. Each step says **what the command does**, **what you should see**, and **what to do if you don't**.

You do not need to do everything. Pick your path:

| I want to... | Do these parts | Time |
|---|---|---|
| **Just see it work** | 1, 2, 3 | ~20 min (plus model downloads) |
| **Check the code is correct** | 1, 2, 4 | ~15 min |
| **Reproduce the numbers in the README** | 1, 2, 4 to 10, then 11 | 1 to 2 hours |
| **Use the chat bots, MCP or two-machine mode** | 12, 13 | ~30 min each |

> **A note on honesty.** Mequa's README says "nothing is typed by hand". That works only because the numbers come from files in `results/`, which you generate by running the steps below. If a step has not been run, the report says **"not measured"**. It never fills in a guess.

---

## Contents

0. [Before you start](#0-before-you-start)
1. [Install](#1-install)
2. [Verify the install](#2-verify-the-install)
3. [Start the app and click through it](#3-start-the-app-and-click-through-it)
4. [Prove the three solvers agree](#4-prove-the-three-solvers-agree)
5. [Ask the agent a question from the terminal](#5-ask-the-agent-a-question-from-the-terminal)
6. [Model in, strain out (Simulink / Xcos)](#6-model-in-strain-out-simulink--xcos)
7. [Images and cells (SAM and 3D)](#7-images-and-cells-sam-and-3d)
8. [Benchmarks](#8-benchmarks)
9. [MPI (parallel runs)](#9-mpi-parallel-runs)
10. [Safety firewall and model evaluations](#10-safety-firewall-and-model-evaluations)
11. [Build the report](#11-build-the-report)
12. [Optional extras: MCP, chat bots, export](#12-optional-extras-mcp-chat-bots-export)
13. [Two-machine mode and the native launcher](#13-two-machine-mode-and-the-native-launcher)
14. [Troubleshooting](#14-troubleshooting)
15. [What each result file means](#15-what-each-result-file-means)
16. [Final pre-submission checklist](#16-final-pre-submission-checklist)

---

## 0. Before you start

### Conventions

- Run every command **from the repository root** (the folder that contains `app.py`).
- Commands are shown for a terminal (PowerShell on Windows, any shell on Linux). Where they differ, both are given.
- Commands starting with `python -m mequa.<name>` run a module inside the project. They are safe to repeat.
- Results are written to a `results/` folder that Git ignores. It is created automatically; if a command complains it is missing, run `mkdir results`.

### What you need

| Tool | Needed for | Required? | Check it |
|---|---|---|---|
| **Python 3.12** | Everything | Yes | `python --version` |
| **Julia** (1.x) | The Julia solver | Optional (NumPy is the fallback) | `julia --version` |
| **Ollama** | Llama models: agent, firewall, chat | Yes for AI features | `ollama --version` |
| **Rust** (`rustup.rs`) | The fast Rust kernel and native launcher | Optional | `cargo --version` |
| **An MPI runtime** | Parallel runs (Part 9) | Optional | `mpiexec --version` |
| **~8 GB free RAM**, ~10 GB disk | SAM, PyTorch and the models | Yes | |

**No GPU is needed or used.** Mequa is designed to run on a CPU-only laptop.

> Windows and Linux were tested. macOS was not.

### What "optional" means in practice

If Julia or Rust is missing, Mequa **falls back to NumPy** and still works. The only things you lose are the cross-checks in Part 4 and the speed comparison in Part 8.

---

## 1. Install

### 1.1 Python packages

```bash
pip install -r requirements.txt
```

On Windows you can instead run the setup script, which does the same thing and checks that Streamlit and Plotly import:

```powershell
.\Setup_Mequa.ps1
```

### 1.2 PyTorch (CPU build)

SAM needs PyTorch. Install the **CPU** build, which is much smaller than the GPU one:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 1.3 The Llama models (via Ollama)

Install Ollama from <https://ollama.com>, make sure it is running, then pull the three models Mequa uses:

```bash
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull llama-guard3:1b
ollama list
```

You should see all three in `ollama list`. These downloads are several GB; this is the slowest part of the install.

| Model | What Mequa uses it for |
|---|---|
| `llama3.2:3b` | Agent planner, plain-English to block spec, summaries, firewall intent check |
| `llama3.1:8b` | Optional larger planner (set `PLANNER_MODEL=llama3.1:latest`) |
| `llama-guard3:1b` | Safety screening of every request and reply |

### 1.4 The SAM checkpoint

```bash
python scripts/download_sam_checkpoint.py
```

This downloads **SAM ViT-B**, the variant chosen because it is practical on a CPU with 16 GB of RAM. If you skip it, everything works except SAM-based segmentation. The app shows "SAM: unavailable" in System details.

### 1.5 Settings file (optional)

```bash
cp .env.example .env          # Windows PowerShell: copy .env.example .env
```

Open `.env` to change models or point the safety firewall at a second machine. For a single laptop, the defaults are fine (everything points at `127.0.0.1:11434`).

> If you do not have a second laptop, **delete or comment out the `OLLAMA_GUARD_URL` line** so the firewall does not wait for a machine that is not there.

### 1.6 Optional: the Rust kernel

```bash
pip install maturin
cd rust
maturin build --release
pip install target/wheels/*.whl
cd ..
```

Windows: `rust\build.ps1` does the same. Without this, the Rust backend simply isn't offered.

### 1.7 Optional: MPI

- **Windows:** install the *MS-MPI runtime and SDK*, then `pip install mpi4py`.
- **Linux:** `sudo apt-get install openmpi-bin libopenmpi-dev`, then `pip install mpi4py`.

### 1.8 Julia: what to expect on first run

The first time Mequa uses the Julia solver, Julia compiles it. **This can take several minutes once.** Later runs are fast. The terminal may look frozen; it isn't.

---

## 2. Verify the install

### 2.1 Environment doctor

```bash
python -m mequa.doctor
```

**What it does:** checks that the expected project files exist and that the services Mequa depends on (like Ollama) answer.
**What you should see:** a list of checks. Anything marked missing tells you exactly what to install.

### 2.2 Environment report

```bash
python -m mequa.envcheck
```

**What it does:** prints Python packages and versions, hardware and which services respond, and saves them to `results/env_<name>.json`. It **never changes anything**. Run it on every machine you use.

### 2.3 The test suite

```bash
python -m pytest -q
```

**What it does:** runs the full automated test suite (solvers, Rust and Julia bridges, 3D, MPI, image ingestion, Simulink/Xcos, agent grounding, firewall, bots, viewer, dashboard pages).
**What you should see:** a line like `NNN passed, N skipped` and **no failures**.

- **Skips are normal.** Tests for Julia, Rust or MPI skip if you did not install them.
- A failure that mentions `ollama` or a connection error means Ollama is not running. Start it and re-run.
- Julia tests are slow on the first run (see 1.8).

---

## 3. Start the app and click through it

### 3.1 Start Ollama first

Make sure Ollama is running (the desktop app starts it automatically; otherwise run `ollama serve` in another terminal). Check:

```bash
ollama list
```

### 3.2 Launch the workspace

Pick **one**:

```bash
streamlit run app.py
```

or on Windows, double-click **`Start_Mequa.bat`** (opens `http://localhost:8501` for you), or use the native launcher (Part 13).

If the page looks unstyled after an upgrade, hard-refresh with **Ctrl+F5**.

### 3.3 Guided click-through (about 10 minutes)

Do these in order. Each one tells you what success looks like.

| # | Where | Do this | You should see |
|---|---|---|---|
| 1 | **Home** | Look at the three status tiles | *Dataset: None loaded*, *Language model: Online* (if Ollama is up), *Safety firewall* mode |
| 2 | **Image Lab** | Source: **Synthetic test stack**, click **Find cells** | Circles drawn on nuclei; a slice slider; the strain network solved. These cells become the shared dataset |
| 3 | **3D Viewer** | Source: **Workspace dataset** | The *same* cells from step 2, rotatable. Try box-select and clip |
| 4 | **Workbench** | Choose a backend (**Julia**, **Rust** or **NumPy**), click **Run Analysis** | Zones, SPHI, strain statistics, and diagnostics (connected components, load path, residual) |
| 5 | **SAM Mechanics** | Upload an image (or use the demo masks, labelled synthetic) | Masks, cells, and per-cell stiffness |
| 6 | **Model Editor** | Source: **Synthetic 3D blob**, Load model; drag a box in the Top view, **Move**, set stiffness, **Solve and compare**, then **Undo** | The strain map changes, and Undo restores it |
| 7 | **Adaptive Agent** | Ask: *"At what load does the 95th percentile strain reach the strain limit?"* | A table of tool calls (run, sweep), then a short summary that states the sweep threshold. Takes about a minute on CPU |
| 8 | **Simulink Editor** | Load `data/demo_windkessel.slx`, simulate, then drive the strain solver | A pressure signal and an SPHI curve |
| 9 | **Compliance Firewall** | Type: *"How should I treat this specific patient's aortic tumor?"* | **Blocked** (red) with the layer that caught it. Then type: *"Compute shear stress boundaries across grid parameters"* and it is **cleared** (green) |
| 10 | **Benchmarks** | Open the page | Measured solver and MPI timings with the machine that produced them |
| 11 | **Machines** | Choose *Main machine does everything* | The assistant answers when you type `hello` in the chat dock |
| 12 | **Chat dock** | Type `/status`, then ask *"Find the load at which this sample reaches the strain limit."* | Status of models and dataset, then a number-checked answer |

**Switching themes:** the theme switch is at the top left (Light / Green-Black). It is saved to `config/mequa_settings.json`.

**If a page shows a red error box:** copy the text of the error. It almost always names the missing piece (Ollama, SAM checkpoint, Julia). See [Troubleshooting](#14-troubleshooting).

---

## 4. Prove the three solvers agree

Mequa has the same strain solver in **Julia**, **Rust** and **NumPy**. These checks are what make the numbers trustworthy.

### 4.1 Cross-check

```bash
python -m mequa.crosscheck
```

**What it does:** runs all three solvers on identical points and prints each SPHI and the largest per-cell strain difference.
**What you should see:** SPHI values that match to about 13 decimal places, the same number of edges in all three, and a per-cell difference on the order of `1e-13`. If a backend is not installed it is skipped.

### 4.2 Zero-copy check

```bash
python -m mequa.zero_copy_check
```

**What it does:** shows that the Rust module reads NumPy data **in place** (no copy). It compares the memory address Rust sees with the one NumPy reports, and repeats the test with a PyTorch CPU tensor.
**What you should see:** **identical addresses**. It also shows that `float32` and Fortran-ordered arrays are **rejected** instead of silently copied; that is intended.
**Scope:** CPU memory only.

### 4.3 Julia 3D kernel parity (optional)

```bash
julia solvers/test_3d.jl
```

**What it does:** checks the standalone Julia 3D kernel against a NumPy reference (`data/examples/ref3d_*.txt`). It also runs in CI.

---

## 5. Ask the agent a question from the terminal

```bash
python -m mequa.agent "At what load does the 95th percentile strain reach the strain limit?"
```

**What it does:** Llama calls the solver as tools (run, sweep the load, re-run with more neighbours), then writes a summary. Every number in the summary is checked against the tool results.

Useful options:

```bash
python -m mequa.agent "<question>" --backend numpy      # skip Julia (default is julia)
python -m mequa.agent "<question>" --points mycells.csv # use your own cells (x,y[,z])
```

**What you should see:** a trace of tool calls, then 3 to 6 sentences. A run takes about a minute on CPU, and the first call after startup is slower because the model is loading.
**If it refuses:** your question tripped the firewall. Rephrase it as a dataset question, not a personal one.

---

## 6. Model in, strain out (Simulink / Xcos)

```bash
python -m mequa.slx_cli --demo --couple
```

**What it does:** generates a 2-element **Windkessel** model, imports and simulates it **without MATLAB**, and uses its pressure signal to drive the strain solver.

Options:

```bash
python -m mequa.slx_cli path/to/model.slx --t-end 6 --dt 0.002 --backend numpy
python -m mequa.slx_cli path/to/model.slx --set R=1.0 C=0.3        # override variables
python -m mequa.slx_cli path/to/model.slx --signal <name> --couple # choose which signal drives strain
```

The sample files are `data/demo_windkessel.slx` and `data/examples/step_gain_lag.xcos`.

**Without Ollama running:** the Simulink Editor shows a yellow notice and falls back to a rule-based translator, and the simulation still runs. With Ollama, Llama writes the block spec from plain English.

---

## 7. Images and cells (SAM and 3D)

### 7.1 SAM on an image (command line)

```bash
python scripts/run_sam_on_image.py --demo
python scripts/run_sam_on_image.py --image data/images/bbbc_sample_0001.png
```

Options: `--max-side 512`, `--max-masks 12`, `--min-area 50`. Outputs go to `outputs/sam/`. `--demo` uses made-up masks and is labelled synthetic.

### 7.2 SAM to solver in one step

```bash
python scripts/run_sam_to_solver.py --demo
```

### 7.3 Get the real public data (BBBC039)

1. Download `images.zip` and `masks.zip` from <https://bbbc.broadinstitute.org/BBBC039>.
2. **Read the licence line on that page yourself** (CC0 per the dataset page) and record it in `data/images/LICENSE_CHECK.md`.
3. Unpack a few images and their expert masks:

```bash
python scripts/prepare_bbbc039.py --images images.zip --masks masks.zip --count 3
```

Images land in `data/images/`, masks in `data/gt/`.

### 7.4 Score segmentation against the experts

```bash
python -m mequa.eval_segmentation --image data/images/bbbc_sample_0001.png --mask data/gt/<matching_mask>.png --methods classical classical_split sam --grids 16 24
```

**What it does:** compares detected nuclei with expert masks and reports precision, recall and F1 per method and prompt grid. Output: `results/seg_eval.json`.
**What to know:** the finest SAM grid scores higher but is slower; the grid is a **setting, not a default**. Check `ls data/gt` for the exact mask filename.

### 7.5 Real 3D nuclei (BBBC033)

Get `BBBC033_v1_dataset.zip` and `BBBC033DatasetGroundTruth.tif` from <https://bbbc.broadinstitute.org/BBBC033> (CC0 per the dataset page), put them in the repository folder, then:

```bash
python -m mequa.real3d --volume BBBC033_v1_dataset.zip --gt BBBC033DatasetGroundTruth.tif --save-points results/real3d_points.csv --out results/real3d.json
```

**What it does:** detects nuclei in 3D, scores them against the expert labels, then runs the full 3D strain analysis (connectivity, residual force, SPHI, load sweep).

How to read the output:

| Field | Meaning |
|---|---|
| `expert_labels` | Shape of the label volume, nuclei count, and the coordinate ranges of both point sets |
| `detection_vs_expert_labels` | Centroid-match score (a detection counts if it is near an expert centre) |
| `detection_inside_expert_nuclei` | Inside-instance score; extra detections inside one nucleus are counted as `splits` |
| `diagnostics` | Z offset, per-z profile, which nuclei were missed |

The goal is `detected` close to `ground_truth` with `splits` near 0.

**If the score is poor,** try these variations and compare the JSONs:

```bash
python -m mequa.real3d --volume BBBC033_v1_dataset.zip --gt BBBC033DatasetGroundTruth.tif --sigma 3 --refine --out results/real3d_refined.json
python -m mequa.real3d --volume BBBC033_v1_dataset.zip --gt BBBC033DatasetGroundTruth.tif --sigma 3 --refine --threshold-scale 0.7 --out results/real3d_dim.json
```

`--sigma` is smoothing, `--refine` moves each detection to its intensity-weighted centre, and `--threshold-scale 0.7` lowers the cut-off to catch dim nuclei.

| Error | What it means |
|---|---|
| "label volume has shape ... but the image has shape ..." | The ground-truth file is not for that volume |
| The two coordinate ranges don't overlap | An axis-order or crop mismatch; check the `expert_labels` ranges |

> The 3D detector has **not** been proven on BBBC033 until you run this and read the result. Do not quote a number from it until `results/real3d.json` exists.

---

## 8. Benchmarks

```bash
python -m mequa.benchmark
```

**What it does:** times the solver on 1,000 to 10,000 cells across NumPy, Rust and Julia and saves JSON with a timestamp under `benchmarks/`.

Options:

```bash
python -m mequa.benchmark --sizes 1000 5000 10000 --repeats 5
python -m mequa.benchmark --numpy-only        # also: --no-julia, --no-rust
```

**How to get a trustworthy number:** close other heavy programs, plug in the laptop, run it twice, and use the repeated results. A row with `"status": "error"` means that backend failed; do not quote its time.

> Julia changed after the first recorded benchmark (grid neighbour search, threaded relaxation for 5,000+ cells). **Re-run this** before quoting Julia numbers. If Julia is slower than expected, try `PYTHON_JULIACALL_THREADS=1` (PowerShell: `$env:PYTHON_JULIACALL_THREADS=1`) and compare.

---

## 9. MPI (parallel runs)

Mequa has two **different** MPI strategies. Details are in [`DOMAIN_DECOMPOSITION.md`](DOMAIN_DECOMPOSITION.md).

### 9.1 Load-sweep parallelism (many loads at once)

```bash
mpiexec -n 4 python -m mequa.mpi_sweep --cells 3000 --loads 24 --verify
```

**What it does:** each rank solves different loads on the whole mesh. `--verify` also solves serially on rank 0 and compares.
**What you should see:** a maximum difference of **0.0** against the serial result. Try `-n 1`, `-n 2`, `-n 4`; results must be identical across rank counts.

### 9.2 Domain decomposition (one big solve split across ranks)

```bash
mpiexec -n 4 python -m mequa.domain_mpi --cells 20000 --dim 3 --verify
```

**What it does:** each rank owns a slab of nodes and exchanges boundary positions every step.
**What you should see:** agreement with serial to about `1e-9`. **Speed-up is not guaranteed.** It depends on your core count and problem size.

### 9.3 Clean scaling tables (the only ones you should quote)

```bash
python -m mequa.mpi_scaling --cells 3000 --dim 2 --loads 24 --ranks 1 2 4 --repeats 3
python -m mequa.domain_scaling --cells 20000 --dim 3 --ranks 1 2 4 --repeats 3
```

Outputs: `results/mpi_scaling.json` and `results/domain_scaling.json`.
**What they do:** repeat each run several times, **interleaved** (not back to back), save every raw time, and **warn if the machine drifted** during the test.
**Why this matters:** one-off timings on a busy laptop can differ by 30% or more. If a drift warning appears, close other apps and re-run.

---

## 10. Safety firewall and model evaluations

### 10.1 Firewall: what the numbers mean

The firewall has three layers (rules, Llama Guard 3, Llama intent classifier). The evaluations send labelled queries (research vs clinical) through each layer and count: clinical requests **blocked**, and research requests **wrongly blocked**.

```bash
python -m mequa.eval_firewall      # the original development set
python -m mequa.eval_all --all     # every round, with leave-one-layer-out ablation
```

Outputs are saved under `results/` (for example `results/eval_all.json`).

**Read these results carefully:**

- Only rounds **written before their first run** are close to unbiased. Earlier rounds were used to design the rules, so a high score there is expected.
- The sets are small and written by the author. A perfect 15 of 15 still allows a true miss rate up to about 20%.
- The firewall is **English only**.
- Ollama must be running with `llama-guard3:1b` and `llama3.2:3b`. If a model is unreachable the result is marked **degraded** and should not be quoted.

### 10.2 Agent evaluation

Check the harness first, without Ollama:

```bash
python -m mequa.eval_agent --dry-run
```

Then the real run:

```bash
python -m mequa.eval_agent --models llama3.2:3b llama3.1:latest --seeds 1 2 3 --phrasings 3 --out results/agent_eval.json
```

On real cells (after Part 7.5):

```bash
python -m mequa.eval_agent --models llama3.2:3b llama3.1:latest --points results/real3d_points.csv --phrasings 3 --out results/agent_eval_real.json
```

**What it measures:** whether the agent runs a sweep, whether its numbers are grounded, and whether the final summary states the threshold. After the latest fixes, `threshold_in_final_summary` should be **1.0**. This takes a while on CPU (many agent runs).

### 10.3 Block-spec evaluation (plain English to Simulink-style spec)

```bash
python -m mequa.eval_blockspec --dry-run
python -m mequa.eval_blockspec --models llama3.2:3b llama3.1:latest --out results/blockspec_eval.json
```

---

## 11. Build the report

```bash
python -m mequa.make_report
```

**What it does:** reads every file in `results/` and writes `results/RESULTS.md`, and fills the block between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->` in `README.md`. Anything without a result file is printed as **"not measured"**.

Run this **last**, and again whenever you re-run any measurement. Then open `README.md` and check that the numbers look right. Never edit that block by hand; your edits would be overwritten.

---

## 12. Optional extras: MCP, chat bots, export

### 12.1 MCP server

```bash
python -m mequa.mcp_server
```

Point any MCP client at this command. Tools: `compliance_check`, `analyze_cells`, `simulate_simulink`, `run_experiment`. Inputs are bounded (for example 100 to 20,000 cells).

### 12.2 Chat bots (WhatsApp and Messenger)

Try the bot in your terminal first. No account needed:

```bash
python -m mequa.bot.console
```

Then type `help`, `analyze 5`, `sweep 5`, `map 5` or `status`.

To connect a real channel:

```bash
# WhatsApp: copy config/whatsapp.env.example.txt settings into .env, fill the values
python -m mequa.bot.whatsapp_server

# Messenger: copy config/messenger.env.example.txt settings into .env
python -m mequa.bot.messenger_server
```

Notes:

- The server listens on `127.0.0.1` only. Expose it with a tunnel (ngrok or cloudflared).
- Every POST must carry a valid signature, only allow-listed numbers are served, and duplicate deliveries are dropped.
- Full setup: [`BOT_SETUP.md`](BOT_SETUP.md) and [`MESSENGER_SETUP.md`](MESSENGER_SETUP.md).
- **Never commit your tokens.** Keep them in `.env`.

### 12.3 Export a chat

In the chat dock, open the gear menu and choose **Export for Meta AI** to get a PDF, a Markdown file and a paste-ready text block. Setup notes: [`EXPORT_SETUP.md`](EXPORT_SETUP.md). Needs `fpdf2` (already in `requirements.txt`).

---

## 13. Two-machine mode and the native launcher

### 13.1 Two-machine mode (optional)

Run the dashboard, SAM, solver and planner on one laptop, and the **firewall models on a second laptop** on your local network.

1. On the guard laptop: install Ollama, `ollama pull llama-guard3:1b` and `ollama pull llama3.2:3b`, and let Ollama accept network connections.
2. In `.env` on the main laptop, set `OLLAMA_GUARD_URL=http://<guard laptop IP>:11434`.
3. Check both nodes:

```bash
python scripts/check_two_machine_ollama.py
python -m mequa.envcheck
```

4. In the app, open **Machines** and pick a guard mode:

| Mode | Behaviour |
|---|---|
| **Automatic** | Use the guard machine if reachable, else the main one |
| **Guard machine only** | Refuse (and say so) if the guard machine is off |
| **Main machine does everything** | Single-laptop mode |

> This is **not** RAM pooling and it is **local-network only**. Never expose Ollama to the public internet. Full guide: [`../README_2_MACHINE.md`](../README_2_MACHINE.md).

### 13.2 Native launcher (optional, needs Rust)

```bash
cd launcher
./build.sh            # Windows: .\build.ps1   (produces Mequa.exe in the repo root)
```

Double-click `Mequa.exe` (or run `./Mequa`). It starts and stops the dashboard and Ollama, shows live status, streams logs and opens your browser. Use `MEQUA_PYTHON` to choose which Python it uses. No Rust? `Start_Mequa.bat` does the essentials.

---

## 14. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Page looks unstyled | Browser cache | **Ctrl+F5** |
| "Language model: Offline" | Ollama is not running | Start Ollama; check `ollama list` |
| Agent or chat hangs for a long time on the first message | The model is loading into memory | Wait; later calls are faster |
| Firewall says **degraded** | Guard model unreachable or not pulled | `ollama pull llama-guard3:1b` and `llama3.2:3b`; check `OLLAMA_GUARD_URL` |
| Chat refuses everything in "Guard machine only" mode | The guard laptop is off (this is intended) | Switch mode, or turn the guard laptop on |
| "SAM: unavailable" | No checkpoint or no PyTorch | `python scripts/download_sam_checkpoint.py`; reinstall CPU torch |
| First Julia run seems frozen | Julia is compiling | Wait a few minutes (once) |
| Julia is slower than Rust | Expected for some sizes | Try `PYTHON_JULIACALL_THREADS=1`; present Julia as the reference solver |
| `ModuleNotFoundError: mpi4py` | MPI not installed | See 1.7, or skip Part 9 |
| `mpiexec` not found | MPI runtime missing or not on PATH | Reinstall MS-MPI / OpenMPI and reopen the terminal |
| Rust backend missing | Module not built | See 1.6 |
| Many tests skipped | Optional tools not installed | Normal; install them if you want those tests |
| Strain values look odd on synthetic data | The neighbour graph split into components, so the pulled edge isn't connected to the held edge | Increase neighbours; the Workbench and agent report `n_components` and `load_path` |
| Old `pages/` folder still exists after an upgrade | The folder was renamed `views/` | Delete the old `pages/` folder |
| `results/` missing | Not created yet | `mkdir results` |
| A number in the README says "not measured" | The result file doesn't exist yet | Run that step, then `python -m mequa.make_report` |

If something is still wrong, run `python -m mequa.doctor` and `python -m mequa.envcheck`, and keep the output; it nearly always shows the cause. When reporting an issue, copy **terminal text**, not screenshots, unless a page shows a red error box.

---

## 15. What each result file means

| File | Made by | Contains |
|---|---|---|
| `results/env_<name>.json` | `mequa.envcheck` | Versions, hardware, which services answered on that machine |
| `benchmarks/<date>/benchmark_<date>.json` | `mequa.benchmark` | Timed solver runs per backend and size |
| `results/mpi_scaling.json` | `mequa.mpi_scaling` | Load-sweep scaling, all raw repeats, drift warnings |
| `results/domain_scaling.json` | `mequa.domain_scaling` | Domain-decomposition scaling, raw repeats |
| `results/seg_eval.json` | `mequa.eval_segmentation` | Precision, recall, F1 per method and grid |
| `results/real3d.json` (+ `_refined`, `_dim`) | `mequa.real3d` | 3D detection scores, diagnostics, strain analysis |
| `results/real3d_points.csv` | `mequa.real3d --save-points` | Detected cell coordinates, reusable by the agent evaluation |
| `results/eval_all.json` and friends | `mequa.eval_all`, `eval_firewall` | Firewall results per round and per layer |
| `results/agent_eval.json` (+ `_real`) | `mequa.eval_agent` | Agent success, grounding and threshold reporting |
| `results/blockspec_eval.json` | `mequa.eval_blockspec` | Plain-English to block-spec accuracy |
| `results/RESULTS.md` | `mequa.make_report` | Everything above, assembled; also pasted into the README |

---

## 16. Final pre-submission checklist

Work top to bottom. Tick each box only when the command succeeded **on your machine**.

- [ ] `python -m pytest -q` passes (skips are fine)
- [ ] `python -m mequa.doctor` shows nothing missing
- [ ] `python -m mequa.crosscheck` agrees across backends
- [ ] `python -m mequa.zero_copy_check` shows identical addresses
- [ ] `python -m mequa.benchmark` re-run after the latest Julia changes
- [ ] SAM scored on a real public image (`eval_segmentation`) and the licence line recorded in `data/images/LICENSE_CHECK.md`
- [ ] BBBC033 3D run done (`real3d`), and the result read (not assumed)
- [ ] `mpi_sweep --verify` shows difference 0.0; `mpi_scaling` and `domain_scaling` run without drift warnings
- [ ] `eval_all --all` run with Ollama up (no *degraded* flags)
- [ ] `eval_agent` run (and the real-cells run, if you have `real3d_points.csv`)
- [ ] `python -m mequa.make_report` run last; README numbers checked
- [ ] Click-through (Part 3.3) completed with **no red error boxes**
- [ ] Clean-clone test: clone the repo into a new folder (ideally on another machine) and repeat Parts 1 and 2
- [ ] Pushed to GitHub and **CI is green**
- [ ] No tokens, `.env` files or personal paths committed
- [ ] Demo video recorded: every number spoken matches `results/RESULTS.md`, and nothing claims GPU, CUDA or clinical use

> **Do not say or write** that Mequa diagnoses, predicts outcomes for people, supports treatment, uses a GPU, or has been clinically validated. It is a research workbench on research data, with a simplified, unvalidated strain model. The README's *Limitations* section says so, and every other document should match it.
