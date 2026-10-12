# Step-by-step guide: build, train and finish your project

Project: **Speaker-Aware Noise Cancellation and Target Speech Enhancement**
Plan: write and test code on your laptop (CPU), train on Google Colab (GPU), evaluate on your laptop (CPU).

> **Honest status of the code.** I wrote every file in this folder, but I could not install PyTorch in my workspace, so **none of it has been executed yet**. Step 3 (`sanity_check.py`) is there to find any problem within one minute. If something fails, copy the full error message and send it to me; I will fix it.

---

## 0. What the code does (one paragraph)

`prepare_data.py` lists your audio files in manifests. `precompute_embeddings.py` runs the pretrained ECAPA-TDNN once per utterance and stores a 192-number voice fingerprint. During training, `tse/data.py` builds a fresh mixture every time: target speech + (optionally) another speaker + noise, at random SNR and target-to-interferer ratio, and gives the model the fingerprint of a *different* utterance by the target speaker. `tse/models.py` is a **causal Conv-TasNet** whose internal features are modified by that fingerprint (`none`, `concat`, `film` or `gating`). `train.py` trains it with the SI-SDR loss, a curriculum from easy to hard mixtures, and saves a checkpoint every epoch so Colab disconnects cost you almost nothing. `evaluate.py` measures SI-SDR, SI-SDRi, STOI, PESQ and CPU speed. `infer.py` is the demo.

Files:

```
tse-project/
  tse/            models.py  data.py  mixing.py  losses.py  speaker.py
  prepare_data.py            # make manifests (speech + noise)
  precompute_embeddings.py   # ECAPA fingerprints
  sanity_check.py            # run first
  train.py  evaluate.py  infer.py
  requirements.txt          # Colab (no torch, Colab already has it)
  requirements-local.txt    # laptop, CPU only
  .gitignore  GUIDE.md
```

---

## 1. Set up your laptop

Use **Python 3.10 or 3.11**. Python 3.13 is too new and some audio packages have no wheels for it yet.

### Windows (PowerShell)

```powershell
cd path\to\tse-project
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-local.txt
```

If PowerShell blocks the activate script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

### Linux / macOS

```bash
cd path/to/tse-project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-local.txt
```

There are **two requirements files**, because your laptop has no CUDA GPU and you train on Colab:

| File | Where | Contains |
|---|---|---|
| `requirements-local.txt` | your laptop (CPU) | PyTorch CPU build and everything for the sanity check, smoke test, evaluation and demo |
| `requirements.txt` | Google Colab | only the extra packages (speechbrain, soundfile, pystoi, pesq, ...). It deliberately leaves out torch, torchaudio, numpy and scipy, because Colab already has a GPU build of them and reinstalling could break it |

Plain `pip install torch` gives a CPU build on Windows and macOS, and a CUDA build on Linux (large download). On Linux you can use the smaller CPU wheel first: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`, then run the install command above.

### Known install problems

| Problem | Fix |
|---|---|
| `AttributeError: torchaudio ... list_audio_backends` (SpeechBrain with a very new torchaudio) | `pip install -U speechbrain`. If it persists, `pip install torch==2.5.1 torchaudio==2.5.1` |
| SpeechBrain fails downloading the model on Windows (symlink / privilege error) | Windows Settings → Privacy & security → For developers → turn on **Developer Mode**, then rerun. Or just do the ECAPA step on Colab (Step 6) |
| `pesq` fails to build | It is optional. Skip it on the laptop. `evaluate.py` runs without it. It usually installs fine on Colab: `pip install pesq` |
| `soundfile` cannot read FLAC | `pip install -U soundfile` (needs libsndfile; the wheel includes it on Windows/macOS). On Ubuntu: `sudo apt install libsndfile1` |

---

## 2. Put the project under git (so Colab can download it)

```bash
git init
git add .
git commit -m "Initial TSE project"
```

Create a **private** repository on GitHub, then:

```bash
git remote add origin https://github.com/<you>/tse-project.git
git branch -M main
git push -u origin main
```

For a private repo, Colab needs a token: GitHub → Settings → Developer settings → Personal access tokens → Fine-grained → give it read access to this repo only. Copy it once and keep it safe. Never paste a token into the code you commit.

---

## 3. Run the sanity check (1 minute, no downloads)

```bash
python sanity_check.py
```

Expected output (numbers will differ):

```
[ok] mixing: SNR/TIR match requested values
[ok] SI-SDR: identical/scaled signals ~ high, unrelated ~ low
[ok] model cond=none     shape, causality (diff=...), train step; params=...
[ok] model cond=concat   ...
[ok] model cond=film     ...
[ok] model cond=gating   ...
[ok] default model: ... params, RTF on this CPU = 0.xxx
All sanity checks passed.
```

What it proves: the mixing math is right, the model keeps the input length, **changing future audio does not change past output** (causality), and one training step works. The printed **RTF** is how long the model takes per second of audio on your CPU. It must be below 1.0 for real-time use. If it is above 1.0, use a smaller model (`--N 128 --H 128`) later.

Fix every failure here before moving on.

---

## 4. Smoke test the full pipeline on your laptop (20-30 minutes)

Goal: prove data → embeddings → training → evaluation → inference all run, before spending Colab time. Quality does not matter here.

### 4.1 Download one small dataset

Download **dev-clean** (337 MB) from https://www.openslr.org/12 and extract it into `data/` so you get `data/LibriSpeech/dev-clean/<speaker>/<chapter>/*.flac`.

Windows: use 7-Zip to extract the `.tar.gz` twice (gz, then tar). Linux/macOS: `tar -xzf dev-clean.tar.gz -C data`.

### 4.2 Manifests

```bash
python prepare_data.py speech --root data --subdir LibriSpeech/dev-clean --out data/tiny_speech.jsonl --max-utts 8
```

Expect "40 speakers" and "wrote ~320 rows".

### 4.3 Embeddings (downloads the ECAPA model, about 80 MB)

```bash
python precompute_embeddings.py --root data --manifests data/tiny_speech.jsonl --out data/tiny_emb.pt
```

### 4.4 Tiny training run (uses synthetic noise because no noise manifest is given)

Put the command on **one line** in PowerShell.

```bash
python train.py --data-root data --train-speech data/tiny_speech.jsonl --train-emb data/tiny_emb.pt --val-speech data/tiny_speech.jsonl --val-emb data/tiny_emb.pt --out-dir runs/smoke --cond film --N 64 --B 32 --H 64 --X 4 --R 2 --batch 4 --epochs 3 --steps-per-epoch 30 --val-size 20 --workers 0
```

Expected: three lines like `epoch 0 stage 0 train_loss ... val SI-SDRi ...`. The train loss (negative SI-SDR) should go **down** over epochs. Do not worry about the absolute values yet.

Then test resume: run the **same command again** with `--epochs 5`. It must print `RESUMED ... continuing at epoch 3`.

### 4.5 Evaluate and run the demo

```bash
python evaluate.py --ckpt runs/smoke/best.pt --data-root data --speech data/tiny_speech.jsonl --emb data/tiny_emb.pt --n 20 --out results/smoke.csv
```

For `infer.py`, record two files with your phone or laptop (WAV or similar): `enroll.wav` (you speaking alone for 4 s) and `noisy.wav` (you plus someone else plus noise). The smoke model is untrained so the output will be poor; this only tests that the script runs:

```bash
python infer.py --ckpt runs/smoke/best.pt --mixture noisy.wav --enroll enroll.wav --out out.wav
```

If all of 4.1-4.5 work, the code path is sound. Push any fixes to GitHub.

---

## 5. Choose the datasets for the real run

You need three things: speech, noise, and speaker-disjoint train/val/test sets.

| Role | Dataset | Size | Where |
|---|---|---|---|
| Train speech | LibriSpeech **train-clean-100** (251 speakers) | 6.3 GB | https://www.openslr.org/12 |
| Validation speech | LibriSpeech **dev-clean** (40 speakers) | 337 MB | same |
| Test speech (unseen speakers) | LibriSpeech **test-clean** (40 speakers) | 346 MB | same |
| Noise | **DEMAND** 16 kHz versions, 17 categories | about 2.6 GB | https://zenodo.org/records/1227121 |

The three LibriSpeech sets have different speakers by design, so "unseen speaker" tests are honest.

For DEMAND we **hold out 4 categories for testing** (`PCAFETER, STRAFFIC, TMETRO, DWASHING`). The model never hears them in training, which gives the "unseen noise" test from your proposal. Your proposal also mentions MUSAN; DEMAND alone is enough for a capstone, and you can add MUSAN noise later if time allows.

---

## 6. Train on Google Colab

### 6.1 One-time preparation

1. Make sure the code is pushed to GitHub (Step 2).
2. Open https://colab.research.google.com → New notebook.
3. **Runtime → Change runtime type → T4 GPU**.
4. Each cell below is a separate Colab cell. `%%bash` cells run shell code.

### 6.2 Cell 1: check the GPU

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU - change the runtime type!")
```

### 6.3 Cell 2: mount Google Drive (checkpoints live here, they survive disconnects)

```python
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/tse'
for d in ['runs', 'archive', 'results']:
    os.makedirs(f'{DRIVE}/{d}', exist_ok=True)
```

### 6.4 Cell 3: get the code and install packages

```python
from getpass import getpass
token = getpass('GitHub token (input hidden): ')
!git clone https://{token}@github.com/<you>/tse-project.git /content/tse-project
%cd /content/tse-project
!pip install -q -r requirements.txt
```

`requirements.txt` is the Colab file: it leaves out torch, torchaudio, numpy and scipy because Colab already has a GPU (CUDA) build of them. Do not reinstall them. If `pesq` fails to build, delete that line from `requirements.txt` and rerun; evaluation still works without it.

### 6.5 Cell 4: data. Use Colab's local disk, never train from Drive (Drive is very slow for many small files)

The first time, download and prepare. Afterwards, save the prepared data to Drive as a single archive so later sessions skip the downloads.

```bash
%%bash
mkdir -p /content/data && cd /content/data
if [ -f /content/drive/MyDrive/tse/archive/data.tar ]; then
  echo "Restoring from Drive archive..."
  tar -xf /content/drive/MyDrive/tse/archive/data.tar -C /content/data
else
  for s in train-clean-100 dev-clean test-clean; do
    wget -q https://www.openslr.org/resources/12/$s.tar.gz && tar -xzf $s.tar.gz && rm $s.tar.gz
  done
  mkdir -p DEMAND && cd DEMAND
  for n in DKITCHEN DLIVING DWASHING NFIELD NPARK NRIVER OHALLWAY OMEETING OOFFICE PCAFETER PRESTO PSTATION SPSQUARE STRAFFIC TBUS TCAR TMETRO; do
    wget -q https://zenodo.org/records/1227121/files/${n}_16k.zip && unzip -q -o ${n}_16k.zip && rm ${n}_16k.zip
  done
fi
ls /content/data
```

If a download fails with 404, open the dataset page in your browser, copy the exact file link, and use it. If openslr.org is slow, try the mirror https://openslr.elda.org/12/.

### 6.6 Cell 5: manifests and embeddings (skip if restored from the archive; they are inside it)

```bash
%%bash
cd /content/tse-project
python prepare_data.py speech --root /content/data --subdir LibriSpeech/train-clean-100 --out /content/data/train_speech.jsonl --max-utts 60
python prepare_data.py speech --root /content/data --subdir LibriSpeech/dev-clean --out /content/data/val_speech.jsonl --max-utts 20
python prepare_data.py speech --root /content/data --subdir LibriSpeech/test-clean --out /content/data/test_speech.jsonl --max-utts 20
# "seen speakers" test: other utterances of training speakers
python prepare_data.py speech --root /content/data --subdir LibriSpeech/train-clean-100 --out /content/data/seen_speech.jsonl --max-utts 10 --exclude /content/data/train_speech.jsonl
python prepare_data.py noise --root /content/data --subdir DEMAND --holdout PCAFETER,STRAFFIC,TMETRO,DWASHING --out-train /content/data/noise_train.jsonl --out-test /content/data/noise_test.jsonl
python precompute_embeddings.py --root /content/data --manifests /content/data/train_speech.jsonl --out /content/data/train_emb.pt
python precompute_embeddings.py --root /content/data --manifests /content/data/val_speech.jsonl --out /content/data/val_emb.pt
python precompute_embeddings.py --root /content/data --manifests /content/data/test_speech.jsonl --out /content/data/test_emb.pt
python precompute_embeddings.py --root /content/data --manifests /content/data/seen_speech.jsonl --out /content/data/seen_emb.pt
```

Then save everything (not the raw LibriSpeech, just manifests, embeddings and noise) so you never repeat this:

```bash
%%bash
cd /content/data
tar -cf /content/drive/MyDrive/tse/archive/data.tar LibriSpeech DEMAND noise16k *.jsonl *.pt 2>/dev/null || tar -cf /content/drive/MyDrive/tse/archive/data.tar LibriSpeech DEMAND *.jsonl *.pt
ls -lh /content/drive/MyDrive/tse/archive/
```

The archive is about 8 GB. Check your Drive has space (free Drive is 15 GB). If it is too tight, delete `LibriSpeech/train-clean-100` files you did not select, or tar only `*.jsonl *.pt DEMAND` and re-download LibriSpeech each time (about 10 minutes).

### 6.7 Cell 6: train (this exact cell is also your "resume" cell)

```bash
%%bash
cd /content/tse-project
python train.py --data-root /content/data \
  --train-speech /content/data/train_speech.jsonl --train-emb /content/data/train_emb.pt --train-noise /content/data/noise_train.jsonl \
  --val-speech /content/data/val_speech.jsonl --val-emb /content/data/val_emb.pt --val-noise /content/data/noise_train.jsonl \
  --out-dir /content/drive/MyDrive/tse/runs/film --cond film \
  --batch 6 --segment 3.0 --epochs 60 --steps-per-epoch 500 --stage-epochs 8 --amp --workers 2 --max-hours 3.5
```

Key facts:

- Checkpoints (`last.pt`, `best.pt`) and `log.csv` are written to **Drive** after every epoch.
- If Colab disconnects, open a new session, run Cells 1-4 again, then this cell again. It prints `RESUMED ... continuing at epoch N`.
- `--max-hours 3.5` stops cleanly before Colab's session limit. Free Colab sessions are not guaranteed; they can end earlier.
- **First, time one epoch.** The log prints minutes per epoch. Multiply by your epoch count to plan. I could not measure this, so do not trust any number I would guess.
- **Out of memory?** Lower `--batch` to 4, then 2, and add `--accum 2` (or 3) to keep the effective batch size. You can also use `--segment 2.0`.
- **Speed vs GPU.** If the GPU sits idle, raise `--workers` to 4. Data loading from `/content/data` is the usual bottleneck.
- The model here is a **compact** Conv-TasNet (`N=256, B=128, H=256, X=8, R=3`). For the proposal's full size add `--N 512 --H 512`; it needs a smaller batch (try `--batch 3`).

### 6.8 Cell 7: watch progress

```python
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/tse/runs/film/log.csv')
print(df.tail(10))
df.plot(x='epoch', y=['train_loss', 'val_sisdri'], subplots=True)
```

What good progress looks like: `val_sisdri` (improvement over the noisy input, in dB) climbs and eventually flattens. It will jump down a little when the curriculum moves to the next, harder stage (every `--stage-epochs`); that is normal. Stop when it has been flat for around 10 epochs.

### 6.9 Keep the session alive and tidy

- Keep the tab open. Colab disconnects idle sessions, but a running cell counts as activity.
- When you stop, always run training once more until it says it stopped cleanly, so `last.pt` is current.
- Your files on Drive: `tse/runs/film/best.pt` is the model you will use.

---

## 7. The experiments to run (your "results" section)

Run the same training command with a different `--out-dir` and `--cond`. Keep **everything else identical** so the comparison is fair.

| Run | `--cond` | `--out-dir` | Purpose |
|---|---|---|---|
| A | `none` | `runs/none` | Baseline: no speaker info. Shows the speaker fingerprint matters |
| B | `concat` | `runs/concat` | Simplest conditioning (from your proposal) |
| C | `film` | `runs/film` | Main model |
| D (optional) | `gating` | `runs/gating` | Third mechanism from your proposal |

Priority if time is short: **C first, then B, then A, then D.**

`none` is not in the proposal but is the most convincing comparison: without the fingerprint the model cannot know which speaker to keep.

Extra optional experiment: a bigger model (`--N 512 --H 512`) for the FiLM condition.

---

## 8. Evaluate (the four generalization tests from your proposal)

On Colab (or locally after downloading `best.pt` and the manifests/embeddings):

```bash
%%bash
cd /content/tse-project
R=/content/drive/MyDrive/tse/runs/film/best.pt
# seen speakers, seen noise
python evaluate.py --ckpt $R --data-root /content/data --speech /content/data/seen_speech.jsonl --emb /content/data/seen_emb.pt --noise /content/data/noise_train.jsonl --n 300 --out /content/drive/MyDrive/tse/results/film_seen_seen.csv
# unseen speakers, seen noise
python evaluate.py --ckpt $R --data-root /content/data --speech /content/data/test_speech.jsonl --emb /content/data/test_emb.pt --noise /content/data/noise_train.jsonl --n 300 --out /content/drive/MyDrive/tse/results/film_unseen_seen.csv
# seen speakers, unseen noise
python evaluate.py --ckpt $R --data-root /content/data --speech /content/data/seen_speech.jsonl --emb /content/data/seen_emb.pt --noise /content/data/noise_test.jsonl --n 300 --out /content/drive/MyDrive/tse/results/film_seen_unseen.csv
# unseen speakers, unseen noise (the hardest, main result)
python evaluate.py --ckpt $R --data-root /content/data --speech /content/data/test_speech.jsonl --emb /content/data/test_emb.pt --noise /content/data/noise_test.jsonl --n 300 --out /content/drive/MyDrive/tse/results/film_unseen_unseen.csv
```

Repeat for each run (`none`, `concat`, `gating`) by changing `R` and the output name. Each command prints a summary table; the CSVs have per-mixture detail for plots.

Also run the hard-condition sweep for one plot ("SI-SDRi versus input SNR"):

```bash
python evaluate.py ... --snr-range -5 0 --out ...snr_m5_0.csv
python evaluate.py ... --snr-range 0 10 --out ...snr_0_10.csv
python evaluate.py ... --snr-range 10 20 --out ...snr_10_20.csv
```

### CPU speed (on your laptop, the deployment target)

1. Download `best.pt` from Drive.
2. Run `evaluate.py` locally with `--threads 4`. The summary prints **RTF** on your CPU. Below 1.0 means faster than real time.
3. Record the numbers in your report: parameters (printed at start), RTF, and RAM (Task Manager / `htop` peak while running).
4. Theoretical latency: the encoder window is 16 samples = 1 ms of lookahead, plus whatever chunk size you process. Say clearly in the report that `evaluate.py` measures the full-clip speed and that a chunked real-time wrapper is future work.

---

## 9. Demo

1. Record `enroll.wav`: the target person alone, 4 s, quiet room.
2. Record `noisy.wav`: the same person talking while another person talks and there is background noise (fan, music, café video).
3. Run:

```bash
python infer.py --ckpt best.pt --mixture noisy.wav --enroll enroll.wav --out clean.wav
```

Make 3-4 demo triples (noisy / your output / clean reference) with the same person, a different person, and a hard case. Include one honest failure case (very similar voice, very low SNR) in the report.

---

## 10. Honest limits to state in your report

- The model is **causal at frame level**, with a 1 ms encoder lookahead. Say that, instead of "zero latency".
- `BSRNN` and `TF-GridNet` from your proposal are **not implemented**. State them as future work. Conv-TasNet with three conditioning variants plus a no-conditioning baseline is a complete, defensible study.
- The ECAPA encoder is **frozen**. The "fine-tuned encoder" ablation is also future work.
- Reverberation and the DNS noise set are not used. Real-world recordings (your own) are used only for the demo unless you add the local-recording test.
- Your proposal's Expected Outcomes says "speaker-independent speech enhancement". Change it to "target-speaker extraction".

---

## 11. Timeline and team split

Check your real deadline and work backwards. Suggested order:

| Phase | What | Who |
|---|---|---|
| 1 | Steps 1-4 done on every laptop; sanity check passes | everyone |
| 2 | Data download, manifests, embeddings on Colab; archive to Drive | Austina |
| 3 | First full training run (FiLM); measure epoch time; start long run early | Sostika |
| 4 | Concat / none / gating runs; any model tweaks | Prashamsa |
| 5 | Evaluation tables, plots, CPU benchmark | Sostika + Austina |
| 6 | Demos, failure analysis, report, slides | everyone |

Colab free accounts: three people can each run a different experiment from their own account in parallel, as long as everyone reads/writes a shared Drive folder (share the `tse` folder with edit access, and mount the same folder path). Do not train two runs into the same `--out-dir`.

Start the long FiLM run as soon as the smoke test passes. Training time is your bottleneck.

---

## 12. Report outline (matches your proposal)

1. Introduction, problem statement, objectives (reuse your proposal text).
2. Literature review (reuse, add 2-3 sentences on what you changed).
3. Methodology: data and mixing (with the SNR/TIR ranges and curriculum stages), ECAPA embedding, causal Conv-TasNet (hyperparameters), conditioning variants, loss, training setup.
4. Experiments: tables from Step 8 (rows: none/concat/film/gating; columns: the four generalization settings; metrics SI-SDRi, STOI, PESQ).
5. Efficiency: parameters, RTF, memory, latency.
6. Demo results and failure cases.
7. Limitations and future work (Step 10).
8. References.

---

## 13. If something goes wrong

Send me: (1) the exact command, (2) the full error text, (3) which step you were on. Common quick checks:

- `ModuleNotFoundError: tse` → run commands from inside the `tse-project` folder.
- `ValueError: Need at least 2 speakers...` → the manifest and embeddings do not match (different `--root`, or embeddings built from another manifest). Rebuild both with the same `--root`.
- `val SI-SDRi` stays near 0 or negative for many epochs → check the condition `none` run; if it is also near 0, the problem is data or mixing. Check that clean files play fine and `sanity_check.py` passes.
- Loss becomes `nan` → rerun with a lower learning rate `--lr 3e-4`, and without `--amp` to see if it is a precision issue.
