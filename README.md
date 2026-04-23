# Helium

**Local-first speech research scaffold for speaker diarization, source separation, and controllable voice conversion.**

Helium is a research codebase — there is no CLI or HTTP API. Interaction with the pipeline and models happens through standalone Python scripts in `scripts/`. This keeps the research loop tight: edit a stage, run a script, inspect artifacts on disk.

## Research Direction

Target problem: `noisy two-speaker speech disentanglement + style-preserving voice conversion`

Example: two people speaking in a cafe → detect two distinct speakers → isolate each → convert one speaker's voice to a different timbre while preserving words, rhythm, and turn-taking.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Optional: full metrics suite
pip install -e ".[metrics]"

# Optional: PyTorch for model backends
pip install -e ".[research]"
```

Copy `.env.example` to `.env` and adjust `HELIUM_DATA_DIR` if needed (default: `~/.helium`).

## Scripts

All interaction with the pipeline happens through scripts in `scripts/`.

### Run the pipeline on audio files

```bash
python scripts/run_pipeline.py --audio-dir ./sample_audio
python scripts/run_pipeline.py --audio-dir ./sample_audio --target-speakers 2 --output ./results
```

Stages run in order: `validate → diarize → separate → convert → evaluate → export`

Scaffold stages (diarize, separate, convert) are marked SKIPPED until you wire in a local model backend. Edit the corresponding file under `helium/pipeline/stages/` and replace the placeholder return with real inference.

### Run a benchmark evaluation

```bash
python scripts/run_benchmark.py --dataset libri2mix --data-dir /data/Libri2Mix --output ./benchmark_results
python scripts/run_benchmark.py --dataset whamr --data-dir /data/whamr --split tt --n-samples 100
python scripts/run_benchmark.py --dataset voxceleb --data-dir /data/VoxCeleb
```

Supported datasets: `libri2mix`, `whamr`, `voxceleb`

To evaluate a custom model, define a `separation_fn` and pass it in `scripts/run_benchmark.py`:

```python
def my_model(mixture_path, sample_rate):
    # load your model, run inference
    return [estimate_s1_array, estimate_s2_array]

separation_fn = my_model  # line ~60 in run_benchmark.py
```

### Compute metrics on existing files

```bash
# Separation metrics (SI-SDR, SDR, PIT-SI-SDR, improvements)
python scripts/compute_metrics.py separation \
    --reference s1_ref.wav s2_ref.wav \
    --estimate  s1_est.wav s2_est.wav \
    --mixture   mixture.wav

# Diarization error rate
python scripts/compute_metrics.py diarization \
    --reference ref_turns.rttm --hypothesis hyp_turns.rttm

# Blind source coherence (no reference required)
python scripts/compute_metrics.py coherence --sources s1_est.wav s2_est.wav

# Speaker similarity (requires resemblyzer or speechbrain)
python scripts/compute_metrics.py speaker-sim --audio-a spk_a.wav --audio-b spk_b.wav

# Word error rate (requires faster-whisper or openai-whisper)
python scripts/compute_metrics.py wer --audio output.wav --reference-text "the quick brown fox"
```

## Plugging In a Model

Each scaffold stage lives in `helium/pipeline/stages/`. To wire in a model:

1. Open the stage file (e.g. `helium/pipeline/stages/separate.py`)
2. Load your model at the top of the `run()` function
3. Replace `return {"status": "placeholder", ...}` with real inference and return a dict with `"artifacts"` pointing to the files you wrote

The `evaluate` and `export` stages are already implemented and will automatically pick up any outputs written by the model stages.

## Suggested Baselines

| Task | Model |
|---|---|
| Diarization | `pyannote/speaker-diarization` |
| Separation | `speechbrain/sepformer-whamr` |
| Voice conversion | `RedRepter/seed-vc-api` |

## Project Structure

```text
helium/
├── config.py               pydantic-settings: data_dir, backends, speaker count
├── models/job.py           Job, StageResult, JobArtifacts, JobStatus
├── pipeline/
│   ├── runner.py           Sequential stage runner with atomic job state saves
│   └── stages/
│       ├── validate.py     IMPLEMENTED — WAV inspection + metadata
│       ├── diarize.py      SCAFFOLD — emits planning JSON, plug in pyannote here
│       ├── separate.py     SCAFFOLD — emits planning JSON, plug in SepFormer here
│       ├── convert.py      SCAFFOLD — emits planning JSON, plug in Seed-VC here
│       ├── evaluate.py     IMPLEMENTED — separation/diarization/WER metrics
│       └── export.py       IMPLEMENTED — artifact manifest
├── metrics/
│   ├── separation.py       SI-SDR, SDR, SDRi, SI-SDRi, PIT-SI-SDR, blind_coherence
│   ├── diarization.py      DER from RTTM files
│   ├── speaker_sim.py      cosine_similarity, compute_speaker_similarity
│   └── content.py          wer, cer, compute_wer_from_audio
├── benchmarks/
│   ├── runner.py           Evaluation loop over a dataset, writes CSV + summary.json
│   └── datasets/           Libri2Mix, WHAMR!, VoxCeleb dataset interfaces
└── storage/local.py        All filesystem I/O: job dirs, artifact paths, atomic saves
scripts/
├── run_pipeline.py         Run full pipeline on a WAV directory
├── run_benchmark.py        Benchmark evaluation on standard datasets
└── compute_metrics.py      Compute individual metrics on existing files
```

## Artifact Layout

Each pipeline run creates a job directory under `~/.helium/jobs/{job_id}/`:

```text
{job_id}/
  audio/                    source WAV files
  artifacts/
    manifests/              validation_report.json
    diarization/            diarization_plan.json → speaker_turns.rttm (when wired)
    separation/             separation_plan.json → speaker_0.wav, speaker_1.wav
    conversion/             conversion_plan.json → converted WAVs
    evaluation/             metrics.json, experiment_summary.json, metrics.csv
    export/                 run_manifest.json
  metadata.json             full job state (JSON)
```

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 0 - Pivot | Done | Repo moved from 3D reconstruction to speech research scaffold |
| 1 - Audio Jobs | Done | WAV storage, job model, pipeline runner |
| 2 - Research Scaffolding | Done | All six pipeline stages, planning artifacts |
| 3 - Local Baselines | TODO | Plug in pyannote, SepFormer, Seed-VC locally |
| 4 - Benchmarking | TODO | Evaluate on Libri2Mix / WHAMR / VoxCeleb |
| 5 - Research Contribution | TODO | Diarization-guided conversion improvement |

## License

Apache 2.0
