# Canonical sound references

The four user-supplied videos define the sound direction, independently of the blue terminal UI reference. Exact source paths are in `audio_metrics.json`. They are split-screen artist demonstrations; rendering the full pictures also analyzes the presenter and animated score. Automatic timing on them is not a clean object-tracking benchmark.

| ID | Duration | RMS dBFS | Energy below 700 Hz |
| --- | --- | --- | --- |
| ref_01 | 26.330 s | -16.69 | 87.5% |
| ref_02 | 9.287 s | -16.75 | 92.4% |
| ref_03 | 8.938 s | -16.79 | 91.7% |
| ref_04 | 19.225 s | -16.24 | 82.3% |

Spectral averages describe body and brightness, not sound-design quality. The user rejected the former FM/scale-based engine as a cheap toy synthesizer despite closer loudness and EQ. Matching these numbers is not the acceptance criterion.

## Current direction: material-2, 2026-10-04

- Expressive, articulated synthetic actions with clear attacks, changing resonances, releases and gaps.
- Contacts contain imperfect microscopic collisions, low-mid body, inharmonic resonant modes and small debris. The scene retains a stable object character across contacts.
- Motion gestures excite that object with a leading contact and irregular rebounds. Continuous time deformation bends its spectrum with movement. A darker resonant body opens behind the attack. No scale quantization or melodic note generator.
- Friction supports the gesture. The user rejected the first material-engine preview as too noisy and lacking expressive gestures. Version 2 lowers friction/debris and replaces broad noise sweeps with bending resonances.
- No obligatory cinematic impact, final swell or continuous drone. Explicit heavy impacts, tape stops, micro-repeats and dropouts remain available.
- Short reflections, parallel oversampled saturation, gentle EQ/compression and peak limiting finish the sound. Full-clip RMS has a ceiling of -16.6 dBFS; sparse scenes can be quieter. This is RMS, not integrated LUFS. The master additionally uses stereo-linked lookahead peak control and reserves about 2.2 dB of oversampled peak headroom after actual AAC renders revealed codec overshoots. The reference-render script checks the decoded AAC with 4× oversampling and rejects outputs above 0 dBFS.

`foley_engine.py` owns synthesis. `foley.compose` preserves the pipeline interface. `key` and `mode` remain compatible director metadata and do not affect audio. Existing vibe names now choose layer balance and material character.

## Evidence and limits

`qa/render_material_engine.py` renders all four pictures with seed 42. Current WAV/MP4 files, event data, contact sheets and levels live in `qa/material-engine/`. Previous-engine WAVs use `.backups/2026-10-04-before-material-engine/foley.py`. `ref_03_expressive-v2.mp4` is the distinct review copy after the noise feedback.

The renderer exposes diagnostic stems. They share the final limiter envelope and level trim but precede master EQ/compression; summing them does not exactly reconstruct the mastered WAV. Meters use these stems. UI frequency metadata denotes the object's base resonance, not an exact time-varying fundamental.

Tests verify timing, silence, material distinction, deterministic variation, finite/bounded audio and application workflow. They do not establish perceptual equivalence. The user accepted the current result on 2026-10-04: «все супер». Preserve material-2 and the current design unless further changes are requested. Reports under `qa/reference-check/` and `qa/sound_reference_comparison/` describe superseded engines.

Reference measurements use decoded stereo at 24 kHz, 10 ms RMS windows and a 2048-sample Hann STFT. Spectrum bands use a mono downmix. Source audio is not copied into synthesized results.
