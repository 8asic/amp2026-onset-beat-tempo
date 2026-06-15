"""Standalone 127-file validation harness for fast experiment iteration.

Usage:
    python experiments/validate.py                  # current config
    python experiments/validate.py --tempo-method argmax
    python experiments/validate.py --tempo-method comb_fusion --diag-tempo

Audio is cached to experiments/.cache/*.npy so reruns are fast.
mir_eval.onset.f_measure returns (f_measure, precision, recall) -- name the
first value `f` (see memory/gotchas.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mir_eval

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import config  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402
from src.detectors import Pipeline  # noqa: E402

CACHE = ROOT / "experiments" / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)


def load_audio_cached(loader: DataLoader, stem: str, wav: str):
    cpath = CACHE / f"{stem}.npy"
    if cpath.exists():
        return np.load(cpath), config.audio.sample_rate
    y, sr = loader.load_audio(Path(wav))
    if y is not None:
        np.save(cpath, y)
    return y, sr


def tempo_pscore(gt_tempo, est_tempo):
    if len(gt_tempo) == 1:
        ref_tempi = np.array([gt_tempo[0] / 2, gt_tempo[0]])
        ref_weight = 0.5
    elif len(gt_tempo) == 2:
        ref_tempi = np.array(gt_tempo)
        ref_weight = 0.5
    else:
        ref_tempi = np.array(gt_tempo[:2])
        ref_weight = gt_tempo[2]
    est_tempi = np.array(est_tempo[:2])
    p, _, _ = mir_eval.tempo.detection(ref_tempi, ref_weight, est_tempi, tol=0.08)
    return float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tempo-method", default=None, help="argmax | comb | comb_fusion")
    ap.add_argument("--harmonics", type=int, default=None, help="comb harmonics")
    ap.add_argument("--diag-tempo", action="store_true", help="list wrong-tempo files")
    ap.add_argument("--diag-beat", action="store_true", help="list worst-beat files")
    ap.add_argument("--diag-phase", action="store_true", help="phase analysis of tempo-correct beat failures")
    ap.add_argument("--multiband", action="store_true", help="enable multiband onset")
    ap.add_argument("--n-bands", type=int, default=None, help="multiband band count")
    ap.add_argument("--merge-tol", type=float, default=None, help="merge tol ms")
    ap.add_argument("--threshold", type=float, default=None, help="onset threshold")
    ap.add_argument("--whiten", action="store_true", help="enable adaptive whitening")
    ap.add_argument("--whiten-decay", type=float, default=None, help="whiten decay")
    ap.add_argument("--whiten-floor", type=float, default=None, help="whiten floor")
    ap.add_argument("--no-octave-select", action="store_true", help="disable beat octave selection")
    ap.add_argument("--octave-gate", type=float, default=None, help="octave-select gate bpm")
    ap.add_argument("--hpss", action="store_true", help="enable HPSS percussive masking before superflux")
    ap.add_argument("--hpss-kern-harm", type=int, default=None, help="HPSS harmonic median kernel (time frames)")
    ap.add_argument("--hpss-kern-perc", type=int, default=None, help="HPSS percussive median kernel (mel bins)")
    ap.add_argument("--extra-onsets", action="store_true", help="also evaluate on train_extra_onsets (150 files, onset-only)")
    ap.add_argument("--fusion", action="store_true", help="enable multi-ODF fusion onset picking")
    ap.add_argument("--fusion-odfs", default=None, help="comma-list of ODF channels: superflux,complex,hfc,phase")
    ap.add_argument("--learned", action="store_true", help="EXP-016: learned onset activation (NOTE: on 277 this is train-on-train, optimistic; honest number is CV)")
    ap.add_argument("--learned-delta", type=float, default=None, help="peak-pick delta on learned activation")
    ap.add_argument("--learned-beat", action="store_true", help="EXP-018: BLSTM beat activation (needs torch + models/beat_blstm.pt)")
    ap.add_argument("--beat-decode", default=None, choices=["A", "B"], help="EXP-018 decode: B=comb_fusion+octave (default), A=autocorr-on-activation")
    ap.add_argument("--beat-model", default=None, help="EXP-018 override beat model path (e.g. models/beat_blstm_extra.pt for the fair test)")
    ap.add_argument("--beat-decoder", default=None, choices=["dp", "dbn"], help="EXP-024 beat decoder: dp (default) | dbn (joint tempo+phase)")
    ap.add_argument("--dbn-lam-change", type=float, default=None, help="DBN tempo-drift penalty")
    ap.add_argument("--dbn-lam-prior", type=float, default=None, help="DBN comb_fusion-prior anchor")
    ap.add_argument("--tempo-ensemble", action="store_true", help="EXP-025 tempo ensemble (comb_fusion + beat-AC on disagreement)")
    ap.add_argument("--onset-tta", action="store_true", help="EXP-028 pitch-shift TTA for onset CNN")
    ap.add_argument("--no-onset-cnn", action="store_true", help="EXP-020: disable onset CNN (use fusion onset)")
    ap.add_argument("--ibi-tempo", action="store_true", help="EXP-029B: IBI-derived tempo as 3rd oracle")
    ap.add_argument("--onset-blend", type=float, default=None, help="EXP-029C: mix onset CNN act into beat ODF (e.g. 0.15)")
    ap.add_argument("--cnn-model", default=None, help="EXP-020 override onset CNN path (e.g. models/onset_cnn_extra.pt)")
    ap.add_argument("--cnn-delta", type=float, default=None, help="EXP-020 peak-pick delta on the CNN activation")
    args = ap.parse_args()

    if args.tempo_method is not None:
        config.beat.tempo_method = args.tempo_method
    if args.harmonics is not None:
        config.beat.tempo_comb_harmonics = args.harmonics
    if args.multiband:
        config.onset.multiband = True
    if args.n_bands is not None:
        config.onset.n_bands = args.n_bands
    if args.merge_tol is not None:
        config.onset.merge_tol_ms = args.merge_tol
    if args.threshold is not None:
        config.onset.threshold = args.threshold
    if args.whiten:
        config.audio.whiten = True
    if args.whiten_decay is not None:
        config.audio.whiten_decay = args.whiten_decay
    if args.whiten_floor is not None:
        config.audio.whiten_floor = args.whiten_floor
    if args.no_octave_select:
        config.beat.beat_octave_select = False
    if args.octave_gate is not None:
        config.beat.beat_octave_gate = args.octave_gate
    if args.fusion:
        config.onset.fusion = True
    if args.fusion_odfs is not None:
        config.onset.fusion_odfs = tuple(args.fusion_odfs.split(","))
    if args.learned:
        config.onset.learned = True
    if args.learned_delta is not None:
        config.onset.learned_delta = args.learned_delta
    if args.learned_beat:
        config.beat.learned = True
    if args.beat_decode is not None:
        config.beat.learned_decode = args.beat_decode
    if args.beat_decoder is not None:
        config.beat.decoder = args.beat_decoder
    if args.dbn_lam_change is not None:
        config.beat.dbn_lam_change = args.dbn_lam_change
    if args.dbn_lam_prior is not None:
        config.beat.dbn_lam_prior = args.dbn_lam_prior
    if args.beat_model is not None:
        config.beat.learned_model_path = args.beat_model
    if args.tempo_ensemble:
        config.beat.tempo_ensemble = True
    if args.onset_tta:
        config.audio.onset_tta_shifts = (-2,-1,0,1,2)
    if args.no_onset_cnn:
        config.onset.cnn = False
    if args.cnn_model is not None:
        config.onset.cnn_model_path = args.cnn_model
    if args.cnn_delta is not None:
        config.onset.cnn_delta = args.cnn_delta
    if args.ibi_tempo:
        config.beat.ibi_tempo = True
    if args.onset_blend is not None:
        config.beat.onset_blend = args.onset_blend
    if args.hpss:
        config.audio.hpss = True
    if args.hpss_kern_harm is not None:
        config.audio.hpss_kernel_harm = args.hpss_kern_harm
    if args.hpss_kern_perc is not None:
        config.audio.hpss_kernel_perc = args.hpss_kern_perc

    loader = DataLoader()
    train_dir = ROOT / "data" / "processed" / "train"
    train = loader.load_train(train_dir)

    if args.extra_onsets:
        extra_dir = ROOT / "data" / "processed" / "train_extra_onsets"
        extra = loader.load_extra_onsets(extra_dir)
        train = {**train, **extra}

    pipe = Pipeline()

    onset_f, beat_f, tempo_p = [], [], []
    tempo_wrong = []
    beat_bad = []
    beat_phase = []

    for stem, gt in train.items():
        y, sr = load_audio_cached(loader, stem, gt["wav"])
        if y is None:
            continue
        onsets, beats, tempos = pipe.process_file(y, sr)

        if gt.get("onsets") and len(onsets):
            f, _, _ = mir_eval.onset.f_measure(
                np.array(gt["onsets"]), np.array(onsets), window=0.05
            )
            onset_f.append(f)
        bf = None
        if gt.get("beats") and len(beats):
            bf = mir_eval.beat.f_measure(np.array(gt["beats"]), np.array(beats))
            beat_f.append(bf)
        pt = None
        if gt.get("tempo") and tempos:
            pt = tempo_pscore(gt["tempo"], tempos)
            tempo_p.append(pt)
            if pt < 0.5 and args.diag_tempo:
                tempo_wrong.append((stem, gt["tempo"], [round(t, 1) for t in tempos]))
        if args.diag_beat and bf is not None and bf < 0.5:
            n_gt = len(gt["beats"]) if gt.get("beats") else 0
            beat_bad.append((stem, round(bf, 3), n_gt, len(beats),
                             round(pt, 2) if pt is not None else None))
        if (args.diag_phase and bf is not None and bf < 0.5
                and pt is not None and pt >= 0.5
                and gt.get("beats") and len(gt["beats"]) >= 3 and len(beats) >= 3):
            gtb = np.asarray(gt["beats"], dtype=float)
            pb = np.asarray(beats, dtype=float)
            period = float(np.median(np.diff(gtb)))
            # signed offset of each pred beat to nearest GT beat
            idx = np.searchsorted(gtb, pb)
            idx = np.clip(idx, 1, len(gtb) - 1)
            left = gtb[idx - 1]
            right = gtb[idx]
            nearest = np.where(np.abs(pb - left) <= np.abs(pb - right), left, right)
            off = pb - nearest
            phase_frac = float(np.median(np.abs(off)) / period) if period > 0 else 0.0
            beat_phase.append((stem, round(bf, 3), round(period, 3),
                               round(float(np.median(off)), 3),
                               round(phase_frac, 3)))

    mo = np.mean(onset_f) if onset_f else 0.0
    mb = np.mean(beat_f) if beat_f else 0.0
    mt = np.mean(tempo_p) if tempo_p else 0.0
    mean = (mo + mb + mt) / 3

    print(f"tempo_method={config.beat.tempo_method}")
    print(f"  Onset F1:  {mo:.4f}  ({len(onset_f)} files)")
    print(f"  Beat F1:   {mb:.4f}  ({len(beat_f)} files)")
    print(f"  Tempo:     {mt:.4f}  ({len(tempo_p)} files)")
    print(f"  MEAN:      {mean:.4f}")

    if args.diag_tempo:
        print(f"\nWrong-tempo files (p<0.5): {len(tempo_wrong)}")
        for stem, gt_t, est_t in tempo_wrong:
            print(f"  {stem}: GT={gt_t} EST={est_t}")

    if args.diag_beat:
        beat_bad.sort(key=lambda x: x[1])
        print(f"\nWorst-beat files (F<0.5): {len(beat_bad)}  (stem, beatF, nGTbeats, nPred, tempoP)")
        for stem, bf, n_gt, n_pred, pt in beat_bad:
            print(f"  {stem}: beatF={bf} nGT={n_gt} nPred={n_pred} tempoP={pt}")

    if args.diag_phase:
        beat_phase.sort(key=lambda x: x[1])
        print(f"\nTempo-correct but beat-bad (F<0.5, tempoP>=0.5): {len(beat_phase)}")
        print("  (stem, beatF, period_s, medSignedOff_s, |off|/period)")
        print("  phase_frac near 0.5 = anti-phase (off-beat); near 0 = aligned-but-jittery")
        for stem, bf, period, medoff, frac in beat_phase:
            print(f"  {stem}: beatF={bf} period={period} medOff={medoff} frac={frac}")


if __name__ == "__main__":
    main()
