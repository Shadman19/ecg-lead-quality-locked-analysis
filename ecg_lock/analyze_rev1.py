# REVISION-1 COPY of analyze.py (Oct 2026): adds augmentation-matched diag-only baseline + per-seed clean table.
"""Regenerate every manuscript number from locked bundles (all configs x seeds).

Statistics
  * clean metrics: mean +- SD over training seeds, plus seed-averaged paired
    patient-level percentile bootstrap CIs for prespecified contrasts;
  * stress: identical corruption realizations across models (seeded by condition),
    so model contrasts are paired; per level AUROC is averaged over realizations;
  * per-lead: learned synthetic-intact head vs transparent rules (variance, HF energy);
  * repair: oracle vs predicted mask, paired;
  * abstention: area under the risk-coverage curve (AURC) for gates.
Outputs: results_lock/*.csv, *.tex, figures/*.png, and results_manifest.json (bundle hashes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

CONTRASTS = [  # (B, A): report B - A
    ("cnn_full", "cnn_diagonly_aug"),
    ("cnn_full", "resnet_aug"),
    ("cnn_diagonly_aug", "cnn_diagonly_clean"),
    ("resnet_aug", "resnet_clean"),
    ("cnn_full_leadwise", "cnn_full"),
    ("cnn_full_cleanfrac20", "cnn_full"),
    ("cnn_full_leadwise_cleanfrac20", "cnn_full"),
    ("tx_full", "cnn_full"), ("cnn_full_leadwise_cleanfrac20", "cnn_diagonly_aug_matched"), ("cnn_diagonly_aug_matched", "cnn_diagonly_aug"), ("cnn_diagonly_aug_matched", "cnn_diagonly_clean"),
]


def macro_auc(y, p):
    v = [roc_auc_score(y[:, k], p[:, k]) for k in range(y.shape[1]) if 0 < y[:, k].mean() < 1]
    return float(np.mean(v)) if v else np.nan


def macro_ap(y, p):
    v = [average_precision_score(y[:, k], p[:, k]) for k in range(y.shape[1]) if 0 < y[:, k].mean() < 1]
    return float(np.mean(v)) if v else np.nan


def ece(y, p, bins=15):
    y, p = y.ravel().astype(float), p.ravel().astype(float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))


METRICS = {
    "auroc": macro_auc,
    "auprc": macro_ap,
    "f1": lambda y, p: float(f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0)),
    "ece": ece,
    "brier": lambda y, p: float(np.mean((p - y) ** 2)),
}


def groups_of(pid):
    _, inv = np.unique(pid, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, bounds)


def boot_idx(groups, rng):
    return np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def paired_seed_avg(y, pid, A: list, B: list, metric, n=1000, seed=20260926):
    """CI for mean over seeds of metric(B_s) - metric(A_s), resampling patients jointly."""
    g, rng = groups_of(pid), np.random.default_rng(seed)
    point = np.mean([metric(y, b) - metric(y, a) for a, b in zip(A, B)])
    d = []
    for _ in range(n):
        ix = boot_idx(g, rng)
        d.append(np.mean([metric(y[ix], b[ix]) - metric(y[ix], a[ix]) for a, b in zip(A, B)]))
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def aurc(y, p, conf, thr=0.5):
    """Area under selective-risk (Hamming) vs coverage curve; accept most-confident first."""
    err = ((p >= thr).astype(int) != y).mean(1)
    order = np.argsort(-conf, kind="stable")
    cum = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    return float(cum.mean())


def fmt(m, s=None, d=4):
    return f"{m:.{d}f}" if s is None or np.isnan(s) else f"{m:.{d}f} $\\pm$ {s:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    out = a.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    runs = defaultdict(dict)
    for d in sorted(a.runs.iterdir()):
        m = re.match(r"(.+)_s(\d+)$", d.name)
        if m and (d / "lock.json").exists() and "test_stress" in (d / "lock.json").read_text():
            runs[m.group(1)][int(m.group(2))] = d
    manifest = {"runs": {}, "bootstrap": a.bootstrap}
    for cfg_, seeds_ in runs.items():
        for s_, d_ in seeds_.items():
            v = load(d_ / "bundles" / "val_clean.npz")["patient_id"]; t_ = load(d_ / "bundles" / "test_clean.npz")["patient_id"]
            assert not set(v.tolist()) & set(t_.tolist()), f"val/test patient overlap in {d_.name}"
    manifest["checks"] = ["val/test patient-disjoint for every run", "identical eval inputs asserted for primary contrasts"]
    for cfg, seeds in runs.items():
        for s, d in seeds.items():
            manifest["runs"][f"{cfg}_s{s}"] = json.loads((d / "lock.json").read_text())

    # ---------- clean test ----------
    rows, clean, seedrows = [], {}, []
    for cfg, seeds in runs.items():
        per = []
        for s in sorted(seeds):
            z = load(seeds[s] / "bundles" / "test_clean.npz")
            clean[(cfg, s)] = z
            per.append({k: f(z["y_true"], z["y_score"]) for k, f in METRICS.items()})
            per[-1]["auroc_mc"] = macro_auc(z["y_true"], z["y_score_mc"]); seedrows.append({"config": cfg, "seed": s, **per[-1]})
        df = pd.DataFrame(per)
        row = {"config": cfg, "n_seeds": len(per)}
        for k in df.columns:
            row[k + "_mean"], row[k + "_sd"] = df[k].mean(), df[k].std(ddof=1) if len(df) > 1 else np.nan
        rows.append(row)
    T1 = pd.DataFrame(rows).sort_values("config")
    T1.to_csv(out / "T_clean_test.csv", index=False); pd.DataFrame(seedrows).to_csv(out / "T_clean_test_per_seed.csv", index=False)

    # ---------- paired contrasts (clean) ----------
    crow = []
    for b, a_ in CONTRASTS:
        if b not in runs or a_ not in runs:
            continue
        ss = sorted(set(runs[b]) & set(runs[a_]))
        if not ss:
            continue
        y = clean[(b, ss[0])]["y_true"]
        pid = clean[(b, ss[0])]["patient_id"]
        for s in ss:
            assert np.array_equal(clean[(b, s)]["ecg_id"], clean[(a_, s)]["ecg_id"])
        for met in ("auroc", "auprc", "f1", "ece", "brier"):
            pt, lo, hi = paired_seed_avg(y, pid, [clean[(a_, s)]["y_score"] for s in ss],
                                         [clean[(b, s)]["y_score"] for s in ss], METRICS[met], a.bootstrap)
            crow.append({"B": b, "A": a_, "metric": met, "delta_B_minus_A": pt, "ci_lo": lo, "ci_hi": hi, "seeds": len(ss)})
    pd.DataFrame(crow).to_csv(out / "T_clean_contrasts.csv", index=False)

    # ---------- stress (paired realizations) ----------
    srows = []
    stress_scores = {}
    for cfg, seeds in runs.items():
        for s, d in seeds.items():
            z = load(d / "bundles" / "test_stress.npz")
            y = z["y_true"]
            lv = defaultdict(list)
            for k in z:
                if k.endswith("|y_score"):
                    cond = k.rsplit("|r=", 1)[0]
                    lv[cond].append(z[k])
            for cond, arrs in lv.items():
                stress_scores[(cfg, s, cond)] = arrs
                srows.append({"config": cfg, "seed": s, "condition": cond,
                              "auroc": float(np.mean([macro_auc(y, p) for p in arrs])), "n_realizations": len(arrs)})
            stress_y, stress_pid = y, z["patient_id"]
    S = pd.DataFrame(srows)
    S.to_csv(out / "T_stress_raw.csv", index=False)
    Sg = S.groupby(["config", "condition"]).auroc.agg(["mean", "std", "count"]).reset_index()
    Sg.to_csv(out / "T_stress_summary.csv", index=False)
    # paired stress deltas vs augmentation-matched diagnosis-only
    prow = []
    for b, a_ in [("cnn_full", "cnn_diagonly_aug"), ("cnn_full", "resnet_aug"), ("resnet_aug", "resnet_clean"),
                  ("cnn_full_leadwise_cleanfrac20", "cnn_diagonly_aug"), ("cnn_full_leadwise_cleanfrac20", "cnn_diagonly_aug_matched"), ("cnn_diagonly_aug_matched", "cnn_diagonly_clean")]:
        if b not in runs or a_ not in runs:
            continue
        ss = sorted(set(runs[b]) & set(runs[a_]))
        if not ss:
            continue
        for cond in sorted(S.condition.unique()):
            A = [np.mean([p for p in stress_scores[(a_, s, cond)]], 0) for s in ss]
            B = [np.mean([p for p in stress_scores[(b, s, cond)]], 0) for s in ss]
            # realization-averaged probabilities keep pairing; metric on averaged scores
            pt, lo, hi = paired_seed_avg(stress_y, stress_pid, A, B, macro_auc, max(200, a.bootstrap // 5))
            prow.append({"B": b, "A": a_, "condition": cond, "delta_auroc": pt, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(prow).to_csv(out / "T_stress_contrasts.csv", index=False)

    # ---------- per-lead head vs rules; repair; gates ----------
    lrows, rrows, grows = [], [], []
    for cfg, seeds in runs.items():
        for s, d in seeds.items():
            fo = d / "bundles" / "test_traindist_oracle.npz"
            fp = d / "bundles" / "test_traindist_predicted.npz"
            if fo.exists():
                z = load(fo)
                m, ct, sg = z["corruption_mask"], z["corruption_type"], z["noise_sigma"]
                from scipy.stats import rankdata
                rv, rh = -z["rule_variance"], -z["rule_hf"]
                # label-free combination: max of within-array percentile ranks (flat OR noisy)
                comb = np.maximum(rankdata(rv.ravel()), rankdata(rh.ravel())).reshape(rv.shape)
                scorers = {"rule_variance": rv, "rule_hf": rh, "rule_combined": comb}
                if "reliability_score" in z:
                    scorers["learned_head"] = 1 - z["reliability_score"]
                for name, sc in scorers.items():
                    r = {"config": cfg, "seed": s, "detector": name,
                         "auroc_all": roc_auc_score(m.ravel(), sc.ravel()),
                         "auprc_all": average_precision_score(m.ravel(), sc.ravel())}
                    for code, lab in ((1, "dropout"), (2, "noise")):
                        keep = (ct == code) | (ct == 0)
                        r[f"auroc_{lab}"] = roc_auc_score(m[keep], sc[keep])
                    for lo_, hi_ in ((0, 0.05), (0.05, 0.15), (0.15, 0.26)):
                        keep = (ct == 0) | ((ct == 2) & (sg >= lo_) & (sg < hi_))
                        r[f"auroc_noise_sigma_{lo_}-{hi_}"] = roc_auc_score(m[keep], sc[keep])
                    lrows.append(r)
                cz = clean.get((cfg, s))
                if cz is not None and "lead_score_clean" in cz:
                    lrows.append({"config": cfg, "seed": s, "detector": "learned_head_clean_mean_intact",
                                  "auroc_all": float(cz["lead_score_clean"].mean())})
                for f, src in ((fo, "oracle"), (fp, "predicted")):
                    if not f.exists():
                        continue
                    zz = load(f)
                    rrows.append({"config": cfg, "seed": s, "mask": src,
                                  "auroc_pre": macro_auc(zz["y_true"], zz["pre_repair_score"]),
                                  "auroc_post": macro_auc(zz["y_true"], zz["post_repair_score"]),
                                  "replaced_leads_per_record": float(zz["repair_mask"].sum(1).mean()),
                                  "mask_sensitivity": float((zz["repair_mask"] * m).sum() / max(m.sum(), 1)),
                                  "mask_false_replace_rate": float((zz["repair_mask"] * (1 - m)).sum() / max((1 - m).sum(), 1))})
                # gates on corrupted input (AURC, lower is better)
                y, p = z["y_true"], z["y_score"]
                rng = np.random.default_rng(s)
                g = {"random": float(np.mean([aurc(y, p, rng.random(len(y))) for _ in range(50)])),
                     "uncertainty": aurc(y, p, -z["uncertainty"]),
                     "max_prob_confidence": aurc(y, p, np.abs(p - 0.5).mean(1))}
                if "reliability_score" in z:
                    g["mean_intact_score"] = aurc(y, p, z["reliability_score"].mean(1))
                    g["intact_x_certainty"] = aurc(y, p, z["reliability_score"].mean(1) - z["uncertainty"] / (z["uncertainty"].std() + 1e-9) * z["reliability_score"].mean(1).std())
                g["oracle_fraction_corrupted"] = aurc(y, p, -m.mean(1))
                grows.append({"config": cfg, "seed": s, **g})
    L = pd.DataFrame(lrows); L.to_csv(out / "T_perlead_raw.csv", index=False)
    if len(L):
        L.groupby(["config", "detector"]).mean(numeric_only=True).drop(columns="seed").to_csv(out / "T_perlead_summary.csv")
    R = pd.DataFrame(rrows)
    if len(R):
        R["delta"] = R.auroc_post - R.auroc_pre
        R.to_csv(out / "T_repair_raw.csv", index=False)
        R.groupby(["config", "mask"]).agg(["mean", "std"]).drop(columns="seed").to_csv(out / "T_repair_summary.csv")
    G = pd.DataFrame(grows)
    if len(G):
        G.to_csv(out / "T_gate_aurc_raw.csv", index=False)
        G.groupby("config").agg(["mean", "std"]).drop(columns="seed").to_csv(out / "T_gate_aurc_summary.csv")

    # ---------- real recorded artifacts ----------
    nrows = []
    for cfg, seeds in runs.items():
        for s, d in seeds.items():
            f = d / "bundles" / "test_nstdb.npz"
            if not f.exists():
                continue
            z = load(f)
            for k in z:
                if k.endswith("|y_score"):
                    cond = k[: -len("|y_score")]
                    r = {"config": cfg, "seed": s, "condition": cond, "diag_auroc": macro_auc(z["y_true"], z[k])}
                    mk = z[cond + "|mask"]
                    r["det_auroc_rule_variance"] = roc_auc_score(mk.ravel(), -z[cond + "|rule_variance"].ravel())
                    r["det_auroc_rule_hf"] = roc_auc_score(mk.ravel(), -z[cond + "|rule_hf"].ravel())
                    if cond + "|lead" in z:
                        r["det_auroc_learned"] = roc_auc_score(mk.ravel(), 1 - z[cond + "|lead"].astype(float).ravel())
                    nrows.append(r)
    if nrows:
        N = pd.DataFrame(nrows); N.to_csv(out / "T_nstdb_raw.csv", index=False)
        N.groupby(["config", "condition"]).mean(numeric_only=True).drop(columns="seed").to_csv(out / "T_nstdb_summary.csv")

    # ---------- LaTeX: main comparison table ----------
    order = [c for c in ["cnn_full", "cnn_full_leadwise_cleanfrac20", "cnn_diagonly_aug", "cnn_diagonly_aug_matched", "cnn_diagonly_clean",
                         "cnn_diag_lead", "cnn_diag_rec", "tx_full", "resnet_aug", "resnet_clean",
                         "inception_aug", "inception_clean"] if c in set(T1.config)]
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "Configuration & AUROC & AUPRC & F1 & ECE & Brier \\\\", "\\midrule"]
    for c in order:
        r = T1[T1.config == c].iloc[0]
        lines.append(c.replace("_", "\\_") + " & " + " & ".join(fmt(r[f"{k}_mean"], r[f"{k}_sd"]) for k in METRICS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_clean.tex").write_text("\n".join(lines) + "\n")

    # ---------- figure: comparative stress curves ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, fam, xkey in zip(axes, ("missing", "noise", "bw"), ("k", "s", "bw")):
            for c in [c for c in ("cnn_full", "cnn_diagonly_aug", "cnn_diagonly_aug_matched", "cnn_diagonly_clean", "resnet_aug", "resnet_clean",
                                  "cnn_full_leadwise_cleanfrac20") if c in runs]:
                sub = Sg[(Sg.config == c) & Sg.condition.str.startswith(fam + "|")].copy()
                sub["x"] = sub.condition.str.extract(rf"{xkey}=([0-9.]+)").astype(float)
                sub = sub.sort_values("x")
                ax.errorbar(sub.x, sub["mean"], yerr=sub["std"], marker="o", ms=3, capsize=2, label=c)
            ax.set_title({"missing": "Missing leads (k)", "noise": "Gaussian noise $\\sigma$", "bw": "Baseline wander amplitude"}[fam])
            ax.set_ylabel("Macro AUROC (test)")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "figures" / "stress_comparative.png", dpi=200)
    except Exception as exc:  # plotting must never block the numbers
        print("plot skipped:", exc)

    # ---------- LaTeX macros: the ONLY way numbers enter the manuscript ----------
    def mname(*parts):
        words = "_".join(parts).replace("f1", "fone").replace("20", "twenty")
        txt = "".join(p.title() for p in re.split(r"[^A-Za-z]+", words) if p)
        return "\\Res" + txt
    mac = ["% AUTO-GENERATED by analyze.py from locked bundles. Do not edit by hand."]
    for _, r in T1.iterrows():
        for k in METRICS:
            mac.append(f"\\newcommand{{{mname(r.config, k)}}}{{{fmt(r[k + '_mean'], r[k + '_sd'])}}}")
    for r in crow:
        mac.append(f"\\newcommand{{{mname('d', r['B'], 'vs', r['A'], r['metric'])}}}"
                   f"{{{r['delta_B_minus_A']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]}}")
    (out / "results_macros.tex").write_text("\n".join(mac) + "\n")


    # ================= PRE-REGISTERED PRIMARY ENDPOINTS (see PREREG_ANALYSIS_PLAN.md) =================
    PROPOSED = "cnn_full_leadwise_cleanfrac20"
    rows_p = []

    def seed_boot(pid, fn_per_seed, n, alpha, seed=20260926):
        """fn_per_seed(idx) -> list of per-seed deltas. Patients resampled jointly for all seeds/models."""
        g, rng = groups_of(pid), np.random.default_rng(seed)
        full = fn_per_seed(np.arange(len(pid)))
        reps = []
        for _ in range(n):
            ix = boot_idx(g, rng)
            reps.append(np.mean(fn_per_seed(ix)))
        lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return full, float(np.mean(full)), float(lo), float(hi)

    def lead_rows(ix, m):
        return ix  # records; leads expanded inside

    if PROPOSED in runs:
        ss = sorted(runs[PROPOSED])
        Z = {s: load(runs[PROPOSED][s] / "bundles" / "test_traindist_predicted.npz") for s in ss}
        pid = Z[ss[0]]["patient_id"]
        # (a) learned head vs combined rule on noise-corrupted vs intact leads
        from scipy.stats import rankdata

        def delta_a(ix):
            out = []
            for s in ss:
                z = Z[s]
                ct, m = z["corruption_type"][ix], z["corruption_mask"][ix]
                keep = (ct == 2) | (ct == 0)
                learned = (1 - z["reliability_score"][ix])[keep]
                rv, rh = -z["rule_variance"][ix], -z["rule_hf"][ix]
                comb = np.maximum(rankdata(rv.ravel()), rankdata(rh.ravel())).reshape(rv.shape)[keep]
                # Endpoint A = macro average over the 12 lead-specific AUROCs (pre-registered)
                lm = (1 - z["reliability_score"][ix])
                cm = np.maximum(rankdata(rv.ravel()), rankdata(rh.ravel())).reshape(rv.shape)
                dl, dc = [], []
                for L in range(12):
                    k = keep[:, L]
                    if 0 < m[k, L].mean() < 1:
                        dl.append(roc_auc_score(m[k, L], lm[k, L])); dc.append(roc_auc_score(m[k, L], cm[k, L]))
                out.append(float(np.mean(dl) - np.mean(dc)))
            return out
        per, pt, lo, hi = seed_boot(pid, delta_a, a.bootstrap, 0.05)
        rows_p.append({"endpoint": "A_perlead_noise_auroc_learned_minus_combined_rule", "model": PROPOSED,
                       "per_seed": per, "delta": pt, "ci_lo": lo, "ci_hi": hi, "ci_level": 0.95,
                       "pass": lo > 0})
        # (b1) diagnosis macro AUROC under train-distribution corruption: proposed vs augmentation-matched diag-only
        for B1BASE in [b_ for b_ in ("cnn_diagonly_aug_matched", "cnn_diagonly_aug") if b_ in runs]:
            ss2 = sorted(set(ss) & set(runs[B1BASE]))
            D = {s: load(runs[B1BASE][s] / "bundles" / "test_traindist.npz") for s in ss2}
            y = Z[ss2[0]]["y_true"]
            for s in ss2:
                assert np.array_equal(D[s]["ecg_id"], Z[s]["ecg_id"])
                for fld in ("corruption_mask", "corruption_type", "noise_sigma", "baseline_wander", "y_true", "patient_id"):
                    assert np.array_equal(D[s][fld], Z[s][fld]), f"eval inputs not identical: {fld}"
                assert str(D[s]["corruption_seed"]) == str(Z[s]["corruption_seed"])
            def delta_b1(ix):
                return [macro_auc(y[ix], Z[s]["y_score"][ix]) - macro_auc(y[ix], D[s]["y_score"][ix]) for s in ss2]
            per, pt, lo, hi = seed_boot(pid, delta_b1, a.bootstrap, 0.025)
            rows_p.append({"endpoint": "B1_corrupted_macro_auroc_proposed_minus_" + B1BASE, "model": PROPOSED,
                           "per_seed": per, "delta": pt, "ci_lo": lo, "ci_hi": hi, "ci_level": 0.975, "pass": lo > 0})
        # (b2) predicted-mask repair gain (post - pre) for proposed model
        def delta_b2(ix):
            return [macro_auc(Z[s]["y_true"][ix], Z[s]["post_repair_score"][ix]) -
                    macro_auc(Z[s]["y_true"][ix], Z[s]["pre_repair_score"][ix]) for s in ss]
        per, pt, lo, hi = seed_boot(pid, delta_b2, a.bootstrap, 0.025)
        rows_p.append({"endpoint": "B2_predicted_mask_repair_macro_auroc_post_minus_pre", "model": PROPOSED,
                       "per_seed": per, "delta": pt, "ci_lo": lo, "ci_hi": hi, "ci_level": 0.975, "pass": lo > 0})
        # clean-input sanity gate (descriptive, must be reported)
        for cfgname in [c for c in (PROPOSED, "cnn_full") if c in runs]:
            for s in sorted(runs[cfgname]):
                cz = clean.get((cfgname, s))
                lk = json.loads((runs[cfgname][s] / "lock.json").read_text())
                if cz is not None and "lead_score_clean" in cz:
                    sc = cz["lead_score_clean"]
                    rows_p.append({"endpoint": "SANITY_clean_input_intact_score", "model": f"{cfgname}_s{s}",
                                   "mean_intact_score": float(sc.mean()),
                                   "frac_clean_leads_flagged_at_tau_mask": float((sc < lk.get("tau_mask", 0.5)).mean())})
        A = [r for r in rows_p if r["endpoint"].startswith("A_")]
        B = [r for r in rows_p if r["endpoint"].startswith("B")]
        decision = "GO" if (A and A[0]["pass"] and any(r["pass"] for r in B)) else "RE-SCOPE"
        rows_p.append({"endpoint": "DECISION", "model": PROPOSED, "decision": decision})
        pd.DataFrame(rows_p).to_csv(out / "PRIMARY_endpoints_and_decision.csv", index=False)
        print("PRE-REGISTERED DECISION:", decision)

    # AURC at fixed coverages (0.8, 0.9) for gates, validation-free (ordering only)
    cov_rows = []
    for cfg, seeds in runs.items():
        for s, d in seeds.items():
            f = d / "bundles" / "test_traindist_oracle.npz"
            if not f.exists():
                continue
            z = load(f)
            y, p = z["y_true"], z["y_score"]
            err = ((p >= 0.5).astype(int) != y).mean(1)
            confs = {"uncertainty_only": -z["uncertainty"]}
            if "reliability_score" in z:
                confs["intact_score_only"] = z["reliability_score"].mean(1)
                r1 = rankdata(z["reliability_score"].mean(1)); r2 = rankdata(-z["uncertainty"])
                confs["combined_rank_mean"] = (r1 + r2) / 2
            for name, c in confs.items():
                order = np.argsort(-c, kind="stable")
                for cov in (0.8, 0.9):
                    k = int(round(cov * len(err)))
                    cov_rows.append({"config": cfg, "seed": s, "gate": name, "coverage": cov,
                                     "selective_risk_hamming": float(err[order[:k]].mean())})
            for cov in (0.8, 0.9):
                cov_rows.append({"config": cfg, "seed": s, "gate": "all_samples", "coverage": 1.0,
                                 "selective_risk_hamming": float(err.mean())})
    if cov_rows:
        pd.DataFrame(cov_rows).to_csv(out / "T_gate_fixed_coverage.csv", index=False)

    manifest["outputs"] = {p.name: sha(p) for p in sorted(out.glob("*.csv")) + sorted(out.glob("*.tex"))}
    (out / "results_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
