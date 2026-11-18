"""Step 7: tables and figures built ONLY from saved experiment outputs (results/**). Re-run any time; plots whose
inputs do not exist yet are skipped. Nothing here simulates data: training curves are drawn only for models that
recorded epoch-wise histories (LightGBM, XGBoost, MLP)."""
import _setup  # noqa: F401
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.common import load_config, resolve

COL = {"base": "#3b6fb6", "alt": "#d9822b", "good": "#2e9e44", "bad": "#c0392b"}


def comparison(res, fig):
    p = res / "models" / "comparison.jsonl"
    if not p.exists():
        return
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = []
    for r in recs:
        s = r["student"]
        rows.append({
            "Model": r["model"], "Algorithm family": r["family"], "Key configuration": r["config"],
            "CV Macro F1 (mean)": r["cv_macro_f1_mean"], "CV Macro F1 (std)": r["cv_macro_f1_std"],
            "CV Top-1 (mean)": r["cv_top1_mean"], "CV log loss": r["cv_logloss_mean"],
            "Student Top-1": s["top1_acc"], "Student Macro F1 (candidate)": s["cand_macro_f1"],
            "Student Macro F1 (observation-level)": s["obs_macro_f1"], "Student precision (+class)": s["cand_precision"],
            "Student recall (+class)": s["cand_recall"], "Student log loss": s["log_loss"],
            "Route continuity (h=3)": s["route_continuity_h3"], "Route continuity (h=1)": s["route_continuity_h1"],
            "Candidate miss rate": s["candidate_miss_rate"], "Train macro F1": r["train_macro_f1"],
            "Train time (s)": r["train_time_s"], "Inference (ms / 1000 fixes)": r["inference_ms_per_1000_fixes"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(res / "models" / "comparison_table.csv", index=False)
    (res / "models" / "comparison_table.md").write_text(df.round(4).to_markdown(index=False))
    names = df["Model"].tolist()
    x = np.arange(len(names))

    fig_, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].bar(x, df["CV Macro F1 (mean)"], yerr=df["CV Macro F1 (std)"], color=COL["base"], capsize=4)
    ax[0].set_xticks(x, names, rotation=35, ha="right"); ax[0].set_ylabel("CV Macro F1 (5-fold, grouped by trip)")
    ax[0].set_title("Cross-validation Macro F1 (training trips)"); ax[0].set_ylim(max(0.5, df["CV Macro F1 (mean)"].min() - .05), 1)
    folds = [r["cv_fold_top1"] for r in recs]
    ax[1].boxplot(folds, tick_labels=names); ax[1].set_xticklabels(names, rotation=35, ha="right")
    ax[1].set_ylabel("Top-1 accuracy per fold"); ax[1].set_title("CV Top-1 across folds (stability)")
    plt.tight_layout(); plt.savefig(fig / "cv_macro_f1_and_stability.png", dpi=130); plt.close()

    fig_, ax = plt.subplots(1, 3, figsize=(18, 5))
    w = 0.38
    ax[0].bar(x - w / 2, df["Student Top-1"], w, label="Top-1 accuracy", color=COL["base"])
    ax[0].bar(x + w / 2, df["Student Macro F1 (candidate)"], w, label="Macro F1 (candidate)", color=COL["alt"])
    ax[0].set_xticks(x, names, rotation=35, ha="right"); ax[0].legend(); ax[0].set_title("Student-test: Top-1 accuracy and Macro F1")
    ax[0].set_ylim(max(0.5, min(df["Student Top-1"].min(), df["Student Macro F1 (candidate)"].min()) - .05), 1)
    ax[1].bar(x, df["Student log loss"], color=COL["base"]); ax[1].set_xticks(x, names, rotation=35, ha="right")
    ax[1].set_title("Student-test log loss (lower is better)")
    ax[2].scatter(df["Inference (ms / 1000 fixes)"], df["Student Top-1"], s=70, c=COL["alt"])
    for xx, yy, n in zip(df["Inference (ms / 1000 fixes)"], df["Student Top-1"], names):
        ax[2].annotate(n, (xx, yy), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax[2].set_xscale("log"); ax[2].set_xlabel("inference time (ms / 1000 fixes, log)"); ax[2].set_ylabel("student Top-1")
    ax[2].set_title("Accuracy vs inference cost")
    plt.tight_layout(); plt.savefig(fig / "student_test_comparison.png", dpi=130); plt.close()

    # train vs CV vs student macro F1 (over/under-fitting evidence)
    fig_, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, df["Train macro F1"], w / 1.2, label="train (final fit)", color="#9bb7e0")
    ax.bar(x, df["CV Macro F1 (mean)"], w / 1.2, label="CV (held-out trips)", color=COL["base"])
    ax.bar(x + w, df["Student Macro F1 (candidate)"], w / 1.2, label="student-test", color=COL["alt"])
    ax.set_xticks(x, names, rotation=35, ha="right"); ax.legend(); ax.set_title("Train vs CV vs student-test Macro F1")
    ax.set_ylim(max(0.5, df["CV Macro F1 (mean)"].min() - .08), 1.0)
    plt.tight_layout(); plt.savefig(fig / "overfitting_check.png", dpi=130); plt.close()

    # confusion matrices
    n = len(recs)
    cols = 4
    rws = int(np.ceil(n / cols))
    fig_, ax = plt.subplots(rws, cols, figsize=(4 * cols, 3.6 * rws))
    ax = np.atleast_2d(ax)
    for a in ax.ravel():
        a.axis("off")
    for a, r in zip(ax.ravel(), recs):
        cm = np.array(r["student_confusion_0.5"])
        a.axis("on"); a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black", fontsize=9)
        a.set_xticks([0, 1], ["not ref", "ref arc"]); a.set_yticks([0, 1], ["not ref", "ref arc"])
        a.set_xlabel("predicted"); a.set_ylabel("actual"); a.set_title(r["model"], fontsize=10)
    plt.suptitle("Student-test candidate-level confusion matrices (threshold 0.5; colour = row-normalised)")
    plt.tight_layout(); plt.savefig(fig / "confusion_matrices.png", dpi=120); plt.close()

    # epoch-wise histories (only models that produce them)
    cp = res / "models" / "curves.json"
    if cp.exists():
        curves = json.loads(cp.read_text())
        if curves:
            fig_, ax = plt.subplots(1, len(curves), figsize=(5.5 * len(curves), 4.2))
            ax = np.atleast_1d(ax)
            for a, (k, v) in zip(ax, curves.items()):
                a.plot(v["train"], label="train", c=COL["base"])
                if "valid" in v:
                    a.plot(v["valid"], label="validation (held-out CV fold 0)", c=COL["alt"])
                a.set_title(f"{k}: log loss per boosting round / epoch"); a.set_xlabel("round / epoch"); a.set_ylabel("log loss"); a.legend()
            plt.tight_layout(); plt.savefig(fig / "training_curves.png", dpi=130); plt.close()


def hpo(res, fig):
    t = res / "hpo" / "trials.csv"
    j = res / "hpo" / "tuned_vs_untuned.json"
    if t.exists():
        tr = pd.read_csv(t)
        fig_, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].plot(tr["number"], tr["value"], "o-", c=COL["base"], label="trial")
        ax[0].plot(tr["number"], tr["value"].cummax(), c=COL["good"], lw=2, label="best so far")
        ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV Top-1 (3-fold, grouped)"); ax[0].set_title("Bayesian optimisation history (TPE)"); ax[0].legend()
        pcols = [c for c in tr.columns if c.startswith("params_")]
        imp = {c[7:]: abs(pd.Series(tr[c]).rank().corr(tr["value"].rank(), method="spearman")) for c in pcols}
        imp = pd.Series(imp).dropna().sort_values()
        ax[1].barh(imp.index, imp.values, color=COL["alt"]); ax[1].set_title("|Spearman| between parameter and objective (descriptive)")
        plt.tight_layout(); plt.savefig(fig / "hpo_history.png", dpi=130); plt.close()
    if j.exists():
        d = json.loads(j.read_text())
        keys = ["cv_macro_f1_mean", "cv_top1_mean", "student_top1", "student_macro_f1"]
        labels = ["CV Macro F1", "CV Top-1", "Student Top-1", "Student Macro F1"]
        x = np.arange(len(keys))
        fig_, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].bar(x - .2, [d["untuned"][k] for k in keys], .4, label="untuned", color=COL["base"])
        ax[0].bar(x + .2, [d["tuned"][k] for k in keys], .4, label="tuned", color=COL["good"])
        ax[0].set_xticks(x, labels); ax[0].legend(); ax[0].set_title(f"{d['model']}: untuned vs Bayesian-tuned")
        lo = min(min(d["untuned"][k], d["tuned"][k]) for k in keys)
        ax[0].set_ylim(max(0.5, lo - .03), 1)
        ax[1].bar(["log loss (untuned)", "log loss (tuned)"], [d["untuned"]["student_logloss"], d["tuned"]["student_logloss"]], color=[COL["base"], COL["good"]])
        ax[1].set_title("Student-test log loss")
        plt.tight_layout(); plt.savefig(fig / "tuned_vs_untuned.png", dpi=130); plt.close()
        pd.DataFrame(d).to_csv(res / "hpo" / "tuned_vs_untuned.csv")


def hmm(res, fig):
    s = res / "hmm" / "student_summary.json"
    pt = res / "hmm" / "student_per_trip.csv"
    if not s.exists():
        return
    d = json.loads(s.read_text())["summary"]
    keys = [("top1_acc", "Top-1 accuracy"), ("route_continuity_h3", "Route continuity (h=3)"),
            ("route_continuity_h1", "Route continuity (h=1)"), ("seq_set_f1", "Route arc-set F1"), ("seq_lcs_ratio", "Route LCS ratio")]
    meth = [("classifier_only", "classifier only", COL["base"]), ("hmm_default_params", "+ HMM (default params)", COL["alt"]),
            ("hmm_tuned_params", "+ HMM (tuned params)", COL["good"])]
    x = np.arange(len(keys))
    fig_, ax = plt.subplots(figsize=(12, 4.8))
    for i, (m, lab, c) in enumerate(meth):
        ax.bar(x + (i - 1) * .27, [d[m][k] for k, _ in keys], .27, label=lab, color=c)
    ax.set_xticks(x, [l for _, l in keys]); ax.legend(); ax.set_ylim(0.6, 1.0)
    ax.set_title("Student-test: classifier-only vs classifier + HMM/Viterbi")
    plt.tight_layout(); plt.savefig(fig / "hmm_vs_classifier.png", dpi=130); plt.close()
    if pt.exists():
        t = pd.read_csv(pt)
        a = t[t.method == "classifier_only"].set_index("trip")["top1_acc"]
        b = t[t.method == "hmm_tuned_params"].set_index("trip")["top1_acc"]
        dlt = (b - a).sort_index()
        fig_, ax = plt.subplots(figsize=(9, 4.2))
        ax.bar(dlt.index.astype(str), dlt.values, color=[COL["good"] if v >= 0 else COL["bad"] for v in dlt.values])
        ax.axhline(0, c="k", lw=.8); ax.set_ylabel("Top-1 change (HMM - classifier)"); ax.set_xlabel("student-test trip")
        ax.set_title("Per-trip effect of HMM decoding (red = HMM did not help)")
        plt.tight_layout(); plt.savefig(fig / "hmm_per_trip_delta.png", dpi=130); plt.close()


def main():
    cfg = load_config()
    res = resolve(cfg, "results")
    fig = res / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    comparison(res, fig)
    hpo(res, fig)
    hmm(res, fig)
    print("figures ->", fig)


if __name__ == "__main__":
    main()
