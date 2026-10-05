"""Explain the fixed hybrid model and run controlled, split-preserving ablations."""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "results" / "features" / "hybrid"
OUT = ROOT / "results" / "models" / "hybrid_xgboost"
MODEL_PATH = OUT / "hybrid_xgboost_model.joblib"
SCALER_PATH = OUT / "hybrid_scaler.joblib"
SEED = 42
META = ["image_path", "split", "class_id", "class_name"]
CLASSES = {0: "0_NOR", 1: "1_F&S", 2: "2_SD", 3: "3_MY", 4: "4_AP", 5: "5_BN", 6: "6_UN", 7: "7_IM"}
PARAMS = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
          "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1}


def load_frames():
    frames = {s: pd.read_csv(FEATURE_DIR / f"{s}_hybrid_features.csv") for s in ("train", "val", "test")}
    cols = [c for c in frames["train"] if c not in META]
    saved = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pipeline = joblib.load(OUT / "hybrid_xgboost.joblib")
    trained_columns = pipeline.get("feature_columns") if isinstance(pipeline, dict) else getattr(saved, "feature_names_in_", None)
    if trained_columns is None:
        raise ValueError("Saved final model does not contain its training feature order.")
    if list(trained_columns) != cols:
        raise ValueError("Saved model feature order does not match hybrid feature CSV order.")
    if len(cols) != 1342 or any(list(frames[s][cols].columns) != cols for s in frames):
        raise ValueError("Hybrid feature columns are inconsistent or do not have 1,342 dimensions.")
    if not all(np.isfinite(frames[s][cols].to_numpy(dtype=np.float32)).all() for s in frames):
        raise ValueError("NaN or infinite hybrid values found; stopping without modification.")
    return frames, cols, saved, scaler


def groups(cols):
    return {"Shape/Morphological": [c for c in cols if c.startswith("shape_")],
            "GLCM Texture": [c for c in cols if c.startswith("glcm_")],
            "Colour": [c for c in cols if c.startswith("color_")],
            "EfficientNet-B0": [c for c in cols if c.startswith("deep_feature_")]}


def shap_analysis(frames, cols, model, scaler):
    test = frames["test"]
    subset = test.sample(n=300, random_state=SEED).sort_index()
    x = scaler.transform(subset[cols].to_numpy(dtype=np.float32))
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(x))
    # SHAP multiclass layouts vary: normalize to samples x features x classes.
    if values.ndim == 3 and values.shape[0] == len(CLASSES):
        values = np.moveaxis(values, 0, -1)
    if values.shape[:2] != (len(subset), len(cols)):
        raise RuntimeError(f"Unexpected SHAP value shape: {values.shape}")
    mean_abs = np.abs(values).mean(axis=(0, 2))
    top_idx = np.argsort(mean_abs)[-20:][::-1]
    top = pd.DataFrame({"feature": np.array(cols)[top_idx], "mean_abs_shap": mean_abs[top_idx]})
    top.to_csv(OUT / "shap_top20_features.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 7)); ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#377eb8")
    ax.set_xlabel("Mean |SHAP value| across 300 test explanation samples and 8 outputs"); ax.set_title("Top 20 features by mean absolute SHAP value"); fig.tight_layout(); fig.savefig(OUT / "shap_top20_features.png", dpi=220); plt.close(fig)
    group_rows = []
    for name, feature_cols in groups(cols).items():
        ix = [cols.index(c) for c in feature_cols]; value = float(mean_abs[ix].sum())
        group_rows.append({"feature_group": name, "feature_count": len(ix), "sum_mean_abs_shap": value})
    grp = pd.DataFrame(group_rows); grp["shap_contribution_percent"] = 100 * grp.sum_mean_abs_shap / grp.sum_mean_abs_shap.sum()
    grp.to_csv(OUT / "shap_feature_group_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(grp.feature_group, grp.shap_contribution_percent, color="#4daf4a")
    ax.set_ylabel("SHAP-based feature group contribution (%)"); ax.set_title("SHAP-based feature group contribution"); ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(grp.shap_contribution_percent): ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(OUT / "shap_feature_group_importance.png", dpi=220); plt.close(fig)
    # Readable beeswarm-style plot of the 20 globally strongest dimensions.
    fig, ax = plt.subplots(figsize=(10, 8))
    rng = np.random.default_rng(SEED)
    for row, j in enumerate(top_idx[::-1]):
        v = values[:, j, :].mean(axis=1); y = row + rng.normal(0, .10, len(v))
        ax.scatter(v, y, c=subset.iloc[:, subset.columns.get_loc(cols[j])], s=10, alpha=.6, cmap="coolwarm")
    ax.set_yticks(range(20)); ax.set_yticklabels(np.array(cols)[top_idx[::-1]]); ax.set_xlabel("Mean SHAP value across class outputs"); ax.set_title("SHAP summary (top 20 features; 300 test explanation samples)")
    fig.tight_layout(); fig.savefig(OUT / "shap_summary.png", dpi=220); plt.close(fig)
    class_rows = []
    for class_id, name in CLASSES.items():
        mask = subset.class_id.to_numpy() == class_id
        if not mask.any(): continue
        score = np.abs(values[mask, :, class_id]).mean(axis=0); inds = np.argsort(score)[-10:][::-1]
        class_rows.extend({"class_id": class_id, "class_name": name, "feature": cols[i], "mean_abs_shap": score[i], "explanation_samples": int(mask.sum())} for i in inds)
    pd.DataFrame(class_rows).to_csv(OUT / "shap_classwise_top_features.csv", index=False)
    individual_dir = OUT / "shap_individual"; individual_dir.mkdir(exist_ok=True)
    pred = model.predict(x).astype(int); proba = model.predict_proba(x)
    choices = []
    for condition in [lambda: np.where((subset.class_id.to_numpy() == 0) & (pred == 0))[0], lambda: np.where((subset.class_id.to_numpy() != 0) & (pred == subset.class_id.to_numpy()))[0], lambda: np.where(pred != subset.class_id.to_numpy())[0]]:
        found = condition();
        if len(found): choices.append(int(found[0]))
    records = []
    for n, i in enumerate(choices, 1):
        p = int(pred[i]); contrib = values[i, :, p]; inds = np.argsort(np.abs(contrib))[-10:][::-1]
        rows = [{"feature": cols[j], "shap_value": float(contrib[j])} for j in inds]
        pd.DataFrame(rows).to_csv(individual_dir / f"sample_{n}_top_contributions.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5)); colors = ["#d62728" if r["shap_value"] > 0 else "#1f77b4" for r in rows]
        ax.barh([r["feature"] for r in rows][::-1], [r["shap_value"] for r in rows][::-1], color=colors[::-1]); ax.set_title(f"Sample {n}: contribution to predicted {CLASSES[p]}"); ax.set_xlabel("SHAP value (+ increases predicted-class output)"); fig.tight_layout(); fig.savefig(individual_dir / f"sample_{n}_explanation.png", dpi=220); plt.close(fig)
        records.append({"sample": n, "image_path": subset.iloc[i].image_path, "actual_class": CLASSES[int(subset.iloc[i].class_id)], "predicted_class": CLASSES[p], "prediction_confidence": float(proba[i, p]), "correct": bool(p == subset.iloc[i].class_id)})
    pd.DataFrame(records).to_csv(individual_dir / "selected_samples.csv", index=False)
    return top, grp, len(subset), records


def metrics(model, x, y):
    t = time.perf_counter(); p = model.predict(x); inference = time.perf_counter() - t
    return {"accuracy": accuracy_score(y,p), "macro_precision": precision_score(y,p,average="macro",zero_division=0), "macro_recall": recall_score(y,p,average="macro",zero_division=0), "macro_f1": f1_score(y,p,average="macro",zero_division=0), "weighted_f1": f1_score(y,p,average="weighted",zero_division=0), "inference_seconds": inference}


def ablation(frames, cols):
    gs = groups(cols); configs = [("A. Shape only", gs["Shape/Morphological"]), ("B. GLCM Texture only", gs["GLCM Texture"]), ("C. Colour only", gs["Colour"]), ("D. All handcrafted", gs["Shape/Morphological"] + gs["GLCM Texture"] + gs["Colour"]), ("E. EfficientNet only", gs["EfficientNet-B0"]), ("F. Handcrafted + EfficientNet", cols)]
    rows=[]
    for name, features in configs:
        scaler=StandardScaler(); xtr=scaler.fit_transform(frames["train"][features]); xv=scaler.transform(frames["val"][features]); xt=scaler.transform(frames["test"][features])
        model=XGBClassifier(objective="multi:softprob",num_class=8,eval_metric="mlogloss",tree_method="hist",n_jobs=-1,random_state=SEED,verbosity=0,**PARAMS)
        started=time.perf_counter(); model.fit(xtr,frames["train"].class_id); training=time.perf_counter()-started
        val=metrics(model,xv,frames["val"].class_id); test=metrics(model,xt,frames["test"].class_id)
        rows.append({"configuration":name,"feature_dimension":len(features),"training_seconds":training,"validation_accuracy":val["accuracy"],"validation_macro_f1":val["macro_f1"],**{"test_"+k:v for k,v in test.items()}})
    # G is the frozen final model result; no retraining is performed.
    val=json.loads((OUT/"hybrid_validation_metrics.json").read_text()); test=json.loads((OUT/"hybrid_test_metrics.json").read_text())
    rows.append({"configuration":"G. Full Hybrid model (saved final)","feature_dimension":1342,"training_seconds":json.loads((OUT/"hybrid_reproducibility.json").read_text())["final_fit_seconds"],"validation_accuracy":val["accuracy"],"validation_macro_f1":val["macro_f1"],**{"test_"+k:v for k,v in test.items()}})
    result=pd.DataFrame(rows); result.to_csv(OUT/"ablation_results.csv",index=False)
    for metric, fname, title in [("test_accuracy","ablation_accuracy.png","Test accuracy by feature configuration"),("test_macro_f1","ablation_macro_f1.png","Test Macro F1 by feature configuration"),("validation_macro_f1","ablation_validation_macro_f1.png","Validation Macro F1 by feature configuration")]:
        fig,ax=plt.subplots(figsize=(11,5)); ax.bar(result.configuration,result[metric],color="#984ea3"); ax.set_ylim(0,1.05); ax.set_ylabel(metric.replace("_"," ")); ax.set_title(title); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); fig.savefig(OUT/fname,dpi=220); plt.close(fig)
    return result


def main():
    started=time.perf_counter(); frames, cols, model, scaler = load_frames(); top, grp, subset_size, individual = shap_analysis(frames,cols,model,scaler); results=ablation(frames,cols)
    reproducibility={"random_state":SEED,"shap_subset_size":subset_size,"python":sys.version,"xgboost":xgboost.__version__,"shap":shap.__version__,"numpy":np.__version__,"scikit_learn":sklearn.__version__,"feature_dimension":len(cols),"sample_counts":{k:len(v) for k,v in frames.items()},"total_execution_seconds":time.perf_counter()-started}
    (OUT/"shap_reproducibility.json").write_text(json.dumps(reproducibility,indent=2),encoding="utf-8")
    lines=["EXPLAINABILITY + CONTROLLED ABLATION REPORT","",f"Final frozen model verified: {MODEL_PATH}","SHAP used TreeExplainer on a reproducible 300-sample test explanation subset; it is not a tuning/evaluation split.","", "Top 20 global mean |SHAP| features:",top.to_string(index=False),"", "SHAP-based feature group contribution (distinct from XGBoost gain importance):",grp.to_string(index=False),"", "Ablation methodology: models were fit only on training data with StandardScaler fit on training data; validation metrics were descriptive and test was not used for selection. G reuses the existing frozen final model.","",results.to_string(index=False),"",f"Individual explanations saved for {len(individual)} representative samples.","Limitations: SHAP describes model associations/contributions on the explanation subset; it does not establish causal effects. Deep embedding dimensions are latent features."]
    (OUT/"ablation_report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8"); (OUT/"explainability_ablation_report.txt").write_text("\n".join(lines)+"\n\nReproducibility:\n"+json.dumps(reproducibility,indent=2)+"\n",encoding="utf-8")
    print("EXPLAINABILITY_AND_ABLATION_COMPLETE"); print(results.to_string(index=False)); print(grp.to_string(index=False))

if __name__ == "__main__": main()
