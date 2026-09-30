#!/usr/bin/env python3
"""
SIH26170 — Export the trained Stage-2 + Stage-3 models for the standalone
ReliefOS browser dashboard.

Run in the SAME folder containing:
  anomaly_detector.py
  classifier.py
  stage3_drift_predictor.py
  sih_synthetic_burnin_wide.csv

This script imports stage3_drift_predictor.py, so it uses the same Stage-1/Stage-2/Stage-3
training pipeline already used for the reported metrics.

Outputs:
  reliefos_live_models.js
  reliefos_live_model_export_report.txt

The JS bundle contains:
  - Stage-2 decision trees with leaf class probabilities
  - deployed Stage-3 regressors for each cutoff/target
  - exact feature names required by each regressor

No 168h measurement is used as a live input feature.
"""

from __future__ import annotations

import sys
import io
import json
from pathlib import Path
import numpy as np


REPORT = "reliefos_live_model_export_report.txt"
OUT_JS = "reliefos_live_models.js"


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def tree_reg_payload(tree):
    t = tree.tree_
    return {
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": [float(x) for x in t.threshold.tolist()],
        "value": [float(x[0, 0]) for x in t.value],
    }


def tree_clf_payload(tree):
    t = tree.tree_
    classes = [str(c) for c in tree.classes_]
    probs = []
    for node in range(t.node_count):
        counts = np.asarray(t.value[node][0], dtype=float)
        total = counts.sum()
        probs.append(
            (counts / total if total > 0 else np.zeros_like(counts)).tolist()
        )
    return {
        "classes": classes,
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": [float(x) for x in t.threshold.tolist()],
        "probs": probs,
    }


def js_string(obj):
    return json.dumps(obj, separators=(",", ":"))


def main():
    print("=" * 78)
    print("SIH26170 — LIVE MODEL EXPORT")
    print("=" * 78)
    print("Importing the existing Stage 3 pipeline...")
    print("This intentionally retrains using the same scripts/dataset.")

    import stage3_drift_predictor as s3
    import classifier as clf

    payload = {
        "stage2": {},
        "stage3": {},
        "metadata": {
            "risk_levels": ["NORMAL", "MONITOR", "HIGH_RISK"],
            "targets": list(clf.P),
            "limits": {k: float(v) for k, v in clf.LIM.items()},
            "cutoffs": [24, 96],
        },
    }

    print("\n" + "=" * 78)
    print("STAGE 2 TREE EXPORT")
    print("=" * 78)

    for cutoff in (24, 96):
        tree = clf.fitted[cutoff]
        payload["stage2"][str(cutoff)] = {
            "feature_names": list(clf.feats[cutoff].columns),
            "deployed_feature_names": list(clf.deploy_cols_by_cutoff[cutoff]),
            "tree": tree_clf_payload(tree),
        }
        print(
            f"{cutoff}h: exported DecisionTree_d5 with "
            f"{len(clf.deploy_cols_by_cutoff[cutoff])} deployed features"
        )

    print("\n" + "=" * 78)
    print("STAGE 3 REGRESSOR EXPORT")
    print("=" * 78)

    for cutoff in (24, 96):
        payload["stage3"][str(cutoff)] = {}
        for p in clf.P:
            model, best_name, deploy_cols = s3.fitted_reg[cutoff][p]

            model_payload = {
                "model_type": best_name,
                "feature_names": list(deploy_cols),
            }

            if best_name == "RandomForest":
                model_payload["n_trees"] = len(model.estimators_)
                model_payload["trees"] = [
                    tree_reg_payload(t) for t in model.estimators_
                ]

            elif best_name == "GradientBoosting":
                model_payload["learning_rate"] = float(model.learning_rate)
                model_payload["init_value"] = float(model.init_.constant_.ravel()[0])
                model_payload["trees"] = [
                    tree_reg_payload(model.estimators_[i, 0])
                    for i in range(model.estimators_.shape[0])
                ]

            elif best_name == "Ridge":
                model_payload["coef"] = [float(x) for x in model.coef_.ravel()]
                model_payload["intercept"] = float(np.asarray(model.intercept_).ravel()[0])

            else:
                raise ValueError(
                    f"Unsupported deployed model type for {cutoff}h/{p}: {best_name}"
                )

            payload["stage3"][str(cutoff)][p] = model_payload

            print(
                f"{cutoff}h / {p}: {best_name} | "
                f"{len(deploy_cols)} features"
            )

    print("\n" + "=" * 78)
    print("WRITING JAVASCRIPT BUNDLE")
    print("=" * 78)

    bundle_json = js_string(payload)

    runtime = r"""
/*
 * ReliefOS live model bundle.
 * Generated from the actual SIH26170 Stage-2 and Stage-3 sklearn models.
 * Do not edit the model payload manually.
 */
window.RELIEFOS_LIVE_MODELS = __PAYLOAD__;
window.RELIEFOS_STAGE3_READY = true;

function reliefosTreeClassify(treePayload, featureVector) {
  let node = 0;
  const path = [];

  while (treePayload.children_left[node] !== -1) {
    const fi = treePayload.feature[node];
    const name = treePayload.feature_names[fi];
    const threshold = treePayload.threshold[node];
    const value = Number(featureVector[name]);

    if (value <= threshold) {
      path.push(name + " ≤ " + threshold.toFixed(3));
      node = treePayload.children_left[node];
    } else {
      path.push(name + " > " + threshold.toFixed(3));
      node = treePayload.children_right[node];
    }
  }

  const probs = treePayload.probs[node];
  let best = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[best]) best = i;
  }

  const out = {};
  treePayload.classes.forEach((c, i) => out[c] = probs[i]);

  return {
    level: treePayload.classes[best],
    confidence: probs[best],
    proba: out,
    path: path
  };
}

function reliefosClassify(cutoff, featureVector) {
  const m = window.RELIEFOS_LIVE_MODELS.stage2[String(cutoff)];
  return reliefosTreeClassify(
    Object.assign({feature_names: m.deployed_feature_names}, m.tree),
    featureVector
  );
}

function reliefosTreeRegress(tree, featureVector) {
  let node = 0;
  while (tree.children_left[node] !== -1) {
    const fi = tree.feature[node];
    const threshold = tree.threshold[node];
    const name = featureVector.__feature_names[fi];
    const value = Number(featureVector[name]);
    node = value <= threshold ? tree.children_left[node] : tree.children_right[node];
  }
  return tree.value[node];
}

function reliefosRegress(model, featureVector) {
  const names = model.feature_names;
  const f = Object.assign({}, featureVector, {__feature_names: names});

  if (model.model_type === "RandomForest") {
    let sum = 0;
    for (const tree of model.trees) sum += reliefosTreeRegress(tree, f);
    return sum / model.trees.length;
  }

  if (model.model_type === "GradientBoosting") {
    let value = model.init_value;
    for (const tree of model.trees) {
      value += model.learning_rate * reliefosTreeRegress(tree, f);
    }
    return value;
  }

  if (model.model_type === "Ridge") {
    let value = model.intercept;
    for (let i = 0; i < names.length; i++) {
      value += model.coef[i] * Number(featureVector[names[i]]);
    }
    return value;
  }

  throw new Error("Unsupported Stage-3 model type: " + model.model_type);
}

/*
 * This function expects the dashboard to have already built the exact Stage-2
 * deployed feature vector for the chosen cutoff. It obtains the live Stage-2
 * class probabilities from the exported trained tree, adds them, then runs
 * the exact deployed Stage-3 regressor feature subset for each target.
 */
window.reliefosPredict168h = function(cutoff, baseFeatures) {
  const c = String(cutoff);
  const stage2 = reliefosClassify(cutoff, baseFeatures);

  const enriched = Object.assign({}, baseFeatures);
  Object.keys(stage2.proba).forEach(k => {
    enriched["stage2_proba_" + k] = stage2.proba[k];
  });

  const models = window.RELIEFOS_LIVE_MODELS.stage3[c];
  const out = {};

  Object.keys(models).forEach(target => {
    const m = models[target];
    out[target] = reliefosRegress(m, enriched);
  });

  return out;
};
""".replace("__PAYLOAD__", bundle_json)

    Path(OUT_JS).write_text(runtime, encoding="utf-8")

    print(f"Created: {OUT_JS}")
    print(f"Stage-3 models exported for cutoffs: 24h, 96h")
    print("The dashboard can now run the trained Stage-3 regressors in-browser.")

    print("\n" + "=" * 78)
    print("IMPORTANT")
    print("=" * 78)
    print("The export is generated from the same classifier.py/stage3_drift_predictor.py pipeline.")
    print("Keep this JS file beside reliefos_console_live_integrated.html.")
    print("Do not use 168h measurements or defect labels as live inputs.")


if __name__ == "__main__":
    report = Path(REPORT)
    old_out, old_err = sys.stdout, sys.stderr
    fh = report.open("w", encoding="utf-8")
    try:
        sys.stdout = Tee(sys.stdout, fh)
        sys.stderr = Tee(sys.stderr, fh)
        main()
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        raise
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        fh.close()
