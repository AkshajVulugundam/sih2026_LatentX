"""
SIH26170 - Stage 2 (REPLACES the old rule-based risk engine): trained risk classifier

ARCHITECTURE (per SIH26170 brief): anomaly detector -> classifier.
  Stage 1: batch-relative anomaly detector (anomaly_detector.py, rule-based
  method). Its output - the max robust z-score, whether it trips the flag,
  and which parameter drives it - is CHAINED explicitly into Stage 2 as
  named input features (anomaly_maxZ, anomaly_flag, anomaly_driver_*), not
  just implicitly re-derived from overlapping feature engineering. An
  ablation (below) checks whether chaining these actually helps before
  keeping them.
  Stage 2 (this script): a genuinely TRAINED, supervised classifier that
  takes early burn-in readings + Stage 1's anomaly features and predicts a
  risk level: NORMAL / MONITOR / HIGH_RISK.

WHY THE OLD risk_engine.py IS SUPERSEDED
  risk_engine.py combined a regression forecast with hand-tuned thresholds -
  it was a heuristic, not a trained model. This script instead:
    1. builds a ground-truth risk label from each component's ACTUAL 0-168h
       outcome (never used as a model input - only as the training target),
    2. trains real classifiers to predict that label from EARLY data only.

GROUND-TRUTH LABEL (built once, from full 0-168h history, used only as target)
  HIGH_RISK  if a component ever crosses an absolute limit, OR ever reaches
             >=85% of a limit at any measured time.
  MONITOR    if not HIGH_RISK, but it ever reaches >=70% of a limit, OR it is
             ever a genuine batch-relative outlier (|robust z| > 3.5).
  NORMAL     otherwise.
  This is a simulation-assumption threshold choice, validated below against
  the known defect classes (documented in the report).

TWO CUTOFFS, TWO MODELS (per your requirement to support both)
  24h model: sees only 0h + 24h readings.
  96h model: sees 0h + 24h + 96h readings (more accurate, later warning).
  Neither ever sees 168h data as an input feature - only the label uses it.

MODEL CHOICE
  For each cutoff, several classical classifiers are compared by 5-fold
  stratified cross-validation: Logistic Regression, Decision Tree (several
  depths), Random Forest, Gradient Boosting. The deployed/exported model is
  a depth-limited Decision Tree - it is exportable as plain nested if/else
  logic (see export_tree_js.py), which is what makes the live browser tool
  possible without a server, and it is directly explainable ("the path taken
  through the tree" IS the explanation). Its accuracy is reported next to
  the stronger ensemble models so the trade-off is visible, not hidden.

Outputs:
  classifier_report.txt        - full comparison + validation
  classifier_predictions.csv   - per component: features, true label, predictions from all models, both cutoffs
  tree_rules_24h.txt / tree_rules_96h.txt - the deployed decision tree, as readable rules
  batch_stats.json             - per-batch/param/time median+MAD, needed by the live tool to compute z features for a newly entered component
"""
import pandas as pd, numpy as np, io, json
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Stage 1, imported for real - not re-derived. anomaly_detector.build_features()
# and anomaly_detector.rule_flag() are the EXACT functions anomaly_detector.py
# itself uses; running python3 anomaly_detector.py separately produces
# anomaly_scores.csv from this same code.
import anomaly_detector as stage1

pd.set_option('display.width', 200)
df = pd.read_csv('sih_synthetic_burnin_wide.csv')
P = ['Iddq_mA', 'Leakage_uA', 'Propagation_Delay_ns']
LIM = {'Iddq_mA': 5.0, 'Leakage_uA': 200.0, 'Propagation_Delay_ns': 15.0}
T = [0, 24, 96, 168]
LEVELS = ['NORMAL', 'MONITOR', 'HIGH_RISK']
SEED = 26170

buf = io.StringIO()
def pr(*a):
    print(*a); print(*a, file=buf)
def h(s): pr('\n' + '=' * 78 + '\n' + s + '\n' + '=' * 78)

def robust_z(g):
    return stage1.robust_z(g)

# ---------------------------------------------------------------------------
# GROUND-TRUTH LABEL (built from full history - training target only)
# ---------------------------------------------------------------------------
pct_over_time = pd.DataFrame({f'{p}_{t}': df[f'{p}_{t}h'] / LIM[p] * 100 for p in P for t in T})
max_pct = pct_over_time.max(axis=1)
ever_crosses = (pct_over_time > 100).any(axis=1)
zmax_all = pd.DataFrame({f'{p}_{t}': df.groupby('Batch_ID')[f'{p}_{t}h'].transform(robust_z).abs() for p in P for t in T})
ever_anomalous = (zmax_all > 3.5).any(axis=1)

df['TRUE_RISK'] = np.select(
    [ever_crosses | (max_pct >= 85), (max_pct >= 70) | ever_anomalous],
    ['HIGH_RISK', 'MONITOR'], default='NORMAL')

h('GROUND-TRUTH LABEL CONSTRUCTION (validation)')
pr('HIGH_RISK: ever crosses a limit, or ever reaches >=85% of a limit (0-168h)')
pr('MONITOR:   ever reaches >=70% of a limit, or ever a true batch outlier (|z|>3.5)')
pr('NORMAL:    neither')
pr('\nLabel distribution:'); pr(df.TRUE_RISK.value_counts().reindex(LEVELS).to_string())
pr('\nLabel vs true defect subtype (sanity check - labels should track defect severity):')
pr(pd.crosstab(df.Defect_Subtype, df.TRUE_RISK).reindex(columns=LEVELS, fill_value=0).to_string())
fp = (df[df.Defect_Type == 'NORMAL'].TRUE_RISK != 'NORMAL').mean() * 100
pr(f'\nFalse-positive rate on true-NORMAL components: {fp:.2f}%')

# ---------------------------------------------------------------------------
# Feature builders (NEVER use 168h data)
# ---------------------------------------------------------------------------
def build_features(cutoff):
    times = [t for t in (0, 24, 96) if t <= cutoff]

    # ---- STAGE 1: run the ACTUAL anomaly detector code -------------------
    # X_stage1 has the batch-relative z-scores and delta/delta-z features,
    # computed by anomaly_detector.py itself (not re-derived here).
    X_stage1 = stage1.build_features(cutoff)
    max_z, driver = stage1.rule_flag(X_stage1)   # anomaly_detector.py's own rule

    X = X_stage1.copy()

    # ---- STAGE 2 addition: raw measurement values -------------------------
    # anomaly_detector.py's feature set is z-scores/deltas only (that's all
    # a batch-relative anomaly rule needs). The classifier additionally
    # needs the raw readings themselves, to reason about absolute proximity
    # to a limit - so these are the only features NOT coming from Stage 1.
    for p in P:
        for t in times:
            X[f'{p}_{t}h'] = df[f'{p}_{t}h']

    # ---- CHAIN STAGE 1's literal output into Stage 2 as input features ---
    X['anomaly_maxZ'] = max_z
    X['anomaly_flag'] = (max_z > stage1.Z_THRESHOLD).astype(int)
    for p in P:
        X[f'anomaly_driver_{p}'] = driver.str.startswith(p).astype(int)
    return X, [c for c in X_stage1.columns if c.endswith('_z')]

y = df['TRUE_RISK']
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced'),
    'DecisionTree_d3': DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=SEED),
    'DecisionTree_d5': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=SEED),
    'RandomForest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=SEED),
    'GradientBoosting': GradientBoostingClassifier(random_state=SEED),
}
DEPLOY_MODEL_NAME = 'DecisionTree_d5'  # chosen for portability/explainability, see report

out_cols = {'Component_ID': df.Component_ID, 'Batch_ID': df.Batch_ID,
            'Defect_Type': df.Defect_Type, 'Defect_Subtype': df.Defect_Subtype,
            'TRUE_RISK': df.TRUE_RISK}

fitted = {}
feats = {}
deploy_cols_by_cutoff = {}
for cutoff in (24, 96):
    X, base_z_cols = build_features(cutoff)
    feats[cutoff] = X
    base_cols = [c for c in X.columns if not c.startswith('anomaly_')]
    h(f'CUTOFF = {cutoff}h  ({X.shape[1]} features: raw + Stage-1 batch-z/deltas + chained anomaly score, from times <= {cutoff}h)')

    pr(f"\n--- ABLATION: does chaining Stage 1's anomaly score into the classifier actually help? ---")
    abl = {}
    for label, cols in [('without', base_cols), ('with', list(X.columns))]:
        p_ = cross_val_predict(DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=SEED), X[cols], y, cv=cv)
        abl[label] = f1_score(y, p_, average='macro')
        pr(f'{"WITHOUT" if label=="without" else "WITH   "} chained anomaly features: '
           f'accuracy={(p_==y).mean():.3f}  macro_F1={abl[label]:.3f}')

    use_chained = abl['with'] >= abl['without']
    deploy_cols = list(X.columns) if use_chained else base_cols
    deploy_cols_by_cutoff[cutoff] = deploy_cols
    pr(f'-> Decision: deploy {"WITH" if use_chained else "WITHOUT"} chained features at {cutoff}h '
       f'(higher macro-F1 wins; {abl["with"]:.3f} vs {abl["without"]:.3f}).')

    Xd = X[deploy_cols]
    rows = []
    for name, model in models.items():
        pred = cross_val_predict(model, Xd, y, cv=cv)
        macro_f1 = f1_score(y, pred, average='macro')
        by_class_recall = {}
        cm = confusion_matrix(y, pred, labels=LEVELS)
        for i, lvl in enumerate(LEVELS):
            by_class_recall[lvl] = cm[i, i] / cm[i].sum() if cm[i].sum() else float('nan')
        rows.append([name, round((pred == y).mean(), 3), round(macro_f1, 3)] +
                     [round(by_class_recall[l], 3) for l in LEVELS])
        if cutoff == 24:
            out_cols[f'pred24_{name}'] = pred
        else:
            out_cols[f'pred96_{name}'] = pred
    tbl = pd.DataFrame(rows, columns=['model', 'accuracy', 'macro_F1'] + [f'recall_{l}' for l in LEVELS])
    pr(f'\n(Table below uses the deployed feature set: {"with" if use_chained else "without"} chaining)')
    pr(tbl.to_string(index=False))

    # confusion matrix for the deployed model
    deploy_pred = cross_val_predict(models[DEPLOY_MODEL_NAME], Xd, y, cv=cv)
    pr(f'\nConfusion matrix, {DEPLOY_MODEL_NAME} (rows=true, cols=predicted):')
    pr(pd.DataFrame(confusion_matrix(y, deploy_pred, labels=LEVELS), index=LEVELS, columns=LEVELS).to_string())
    pr(f'\nPer-subtype recall, {DEPLOY_MODEL_NAME}:')
    rows = []
    for st in df.Defect_Subtype.unique():
        m = (df.Defect_Subtype == st).values
        rows.append([st, int(m.sum()), round((deploy_pred[m] == y.values[m]).mean(), 3)])
    pr(pd.DataFrame(rows, columns=['subtype', 'n', 'accuracy']).sort_values('subtype').to_string(index=False))

    # fit the deployed model on ALL data for export/deployment
    final_tree = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=SEED)
    final_tree.fit(Xd, y)
    fitted[cutoff] = final_tree
    rules = export_text(final_tree, feature_names=list(Xd.columns))
    open(f'tree_rules_{cutoff}h.txt', 'w').write(rules)

h('DEPLOYMENT CHOICE')
pr(f'Deployed model: {DEPLOY_MODEL_NAME} (max_depth=5), one per cutoff.')
pr('Reason: within a few points of RandomForest/GradientBoosting on macro-F1')
pr('(see tables above), but exports as plain nested if/else logic - so the')
pr('live tool runs a real trained model in the browser with no server, and')
pr("every prediction comes with the exact decision path as its explanation.")
pr('RandomForest/GradientBoosting remain available in this script as the')
pr('higher-accuracy reference if a server-side deployment is ever preferred.')
pr(f'\nWhether Stage 1\'s anomaly score is chained in is decided per cutoff by')
pr(f'the ablation above, not assumed: 24h deploys '
   f'{"WITH" if deploy_cols_by_cutoff[24] != [c for c in feats[24].columns if not c.startswith("anomaly_")] else "WITHOUT"} '
   f'chaining, 96h deploys '
   f'{"WITH" if deploy_cols_by_cutoff[96] != [c for c in feats[96].columns if not c.startswith("anomaly_")] else "WITHOUT"} '
   f'chaining.')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
pd.DataFrame(out_cols).to_csv('classifier_predictions.csv', index=False)
open('classifier_report.txt', 'w').write(buf.getvalue())

# batch stats needed by the live tool to compute z-features for a NEW component.
# Must exactly match how build_features() computed them during training:
# level z uses the batch's median/MAD of the RAW READING at that time; delta z
# uses the batch's median/MAD of the DELTA itself (not the level) - these are
# different distributions, so both are saved separately.
batch_stats = {}
for b, g in df.groupby('Batch_ID'):
    batch_stats[b] = {'levels': {}, 'deltas': {}}
    for p in P:
        batch_stats[b]['levels'][p] = {}
        for t in (0, 24, 96):
            col = g[f'{p}_{t}h']
            med = float(col.median())
            mad = float((col - med).abs().median() * 1.4826)
            batch_stats[b]['levels'][p][str(t)] = {'median': round(med, 4), 'mad_sigma': round(max(mad, 1e-6), 4)}
        batch_stats[b]['deltas'][p] = {}
        for a, bb in [(0, 24), (24, 96)]:
            d = g[f'{p}_{bb}h'] - g[f'{p}_{a}h']
            med = float(d.median())
            mad = float((d - med).abs().median() * 1.4826)
            batch_stats[b]['deltas'][p][f'{a}_{bb}'] = {'median': round(med, 4), 'mad_sigma': round(max(mad, 1e-6), 4)}
json.dump(batch_stats, open('batch_stats.json', 'w'))

print('\nSaved classifier_report.txt, classifier_predictions.csv, tree_rules_24h.txt,')
print('tree_rules_96h.txt, batch_stats.json')

# ---------------------------------------------------------------------------
# Export the deployed decision trees as literal JS functions, so the live
# browser tool runs the ACTUAL trained model (not a re-implementation).
# ---------------------------------------------------------------------------
def tree_to_js(clf, feature_names, fn_name):
    t = clf.tree_
    classes = list(clf.classes_)
    lines = [f'function {fn_name}(f) {{', '  const path = [];']

    def rec(node, indent):
        pad = '  ' * indent
        if t.feature[node] == -2:  # leaf
            counts = t.value[node][0]
            cls = classes[int(np.argmax(counts))]
            conf = counts.max() / counts.sum()
            lines.append(f'{pad}return {{level: "{cls}", confidence: {conf:.3f}, path: path.slice()}};')
            return
        feat = feature_names[t.feature[node]]
        thr = t.threshold[node]
        lines.append(f'{pad}if (f["{feat}"] <= {thr:.5f}) {{')
        lines.append(f'{pad}  path.push("{feat} \\u2264 {thr:.3f}");')
        rec(t.children_left[node], indent + 1)
        lines.append(f'{pad}}} else {{')
        lines.append(f'{pad}  path.push("{feat} > {thr:.3f}");')
        rec(t.children_right[node], indent + 1)
        lines.append(f'{pad}}}')

    rec(0, 1)
    lines.append('}')
    return '\n'.join(lines)

js24 = tree_to_js(fitted[24], deploy_cols_by_cutoff[24], 'classify24h')
js96 = tree_to_js(fitted[96], deploy_cols_by_cutoff[96], 'classify96h')
open('tree_24h.js', 'w').write(js24)
open('tree_96h.js', 'w').write(js96)
print('Saved tree_24h.js, tree_96h.js (live, runnable exported models)')
