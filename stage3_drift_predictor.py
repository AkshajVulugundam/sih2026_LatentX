"""
SIH26170 - Stage 3: 168h drift prediction

FULL PIPELINE NOW: Stage 1 (anomaly detector) -> Stage 2 (risk classifier)
                                                       -> Stage 3 (this script)

Stage 3 predicts each component's ACTUAL 168h Iddq / Leakage / Delay value
(a regression, not a category) - useful for "how close to the limit will it
get" in a way Stage 2's NORMAL/MONITOR/HIGH_RISK label alone doesn't capture.

CHAINING (real, not re-derived):
  Input features = Stage 2's exact deployed feature set (which itself
  already chains Stage 1's anomaly score where the ablation found that
  helped) PLUS Stage 2's own trained classifier output: the predicted
  probability of each risk class, read directly from
  classifier.fitted[cutoff].predict_proba() - not recomputed.

Like Stage 2, an ablation checks whether chaining Stage 2's output into
Stage 3 actually helps before keeping it - per target, per cutoff, not
assumed uniformly.

Never uses 168h_* columns or Defect_Type/Defect_Subtype as inputs.

Outputs:
  stage3_report.txt
  stage3_predictions.csv   - per component: actual + predicted 168h values, both cutoffs, best model
"""
import numpy as np, pandas as pd, io
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

import classifier as clf   # Stage 2 (which itself imports Stage 1)

df = clf.df
P = clf.P
LIM = clf.LIM
CL = ['NORMAL', 'BATCH_ANOMALY', 'GRADUAL_DRIFT', 'SUDDEN_ANOMALY', 'LATENT_DEFECT']
SEED = clf.SEED

buf = io.StringIO()
def pr(*a):
    print(*a); print(*a, file=buf)
def h(s): pr('\n' + '=' * 78 + '\n' + s + '\n' + '=' * 78)

# ---------------------------------------------------------------------------
# Features: Stage 2's deployed feature set + Stage 2's ACTUAL predicted
# class probabilities (read from the real fitted tree, not re-derived)
# ---------------------------------------------------------------------------
def build_features(cutoff):
    cols = clf.deploy_cols_by_cutoff[cutoff]
    X = clf.feats[cutoff][cols].copy()
    tree = clf.fitted[cutoff]
    proba = tree.predict_proba(X)
    for i, c in enumerate(tree.classes_):
        X[f'stage2_proba_{c}'] = proba[:, i]
    return X

cv = KFold(5, shuffle=True, random_state=SEED)
models = {
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(n_estimators=300, random_state=SEED),
    'GradientBoosting': GradientBoostingRegressor(random_state=SEED),
}

out_cols = {'Component_ID': df.Component_ID, 'Batch_ID': df.Batch_ID,
            'Defect_Type': df.Defect_Type, 'Defect_Subtype': df.Defect_Subtype}

fitted_reg = {}
deploy_cols_reg = {}
for cutoff in (24, 96):
    X = build_features(cutoff)
    base_cols = [c for c in X.columns if not c.startswith('stage2_proba_')]
    h(f'CUTOFF = {cutoff}h  ({X.shape[1]} features: Stage-2 deployed feature set + chained Stage-2 class probabilities)')
    fitted_reg[cutoff] = {}
    deploy_cols_reg[cutoff] = {}

    for p in P:
        y = df[f'{p}_168h']
        out_cols[f'{p}_168h_actual'] = y
        pr(f'\n--- Target: {p}_168h  (limit={LIM[p]}, std={y.std():.3f}) ---')

        # ablation: does chaining Stage 2's probabilities help THIS target?
        abl = {}
        for label, cols in [('without', base_cols), ('with', list(X.columns))]:
            yhat = cross_val_predict(RandomForestRegressor(n_estimators=300, random_state=SEED), X[cols], y, cv=cv)
            abl[label] = np.abs(y - yhat).mean()
        use_chained = abl['with'] <= abl['without']
        deploy_cols = list(X.columns) if use_chained else base_cols
        deploy_cols_reg[cutoff][p] = deploy_cols
        pr(f'Ablation (RandomForest MAE): without chaining={abl["without"]:.3f}  with chaining={abl["with"]:.3f}  '
           f'-> deploy {"WITH" if use_chained else "WITHOUT"} Stage-2 probabilities')

        Xd = X[deploy_cols]
        rows = []
        for name, model in models.items():
            yhat = cross_val_predict(model, Xd, y, cv=cv)
            mae = np.abs(y - yhat).mean()
            r2 = r2_score(y, yhat)
            by_class = pd.Series(np.abs(y - yhat)).groupby(df.Defect_Type).mean()
            rows.append([name, round(mae, 3), round(r2, 3)] + [round(by_class.get(c, np.nan), 3) for c in CL])
            out_cols[f'{p}_168h_pred_{cutoff}h_{name}'] = np.round(yhat, 3)
        tbl = pd.DataFrame(rows, columns=['model', 'MAE', 'R2'] + [f'MAE_{c}' for c in CL])
        pr(tbl.to_string(index=False))

        best_name = min(models, key=lambda n: np.abs(y - cross_val_predict(models[n], Xd, y, cv=cv)).mean())
        final_model = type(models[best_name])(**models[best_name].get_params())
        final_model.fit(Xd, y)
        fitted_reg[cutoff][p] = (final_model, best_name, deploy_cols)
        out_cols[f'{p}_168h_pred_{cutoff}h_BEST'] = np.round(final_model.predict(Xd), 3)
        pr(f'-> Deployed for {cutoff}h: {best_name}')

# ---------------------------------------------------------------------------
# Threshold-crossing check (best model, 96h - the more informative cutoff)
# ---------------------------------------------------------------------------
h('PREDICTED VS ACTUAL THRESHOLD CROSSINGS (96h, best model per target)')
for p in P:
    y = df[f'{p}_168h']
    yhat = out_cols[f'{p}_168h_pred_96h_BEST']
    actual_fail = y > LIM[p]
    pred_fail = yhat > LIM[p]
    tp = (actual_fail & pred_fail).sum(); fn = (actual_fail & ~pred_fail).sum(); fp = (~actual_fail & pred_fail).sum()
    pr(f'{p}: {actual_fail.sum()} actual failures | caught={tp} missed={fn} false-alarms={fp}')

h('EARLY-WARNING CHECK: predicted %-of-limit at 168h, by true defect class (96h model)')
for p in P:
    pct = out_cols[f'{p}_168h_pred_96h_BEST'] / LIM[p] * 100
    rows = [[cl, round(pct[df.Defect_Type == cl].mean(), 1), round(pct[df.Defect_Type == cl].max(), 1)] for cl in CL]
    pr(f'\n{p}:'); pr(pd.DataFrame(rows, columns=['class', 'mean_pred_%lim', 'max_pred_%lim']).to_string(index=False))

pd.DataFrame(out_cols).to_csv('stage3_predictions.csv', index=False)
open('stage3_report.txt', 'w').write(buf.getvalue())
print('\nSaved stage3_report.txt, stage3_predictions.csv')
