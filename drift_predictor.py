"""
SIH26170 - Stage 2: 168h drift predictor

Predicts each component's 168h Iddq / Leakage / Propagation Delay using only
0h/24h/96h measurements + derived batch-relative/delta features (same feature
family as the Stage 1 anomaly detector, so the two stages share a vocabulary).

Compares three classical regressors per the roadmap:
  - Ridge (linear baseline)
  - Random Forest Regressor
  - Gradient Boosting Regressor

Evaluated with 5-fold cross-validation (never trains and tests on the same
rows) and reported as MAE overall AND per defect class/subtype, because the
SIH problem cares most about getting the hard cases (drift/latent/sudden)
right, not the easy 87% that's NORMAL.

Defect_Type / Defect_Subtype are used ONLY for evaluation grouping, never as
model input. 168h_Absolute_Fail is used only to check predicted vs actual
threshold crossings, never as an input.

Outputs:
  drift_predictions.csv     - per-component actual vs predicted 168h values
  drift_predictor_report.txt
"""
import pandas as pd, numpy as np, io
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

pd.set_option('display.width', 200)
df = pd.read_csv('sih_synthetic_burnin_wide.csv')
P = ['Iddq_mA', 'Leakage_uA', 'Propagation_Delay_ns']
LIM = {'Iddq_mA': 5.0, 'Leakage_uA': 200.0, 'Propagation_Delay_ns': 15.0}
CL = ['NORMAL', 'BATCH_ANOMALY', 'GRADUAL_DRIFT', 'SUDDEN_ANOMALY', 'LATENT_DEFECT']

buf = io.StringIO()
def pr(*a):
    print(*a); print(*a, file=buf)
def h(s): pr('\n' + '=' * 78 + '\n' + s + '\n' + '=' * 78)

def robust_z(g):
    med = g.median(); mad = (g - med).abs().median() * 1.4826
    mad = mad if mad > 1e-9 else 1e-9
    return (g - med) / mad

# ---------------------------------------------------------------------------
# Features: raw 0h/24h/96h values, batch-relative z at each, and deltas +
# their batch-relative z. Same family as Stage 1, so the risk engine can
# later reuse both stages' features together.
# ---------------------------------------------------------------------------
times = [0, 24, 96]
X = pd.DataFrame(index=df.index)
for p in P:
    for t in times:
        col = f'{p}_{t}h'
        X[col] = df[col]
        X[f'{col}_z'] = df.groupby('Batch_ID')[col].transform(robust_z)
    for a, b in zip(times[:-1], times[1:]):
        dname = f'{p}_d{a}_{b}'
        X[dname] = df[f'{p}_{b}h'] - df[f'{p}_{a}h']
        tmp = pd.DataFrame({'Batch_ID': df['Batch_ID'], 'v': X[dname]})
        X[f'{dname}_z'] = tmp.groupby('Batch_ID')['v'].transform(robust_z)

h('STAGE 2 - 168h DRIFT PREDICTOR: MODEL COMPARISON (5-fold CV)')
pr(f'Inputs: 0h/24h/96h raw values + batch-relative z + deltas ({X.shape[1]} features)')
pr('Never uses: 168h_* columns, Defect_Type, Defect_Subtype')

models = {
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(n_estimators=300, random_state=26170),
    'GradientBoosting': GradientBoostingRegressor(random_state=26170),
}
cv = KFold(5, shuffle=True, random_state=26170)

preds = {}  # preds[(param, model_name)] = array
out_cols = {'Component_ID': df.Component_ID, 'Batch_ID': df.Batch_ID,
            'Defect_Type': df.Defect_Type, 'Defect_Subtype': df.Defect_Subtype}

for p in P:
    y = df[f'{p}_168h']
    out_cols[f'{p}_168h_actual'] = y
    h(f'TARGET: {p}_168h  (limit = {LIM[p]}, target std = {y.std():.3f})')
    rows = []
    for name, model in models.items():
        yhat = cross_val_predict(model, X, y, cv=cv)
        preds[(p, name)] = yhat
        out_cols[f'{p}_168h_pred_{name}'] = np.round(yhat, 3)
        mae = np.abs(y - yhat).mean()
        r2 = r2_score(y, yhat)
        by_class = pd.Series(np.abs(y - yhat)).groupby(df.Defect_Type).mean()
        rows.append([name, round(mae, 3), round(r2, 3)] +
                     [round(by_class.get(c, np.nan), 3) for c in CL])
    tbl = pd.DataFrame(rows, columns=['model', 'MAE_overall', 'R2'] + [f'MAE_{c}' for c in CL])
    pr(tbl.to_string(index=False))

# ---------------------------------------------------------------------------
# Pick best model per parameter (lowest overall MAE) for the summary + CSV
# ---------------------------------------------------------------------------
h('BEST MODEL PER PARAMETER (by overall MAE) - used for the risk-engine handoff')
best = {}
for p in P:
    y = df[f'{p}_168h']
    maes = {name: np.abs(y - preds[(p, name)]).mean() for name in models}
    best_name = min(maes, key=maes.get)
    best[p] = best_name
    out_cols[f'{p}_168h_pred_BEST'] = np.round(preds[(p, best_name)], 3)
    pr(f'{p}: {best_name} (MAE {maes[best_name]:.3f} vs Ridge baseline {maes["Ridge"]:.3f})')

# ---------------------------------------------------------------------------
# Threshold-crossing check: using the best model, how well do we predict
# which components will exceed the absolute limit at 168h?
# ---------------------------------------------------------------------------
h('PREDICTED VS ACTUAL THRESHOLD CROSSINGS (best model per parameter)')
for p in P:
    y = df[f'{p}_168h']
    yhat = preds[(p, best[p])]
    actual_fail = y > LIM[p]
    pred_fail = yhat > LIM[p]
    tp = (actual_fail & pred_fail).sum()
    fn = (actual_fail & ~pred_fail).sum()
    fp = (~actual_fail & pred_fail).sum()
    n_fail = actual_fail.sum()
    pr(f'{p}: {n_fail} actual failures | caught={tp} missed={fn} false-alarms={fp}')
    if fn > 0:
        missed = df.loc[actual_fail & ~pred_fail, ['Component_ID', 'Defect_Type']]
        pr(f'  Missed: {missed.to_dict("records")}')

# ---------------------------------------------------------------------------
# "Close to limit" early-warning check for LATENT_DEFECT and mild classes
# that never actually cross 168h but should show elevated predicted risk.
# ---------------------------------------------------------------------------
h('EARLY-WARNING CHECK: predicted % of limit for classes that never cross it')
for p in P:
    yhat = preds[(p, best[p])]
    pct_of_limit = yhat / LIM[p] * 100
    rows = []
    for cl in CL:
        m = (df.Defect_Type == cl).values
        rows.append([cl, round(pct_of_limit[m].mean(), 1), round(pct_of_limit[m].max(), 1)])
    pr(f'\n{p} - predicted %-of-limit at 168h:')
    pr(pd.DataFrame(rows, columns=['class', 'mean_pred_%lim', 'max_pred_%lim']).to_string(index=False))

pr('\nInterpretation: LATENT_DEFECT and SUDDEN_AFTER_96H should show clearly')
pr('elevated predicted %-of-limit vs NORMAL, even though neither crosses the')
pr('limit itself. That gap is what the risk engine (Stage 3) will use to')
pr('issue a MONITOR/HIGH_RISK call before an absolute failure ever happens.')

out_df = pd.DataFrame(out_cols)
out_df.to_csv('drift_predictions.csv', index=False)
open('drift_predictor_report.txt', 'w').write(buf.getvalue())
print('\nSaved drift_predictions.csv and drift_predictor_report.txt')
