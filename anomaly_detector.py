"""
SIH26170 - Stage 1: Batch-relative anomaly detector (first prototype)

Compares two classical methods on the v2 synthetic dataset:
  A) Robust z-score rule (interpretable, hand-built)
  B) Isolation Forest (classical unsupervised ML)

Both use ONLY features available up to a given cutoff time (24h or 96h) -
never 168h, never Defect_Type/Defect_Subtype (those are evaluation-only).

This module is also imported by classifier.py, which chains this file's
actual build_features()/rule_flag() output into Stage 2 as input features -
Stage 2 does not re-derive the anomaly score itself.

Outputs (only when run directly, python3 anomaly_detector.py):
  anomaly_scores.csv          - one row per component, both methods' scores/flags, at both cutoffs
  anomaly_detector_report.txt - the printed report below
"""
import pandas as pd, numpy as np, io
from sklearn.ensemble import IsolationForest

pd.set_option('display.width', 200)
df = pd.read_csv('sih_synthetic_burnin_wide.csv')
P = ['Iddq_mA', 'Leakage_uA', 'Propagation_Delay_ns']
CL = ['NORMAL', 'BATCH_ANOMALY', 'GRADUAL_DRIFT', 'SUDDEN_ANOMALY', 'LATENT_DEFECT']
Z_THRESHOLD = 3.5  # rule-based flag cutoff

buf = io.StringIO()
def pr(*a):
    print(*a); print(*a, file=buf)
def h(s): pr('\n' + '=' * 78 + '\n' + s + '\n' + '=' * 78)

def robust_z(g):
    med = g.median()
    mad = (g - med).abs().median() * 1.4826
    mad = mad if mad > 1e-9 else 1e-9
    return (g - med) / mad

# ---------------------------------------------------------------------------
# Feature engineering (batch-relative, no leakage from 168h)
# ---------------------------------------------------------------------------
def build_features(cutoff):
    times = [t for t in (0, 24, 96) if t <= cutoff]
    X = pd.DataFrame(index=df.index)
    for p in P:
        for t in times:
            col = f'{p}_{t}h'
            X[f'{p}_{t}h_z'] = df.groupby('Batch_ID')[col].transform(robust_z)
        for a, b in zip(times[:-1], times[1:]):
            dname = f'{p}_d{a}_{b}'
            X[dname] = df[f'{p}_{b}h'] - df[f'{p}_{a}h']
            tmp = pd.DataFrame({'Batch_ID': df['Batch_ID'], 'v': X[dname]})
            X[f'{dname}_z'] = tmp.groupby('Batch_ID')['v'].transform(robust_z)
    return X

# ---------------------------------------------------------------------------
# Method A: robust z-score rule
# ---------------------------------------------------------------------------
def rule_flag(X):
    z_cols = [c for c in X.columns if c.endswith('_z')]
    max_abs_z = X[z_cols].abs().max(axis=1)
    driver = X[z_cols].abs().idxmax(axis=1)
    return max_abs_z, driver

# ---------------------------------------------------------------------------
# Method B: Isolation Forest on the same feature set
# contamination is a simulation assumption: ~15% (a bit under the true 22%
# non-normal rate, since several subtypes are deliberately near-invisible
# early on and shouldn't all be forced into the "anomaly" bucket).
# ---------------------------------------------------------------------------
def iforest_flag(X, contamination=0.15, seed=26170):
    clf = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    clf.fit(X)
    score = -clf.score_samples(X)  # higher = more anomalous
    pred = clf.predict(X) == -1     # True = flagged
    return score, pred

def main():
    results = {}
    h('BATCH-RELATIVE ANOMALY DETECTOR - COMPARISON OF TWO METHODS')
    pr(f'Rule-based flag threshold: |robust z| > {Z_THRESHOLD}')
    pr('Isolation Forest contamination assumption: 0.15 (simulation choice, not a SIH spec)')

    out_cols = {'Component_ID': df.Component_ID, 'Batch_ID': df.Batch_ID,
                'Defect_Type': df.Defect_Type, 'Defect_Subtype': df.Defect_Subtype}

    for cutoff in (24, 96):
        X = build_features(cutoff)
        max_z, driver = rule_flag(X)
        rule_pred = max_z > Z_THRESHOLD
        if_score, if_pred = iforest_flag(X)

        out_cols[f'rule_maxZ_{cutoff}h'] = max_z.round(2)
        out_cols[f'rule_driver_{cutoff}h'] = driver
        out_cols[f'rule_flag_{cutoff}h'] = rule_pred
        out_cols[f'iforest_score_{cutoff}h'] = pd.Series(if_score).round(3)
        out_cols[f'iforest_flag_{cutoff}h'] = if_pred

        h(f'CUTOFF = {cutoff}h (inputs: 0h' + (', 24h' if cutoff == 96 else '') + f', {cutoff}h + deltas)')
        rows = []
        for cl in CL:
            m = df.Defect_Type == cl
            rows.append([cl, int(m.sum()),
                         round(100 * rule_pred[m].mean(), 1),
                         round(100 * if_pred[m].mean(), 1)])
        tbl = pd.DataFrame(rows, columns=['class', 'n', 'rule_flag_rate_%', 'iforest_flag_rate_%'])
        pr(tbl.to_string(index=False))
        pr(f'\n(Row "NORMAL" = false-alarm rate for each method. All other rows = recall.)')

        # per-subtype breakdown
        pr('\nPer-subtype recall:')
        rows = []
        for st in df.Defect_Subtype.unique():
            m = df.Defect_Subtype == st
            if df.loc[m, 'Defect_Type'].iloc[0] == 'NORMAL':
                continue
            rows.append([st, int(m.sum()), round(100 * rule_pred[m].mean(), 1), round(100 * if_pred[m].mean(), 1)])
        pr(pd.DataFrame(rows, columns=['subtype', 'n', 'rule_%', 'iforest_%']).sort_values('subtype').to_string(index=False))

        results[cutoff] = dict(X=X, max_z=max_z, driver=driver, rule_pred=rule_pred, if_pred=if_pred, if_score=if_score)

    # -----------------------------------------------------------------------
    # Agreement between the two methods
    # -----------------------------------------------------------------------
    h('AGREEMENT BETWEEN METHODS (96h cutoff)')
    r96 = results[96]
    agree = (r96['rule_pred'] == r96['if_pred']).mean()
    pr(f'Methods agree on {100*agree:.1f}% of components.')
    both = r96['rule_pred'] & r96['if_pred']
    only_rule = r96['rule_pred'] & ~r96['if_pred']
    only_if = ~r96['rule_pred'] & r96['if_pred']
    pr(f'Flagged by both: {both.sum()}, only rule: {only_rule.sum()}, only IsolationForest: {only_if.sum()}')

    # -----------------------------------------------------------------------
    # Explainability: example flagged components
    # -----------------------------------------------------------------------
    h('EXAMPLE EXPLANATIONS (96h cutoff, rule-based - the interpretable method)')
    X96 = r96['X']
    example_ids = []
    for cl in ['BATCH_ANOMALY', 'GRADUAL_DRIFT', 'LATENT_DEFECT', 'SUDDEN_ANOMALY']:
        m = (df.Defect_Type == cl) & r96['rule_pred']
        if m.any():
            example_ids.append(df.index[m][0])

    for i in example_ids:
        cid = df.loc[i, 'Component_ID']
        batch = df.loc[i, 'Batch_ID']
        dtype = df.loc[i, 'Defect_Type']
        z_cols = [c for c in X96.columns if c.endswith('_z')]
        row = X96.loc[i, z_cols].abs().sort_values(ascending=False)
        top3 = row.head(3)
        pr(f'\n{cid} (batch {batch}, true label = {dtype} - label used for this demo only, never as model input)')
        for feat, val in top3.items():
            signed = X96.loc[i, feat]
            pr(f'  - {feat}: robust z = {signed:+.2f} (|z|={val:.2f})')
        pr(f'  -> Flagged: max |z| = {row.iloc[0]:.2f} exceeds the {Z_THRESHOLD} threshold, '
           f'driven mainly by {row.index[0]}.')

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    out_df = pd.DataFrame(out_cols)
    out_df.to_csv('anomaly_scores.csv', index=False)
    open('anomaly_detector_report.txt', 'w').write(buf.getvalue())
    print('\nSaved anomaly_scores.csv and anomaly_detector_report.txt')


if __name__ == '__main__':
    main()
