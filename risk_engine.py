"""
SIH26170 - Stage 3: Risk engine

Combines the two earlier stages into one risk call per component:
  A) CURRENT anomaly evidence   -> anomaly_scores.csv   (Stage 1, 96h cutoff)
  B) FUTURE drift evidence      -> drift_predictions.csv (Stage 2, best model)
  C) Absolute-threshold proximity of the *predicted* 168h value

Rules are simple, transparent and configurable (all thresholds are named
constants below) per the project's "no unvalidated hard-coded thresholds"
principle - they are validated against the known ground truth at the bottom
of this script, and should be re-checked if the generator changes again.

Ground truth (Defect_Type/Defect_Subtype) is used ONLY to evaluate the
engine's calls afterward - never inside the risk logic itself.

Output levels: NORMAL / MONITOR / HIGH_RISK

Outputs:
  risk_assessment.csv     - one row per component: evidence + risk level + explanation
  risk_engine_report.txt  - validation of the engine against known defect classes
"""
import pandas as pd, numpy as np, io

pd.set_option('display.width', 200)
anom = pd.read_csv('anomaly_scores.csv')
drift = pd.read_csv('drift_predictions.csv')
P = ['Iddq_mA', 'Leakage_uA', 'Propagation_Delay_ns']
LIM = {'Iddq_mA': 5.0, 'Leakage_uA': 200.0, 'Propagation_Delay_ns': 15.0}
CL = ['NORMAL', 'BATCH_ANOMALY', 'GRADUAL_DRIFT', 'SUDDEN_ANOMALY', 'LATENT_DEFECT']
id_cols = ['Component_ID', 'Batch_ID', 'Defect_Type', 'Defect_Subtype']

df = anom.merge(drift.drop(columns=['Batch_ID', 'Defect_Type', 'Defect_Subtype']),
                 on='Component_ID', how='inner')

buf = io.StringIO()
def pr(*a):
    print(*a); print(*a, file=buf)
def h(s): pr('\n' + '=' * 78 + '\n' + s + '\n' + '=' * 78)

# ---------------------------------------------------------------------------
# CONFIGURABLE THRESHOLDS (simulation assumptions, tune per SIH feedback)
# ---------------------------------------------------------------------------
Z_HIGH = 5.0          # current anomaly: |z| above this alone is HIGH_RISK
Z_MONITOR = 3.5        # current anomaly: |z| above this is at least MONITOR (matches Stage 1 flag)
PCT_LIMIT_HIGH = 90.0   # predicted 168h >= this %-of-limit -> HIGH_RISK
PCT_LIMIT_MONITOR = 70.0  # predicted 168h >= this %-of-limit -> at least MONITOR

# ---------------------------------------------------------------------------
# Evidence B/C: worst predicted %-of-limit across the 3 parameters, and
# which parameter drives it (using each parameter's best model from Stage 2)
# ---------------------------------------------------------------------------
pred_pct = pd.DataFrame(index=df.index)
for p in P:
    pred_pct[p] = df[f'{p}_168h_pred_BEST'] / LIM[p] * 100
df['worst_pred_pct_limit'] = pred_pct.max(axis=1)
df['worst_pred_param'] = pred_pct.idxmax(axis=1)
df['predicted_absolute_fail'] = (pred_pct > 100).any(axis=1)

# ---------------------------------------------------------------------------
# Risk logic - transparent, rule-based, combines A + B + C
# ---------------------------------------------------------------------------
def assess(row):
    z = row['rule_maxZ_96h']
    pct = row['worst_pred_pct_limit']
    will_fail = row['predicted_absolute_fail']

    if will_fail or z >= Z_HIGH or (row['rule_flag_96h'] and pct >= PCT_LIMIT_HIGH):
        level = 'HIGH_RISK'
    elif row['rule_flag_96h'] or z >= Z_MONITOR or pct >= PCT_LIMIT_MONITOR:
        level = 'MONITOR'
    else:
        level = 'NORMAL'

    reasons = []
    if will_fail:
        reasons.append(f"predicted {row['worst_pred_param']} at 168h exceeds the absolute limit")
    elif pct >= PCT_LIMIT_MONITOR:
        reasons.append(f"predicted {row['worst_pred_param']} at 168h reaches {pct:.0f}% of its limit")
    if row['rule_flag_96h']:
        reasons.append(f"currently {z:.1f} robust SDs from batch median on {row['rule_driver_96h']} at 96h")
    if not reasons:
        reasons.append("no significant current anomaly or predicted drift")
    explanation = "; ".join(reasons)
    return pd.Series({'risk_level': level, 'explanation': explanation})

df[['risk_level', 'explanation']] = df.apply(assess, axis=1)

h('RISK ENGINE - CONFIGURATION')
pr(f'Z_HIGH={Z_HIGH}  Z_MONITOR={Z_MONITOR}  PCT_LIMIT_HIGH={PCT_LIMIT_HIGH}%  PCT_LIMIT_MONITOR={PCT_LIMIT_MONITOR}%')
pr('HIGH_RISK  = predicted absolute failure, OR extreme current anomaly (|z|>=5),')
pr('             OR (currently flagged AND predicted >=90% of limit)')
pr('MONITOR    = currently flagged (Stage 1), OR |z|>=3.5, OR predicted >=70% of limit')
pr('NORMAL     = none of the above')

h('OVERALL RISK DISTRIBUTION')
pr(df.risk_level.value_counts().reindex(['NORMAL', 'MONITOR', 'HIGH_RISK']).to_string())

h('RISK LEVEL BY TRUE DEFECT TYPE (evaluation only - not used by the engine)')
tbl = pd.crosstab(df.Defect_Type, df.risk_level)
for lvl in ['NORMAL', 'MONITOR', 'HIGH_RISK']:
    if lvl not in tbl.columns:
        tbl[lvl] = 0
pr(tbl[['NORMAL', 'MONITOR', 'HIGH_RISK']].reindex(CL).to_string())

h('RISK LEVEL BY SUBTYPE (finer-grained check)')
tbl2 = pd.crosstab(df.Defect_Subtype, df.risk_level)
for lvl in ['NORMAL', 'MONITOR', 'HIGH_RISK']:
    if lvl not in tbl2.columns:
        tbl2[lvl] = 0
pr(tbl2[['NORMAL', 'MONITOR', 'HIGH_RISK']].to_string())

h('KEY VALIDATION CHECKS')
normal_fp = (df[df.Defect_Type == 'NORMAL'].risk_level != 'NORMAL').mean() * 100
pr(f'False-positive rate on NORMAL (flagged MONITOR or HIGH_RISK): {normal_fp:.1f}%')

sudden_after = df[df.Defect_Subtype == 'SUDDEN_AFTER_96H']
pr(f"\nSUDDEN_AFTER_96H (step hasn't happened by 96h yet - true early-warning test):")
pr(sudden_after.risk_level.value_counts().to_string())
pr('These should mostly be NORMAL/MONITOR at 96h, not HIGH_RISK - nothing has')
pr('happened yet. The drift predictor cannot see a jump that has not occurred.')

latent = df[df.Defect_Subtype == 'LATENT_DEFECT']
pr(f"\nLATENT_DEFECT (never crosses absolute limit - tests the early-warning value):")
pr(latent.risk_level.value_counts().to_string())
n_latent_caught = (latent.risk_level != 'NORMAL').sum()
pr(f'{n_latent_caught}/{len(latent)} latent defects caught as MONITOR or HIGH_RISK')
pr('despite NONE of them ever crossing an absolute limit - this is the case')
pr('a static pass/fail check would completely miss.')

h('EXAMPLE EXPLANATIONS')
for lvl in ['HIGH_RISK', 'MONITOR']:
    for cl in CL:
        m = (df.risk_level == lvl) & (df.Defect_Type == cl)
        if m.any():
            r = df[m].iloc[0]
            pr(f"\n{r.Component_ID} (batch {r.Batch_ID}, true label={r.Defect_Type}/{r.Defect_Subtype})")
            pr(f"  Risk level: {r.risk_level}")
            pr(f"  Explanation: {r.explanation}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_cols = id_cols + ['rule_maxZ_96h', 'rule_driver_96h', 'rule_flag_96h',
                       'worst_pred_param', 'worst_pred_pct_limit',
                       'predicted_absolute_fail', 'risk_level', 'explanation']
df[out_cols].to_csv('risk_assessment.csv', index=False)
open('risk_engine_report.txt', 'w').write(buf.getvalue())
print('\nSaved risk_assessment.csv and risk_engine_report.txt')
