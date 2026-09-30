"""Builds dashboard_data.json for reliefos_console.html from the ACTUAL
trained classifier (classifier.py), not the retired rule engine."""
import json, numpy as np, pandas as pd
import classifier as clf

df = clf.df
P = clf.P
LIM = clf.LIM

def explain_path(tree, row, feature_names):
    t = tree.tree_
    classes = list(tree.classes_)
    node = 0
    path = []
    while t.feature[node] != -2:
        feat = feature_names[t.feature[node]]
        thr = t.threshold[node]
        val = row[feat]
        if val <= thr:
            path.append(f'{feat} = {val:.3g} (\u2264 {thr:.3g})')
            node = t.children_left[node]
        else:
            path.append(f'{feat} = {val:.3g} (> {thr:.3g})')
            node = t.children_right[node]
    counts = t.value[node][0]
    cls = classes[int(np.argmax(counts))]
    conf = counts.max() / counts.sum()
    return cls, conf, path

data_per_cutoff = {}
for cutoff in (24, 96):
    cols = clf.deploy_cols_by_cutoff[cutoff]
    X = clf.feats[cutoff][cols]
    tree = clf.fitted[cutoff]
    preds, confs, paths = [], [], []
    for i in X.index:
        cls, conf, path = explain_path(tree, X.loc[i], cols)
        preds.append(cls); confs.append(conf); paths.append(path)
    data_per_cutoff[cutoff] = {'pred': preds, 'conf': confs, 'path': paths}

# driving parameter from Stage 1's rule (for the trajectory trace), per cutoff
driver24 = clf.stage1.rule_flag(clf.stage1.build_features(24))[1]
driver96 = clf.stage1.rule_flag(clf.stage1.build_features(96))[1]

def param_of(driver_str):
    for p in P:
        if driver_str.startswith(p):
            return p
    return P[0]

records = []
for idx, row in df.iterrows():
    p24, c24 = data_per_cutoff[24]['pred'][idx], data_per_cutoff[24]['conf'][idx]
    p96, c96 = data_per_cutoff[96]['pred'][idx], data_per_cutoff[96]['conf'][idx]
    if p24 == 'NORMAL' and p96 == 'NORMAL':
        continue
    param = param_of(driver96.iloc[idx])
    traj = [round(float(row[f'{param}_{t}h']), 3) for t in (0, 24, 96, 168)]
    records.append({
        'id': row.Component_ID, 'b': row.Batch_ID,
        'lvl24': p24, 'conf24': round(float(c24), 3), 'path24': data_per_cutoff[24]['path'][idx],
        'lvl96': p96, 'conf96': round(float(c96), 3), 'path96': data_per_cutoff[96]['path'][idx],
        'param': param.replace('_mA', ' (mA)').replace('_uA', ' (uA)').replace('_ns', ' (ns)').replace('Propagation_Delay', 'Delay'),
        'lim': LIM[param], 'traj': traj,
        'true': row.Defect_Type
    })

overall24 = pd.Series(data_per_cutoff[24]['pred']).value_counts().reindex(['NORMAL', 'MONITOR', 'HIGH_RISK'], fill_value=0).to_dict()
overall96 = pd.Series(data_per_cutoff[96]['pred']).value_counts().reindex(['NORMAL', 'MONITOR', 'HIGH_RISK'], fill_value=0).to_dict()

# accuracy summary (from the CV comparison already computed by classifier.py)
acc = {
    '24': {'accuracy': 0.868, 'macro_f1': 0.716, 'chained': False},
    '96': {'accuracy': 0.908, 'macro_f1': 0.805, 'chained': True},
}

out = {'records': records, 'overall24': overall24, 'overall96': overall96,
       'total': int(len(df)), 'acc': acc}
json.dump(out, open('dashboard_data_v2.json', 'w'), separators=(',', ':'))
print(len(records), 'flagged components (either cutoff)')
print('bytes:', len(json.dumps(out, separators=(",", ":"))))
