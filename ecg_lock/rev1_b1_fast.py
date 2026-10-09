"""Revision 1 fast Point-1 contrasts (AUROC only): B1 single realization, five realizations, stress k grid."""
import re
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
R = Path('/scratch/spath004/ecg_lock_runs'); P = 'cnn_full_leadwise_cleanfrac20'; M = 'cnn_diagonly_aug_matched'; A = 'cnn_diagonly_aug'; C = 'cnn_diagonly_clean'
def L(c, s, f):
    q = R / f'{c}_s{s}/bundles/{f}'
    if not q.exists() and f == 'test_traindist.npz': q = R / f'{c}_s{s}/bundles/test_traindist_predicted.npz'
    return dict(np.load(q))
def mauc(y, p): return float(np.mean([roc_auc_score(y[:, k], p[:, k]) for k in range(y.shape[1]) if 0 < y[:, k].mean() < 1]))
def groups(pid):
    _, inv = np.unique(pid, return_inverse=True); o = np.argsort(inv, kind='stable'); return np.split(o, np.flatnonzero(np.diff(inv[o])) + 1)
def boot(pid, fn, n, alpha, seed=20260926):
    g = groups(pid); rng = np.random.default_rng(seed); full = fn(np.arange(len(pid))); reps = []
    for _ in range(n):
        ix = np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]); reps.append(np.mean(fn(ix)))
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)]); return full, np.mean(full), lo, hi
def show(tag, r): print(f'>> {tag}: delta {r[1]:+.4f} [{r[2]:+.4f}, {r[3]:+.4f}] per-seed {np.round(r[0], 4).tolist()}', flush=True)
k0 = list(np.load(R / f'{M}_s0/bundles/test_traindist_multireal_v1.npz').keys()); print('multireal keys', k0[:12], len(k0), flush=True)
for (b, a), seeds in [((P, M), range(5)), ((P, M), range(3)), ((P, A), range(5)), ((P, A), range(3)), ((M, A), range(5)), ((M, C), range(5)), ((A, C), range(5))]:
    ss = list(seeds); Zb = {s: L(b, s, 'test_traindist.npz') for s in ss}; Za = {s: L(a, s, 'test_traindist.npz') for s in ss}
    for s in ss:
        for f in ('ecg_id', 'corruption_mask', 'noise_sigma', 'y_true'): assert np.array_equal(Zb[s][f], Za[s][f]), f
    y, pid = Zb[ss[0]]['y_true'], Zb[ss[0]]['patient_id']
    show(f'B1single {b} - {a} seeds{len(ss)} (97.5%CI)', boot(pid, lambda ix: [mauc(y[ix], Zb[s]['y_score'][ix]) - mauc(y[ix], Za[s]['y_score'][ix]) for s in ss], 1000, 0.025))
    Mb = {s: L(b, s, 'test_traindist_multireal_v1.npz') for s in ss}; Ma = {s: L(a, s, 'test_traindist_multireal_v1.npz') for s in ss}
    rk = sorted(k for k in Mb[ss[0]] if k.endswith('y_score')); yk = [k for k in Mb[ss[0]] if 'y_true' in k]
    ym = Mb[ss[0]][yk[0]] if yk else y
    show(f'5real {b} - {a} seeds{len(ss)} n_real={len(rk)} (95%CI)', boot(pid, lambda ix: [np.mean([mauc(ym[ix], Mb[s][k][ix]) - mauc(ym[ix], Ma[s][k][ix]) for k in rk]) for s in ss], 500, 0.05))
    if len(ss) == 5:
        Sb = {s: L(b, s, 'test_stress.npz') for s in ss}; Sa = {s: L(a, s, 'test_stress.npz') for s in ss}; ys = Sb[0]['y_true']; ps = Sb[0]['patient_id']
        for cond in ('missing|k=4', 'missing|k=8', 'missing|k=10'):
            ks = [k for k in Sb[0] if k.startswith(cond + '|') and k.endswith('|y_score')]
            if not ks: continue
            pb = {s: np.mean([Sb[s][k] for k in ks], 0) for s in ss}; pa = {s: np.mean([Sa[s][k] for k in ks], 0) for s in ss}
            show(f'stress {cond} {b} - {a}', boot(ps, lambda ix: [mauc(ys[ix], pb[s][ix]) - mauc(ys[ix], pa[s][ix]) for s in ss], 300, 0.05))
print('FAST_DONE')
