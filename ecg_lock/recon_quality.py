import sys, json, csv, numpy as np
from pathlib import Path
sys.path.insert(0, '/home/spath004/ecg_lock')
from data import load_labels, split_indices
from corruption import CorruptionProfile, corrupt_training
root = Path('/home/spath004/cardio ML/ptbxl')
df = load_labels(root)
X = np.load(root / 'cache_x100.npy', mmap_mode='r')
assert X.shape[0] == len(df), (X.shape, len(df))
runs_dir = Path('/scratch/spath004/ecg_lock_runs')
out_dir = Path('/home/spath004/ecg_lock_results/recon_quality'); out_dir.mkdir(exist_ok=True)

def pear(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1)) + 1e-12
    return (a * b).sum(-1) / den

rows = []
block_cache = {}
for run in ['cnn_full_leadwise_cleanfrac20_s0', 'cnn_full_leadwise_cleanfrac20_s1', 'cnn_full_leadwise_cleanfrac20_s2',
            'cnn_full_s0', 'cnn_full_s1', 'cnn_full_s2', 'cnn_diag_rec_s0', 'cnn_diag_rec_s1', 'cnn_diag_rec_s2',
            'cnn_full_leadwise_s0', 'cnn_full_leadwise_s1', 'cnn_full_leadwise_s2']:
    rd = runs_dir / run
    if not (rd / 'bundles' / 'test_traindist_oracle.npz').exists():
        print('missing', run, flush=True); continue
    cfg = json.loads((rd / 'config.json').read_text())
    idx = split_indices(df, cfg['split'])['test']
    lo, hi = int(idx.min()), int(idx.max())
    if (lo, hi) not in block_cache:
        block_cache.clear(); block_cache[(lo, hi)] = np.array(X[lo:hi + 1])
    xr = block_cache[(lo, hi)][idx - lo]
    mu, sd = np.array(cfg['norm_mu'], np.float32), np.array(cfg['norm_sd'], np.float32)
    x = ((xr - mu[None, :, None]) / (sd[None, :, None] + 1e-6)).astype(np.float32)
    prof = CorruptionProfile(clean_frac=0.0, min_one_masked=True, sigma_min=0.0)
    xt, mask, ctype, sig, bw = corrupt_training(x, np.random.default_rng(2001), prof)
    b = np.load(rd / 'bundles' / 'test_traindist_oracle.npz')
    assert np.array_equal(b['corruption_mask'], mask), 'mask mismatch'
    xtrue = b['reconstruction_true'].astype(np.float32); rec = b['reconstruction_pred'].astype(np.float32)
    dmax = float(np.abs(xtrue - x).max()); assert dmax < 0.05, dmax
    pid = b['patient_id']
    err_in = ((xt - x) ** 2).mean(-1); err_rec = ((rec - x) ** 2).mean(-1); var_x = x.var(-1)
    r_in = pear(xt, x); r_rec = pear(rec, x)
    m = mask.astype(bool)
    def summ(name, sel):
        n = int(sel.sum())
        d = dict(run=run, subset=name, n_leads=n,
                 mse_in=float(err_in[sel].mean()), mse_rec=float(err_rec[sel].mean()),
                 rel_reduction=float(1 - err_rec[sel].sum() / err_in[sel].sum()) if err_in[sel].sum() > 0 else float('nan'),
                 frac_improved=float((err_rec[sel] < err_in[sel]).mean()),
                 r_in=float(r_in[sel].mean()), r_rec=float(r_rec[sel].mean()), r_rec_median=float(np.median(r_rec[sel])),
                 var_explained=float(1 - err_rec[sel].sum() / var_x[sel].sum()))
        per_rec_in = (err_in * sel).sum(1); per_rec_rec = (err_rec * sel).sum(1)
        keep = per_rec_in > 0
        rng = np.random.default_rng(0)
        pids, inv = np.unique(pid[keep], return_inverse=True)
        pin = np.bincount(inv, weights=per_rec_in[keep]); prc = np.bincount(inv, weights=per_rec_rec[keep])
        ii = rng.integers(0, len(pin), size=(1000, len(pin)))
        bs = 1 - prc[ii].sum(1) / pin[ii].sum(1)
        d['rel_reduction_ci_lo'] = float(np.percentile(bs, 2.5)); d['rel_reduction_ci_hi'] = float(np.percentile(bs, 97.5))
        rows.append(d)
        print('>>', run, name, 'n', n, 'mse_in %.3f mse_rec %.3f rel %.3f [%.3f, %.3f] improved %.3f r_in %.3f r_rec %.3f (med %.3f) varexp %.3f' % (
            d['mse_in'], d['mse_rec'], d['rel_reduction'], d['rel_reduction_ci_lo'], d['rel_reduction_ci_hi'], d['frac_improved'], d['r_in'], d['r_rec'], d['r_rec_median'], d['var_explained']), flush=True)
    summ('corrupted_all', m)
    summ('dropout', ctype == 1)
    summ('noise_all', ctype == 2)
    summ('noise_sigma_lt_0.05', (ctype == 2) & (sig < 0.05))
    summ('noise_sigma_0.05_0.15', (ctype == 2) & (sig >= 0.05) & (sig < 0.15))
    summ('noise_sigma_ge_0.15', (ctype == 2) & (sig >= 0.15))
    summ('intact', ~m)
    print('>> bw records', int(bw.sum()), 'float16 max dev %.4f' % dmax, flush=True)
with open(out_dir / 'T_recon_quality.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrote', out_dir / 'T_recon_quality.csv', len(rows), 'rows', flush=True)
