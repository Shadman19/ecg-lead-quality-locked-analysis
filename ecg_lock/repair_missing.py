import sys, json, csv, zlib, hashlib, numpy as np, torch
from pathlib import Path
sys.path.insert(0, '/home/spath004/ecg_lock')
from data import load_labels, split_indices, CLASSES
from corruption import corrupt_stress
from models import build
from export import infer, repaired
from sklearn.metrics import roc_auc_score

def macro_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, k], p[:, k]) for k in range(y.shape[1]) if 0 < y[:, k].mean() < 1]))

root = Path('/home/spath004/cardio ML/ptbxl'); df = load_labels(root); X = np.load(root / 'cache_x100.npy', mmap_mode='r')
Y = df[CLASSES].values.astype(np.int8)
runs_dir = Path('/scratch/spath004/ecg_lock_runs'); out_dir = Path('/home/spath004/ecg_lock_results/recon_quality'); out_dir.mkdir(exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
runs = ['cnn_full_leadwise_cleanfrac20_s0', 'cnn_full_leadwise_cleanfrac20_s1', 'cnn_full_leadwise_cleanfrac20_s2', 'cnn_full_s0', 'cnn_full_s1', 'cnn_full_s2']
KS = [2, 4, 6, 8, 10]; R = 5; sseed = 2000
res = {}; block = None; blo = None
for run in runs:
    rd = runs_dir / run
    cfg = json.loads((rd / 'config.json').read_text()); lock = json.loads((rd / 'lock.json').read_text())
    assert hashlib.sha256((rd / 'best.pt').read_bytes()).hexdigest() == lock['checkpoint_sha256'], run
    model = build(cfg['arch'], tuple(cfg['tasks']), cfg.get('lead_head', 'pooled')).to(dev)
    model.load_state_dict(torch.load(rd / 'best.pt', map_location=dev))
    idx = split_indices(df, cfg['split'])['test']; lo, hi = int(idx.min()), int(idx.max())
    if block is None:
        block = np.array(X[lo:hi + 1]); blo = lo
    xr = block[idx - blo]; mu, sd = np.array(cfg['norm_mu'], np.float32), np.array(cfg['norm_sd'], np.float32)
    x = ((xr - mu[None, :, None]) / (sd[None, :, None] + 1e-6)).astype(np.float32); y = Y[idx]; pid = df.patient_id.values[idx]
    st = np.load(rd / 'bundles' / 'test_stress.npz')
    res[run] = dict(y=y, pid=pid)
    for k in KS:
        for r in range(R):
            rng = np.random.default_rng(zlib.crc32(f'{sseed}|missing|{k}|0.0|0.0|{r}'.encode()))
            xs, ms, _, _ = corrupt_stress(x, rng, k, 0.0, 0.0)
            o = infer(model, xs, dev)
            key = f'missing|k={k}|s=0.0|bw=0.0|r={r}'
            dmatch = float(np.abs(o['diag'] - st[key + '|y_score']).max())
            post = infer(model, repaired(xs, o['rec'], ms), dev)['diag']
            res[run][(k, r)] = dict(pre=o['diag'], post=post, match=dmatch)
        pre_m = np.mean([macro_auc(y, res[run][(k, r)]['pre']) for r in range(R)]); post_m = np.mean([macro_auc(y, res[run][(k, r)]['post']) for r in range(R)])
        print('>>', run, 'k=%d pre %.4f post %.4f delta %+.4f (bundle match max %.1e)' % (k, pre_m, post_m, post_m - pre_m, max(res[run][(k, r)]['match'] for r in range(R))), flush=True)

def config_of(r): return r.rsplit('_s', 1)[0]
configs = []
for r in runs:
    if config_of(r) not in configs: configs.append(config_of(r))
rng = np.random.default_rng(0)
pid0 = res[runs[0]]['pid']; pids, inv = np.unique(pid0, return_inverse=True)
groups = [np.flatnonzero(inv == g) for g in range(len(pids))]
B = 500
samples = [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]
rows = []
for c in configs:
    seeds = [r for r in runs if config_of(r) == c]
    for k in KS:
        def delta(smp):
            vals = []
            for r in seeds:
                y = res[r]['y'][smp]
                vals.append(np.mean([macro_auc(y, res[r][(k, q)]['post'][smp]) - macro_auc(y, res[r][(k, q)]['pre'][smp]) for q in range(R)]))
            return float(np.mean(vals))
        full = np.arange(len(res[seeds[0]]['y']))
        d = delta(full); bs = [delta(s) for s in samples]; lo_, hi_ = np.percentile(bs, [2.5, 97.5])
        pre_abs = np.mean([macro_auc(res[r]['y'], res[r][(k, q)]['pre']) for r in seeds for q in range(R)])
        post_abs = np.mean([macro_auc(res[r]['y'], res[r][(k, q)]['post']) for r in seeds for q in range(R)])
        rows.append(dict(config=c, k=k, pre=pre_abs, post=post_abs, delta=d, ci95_lo=float(lo_), ci95_hi=float(hi_)))
        print('##', c, 'k=%d pre %.4f post %.4f delta %+.4f [%+.4f, %+.4f]' % (k, pre_abs, post_abs, d, lo_, hi_), flush=True)
with open(out_dir / 'T_repair_missing.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrote', flush=True)
