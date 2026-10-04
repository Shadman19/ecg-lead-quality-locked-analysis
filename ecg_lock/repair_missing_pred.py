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
runs = ['cnn_full_leadwise_cleanfrac20_s0', 'cnn_full_leadwise_cleanfrac20_s1', 'cnn_full_leadwise_cleanfrac20_s2']
KS = [2, 4, 6, 8, 10]; R = 5; sseed = 2000
res = {}; block = None; blo = None
for run in runs:
    rd = runs_dir / run
    cfg = json.loads((rd / 'config.json').read_text()); lock = json.loads((rd / 'lock.json').read_text())
    assert hashlib.sha256((rd / 'best.pt').read_bytes()).hexdigest() == lock['checkpoint_sha256'], run
    tau = float(lock['tau_mask'])
    model = build(cfg['arch'], tuple(cfg['tasks']), cfg.get('lead_head', 'pooled')).to(dev)
    model.load_state_dict(torch.load(rd / 'best.pt', map_location=dev))
    idx = split_indices(df, cfg['split'])['test']; lo, hi = int(idx.min()), int(idx.max())
    if block is None:
        block = np.array(X[lo:hi + 1]); blo = lo
    xr = block[idx - blo]; mu, sd = np.array(cfg['norm_mu'], np.float32), np.array(cfg['norm_sd'], np.float32)
    x = ((xr - mu[None, :, None]) / (sd[None, :, None] + 1e-6)).astype(np.float32); y = Y[idx]; pid = df.patient_id.values[idx]
    st = np.load(rd / 'bundles' / 'test_stress.npz')
    res[run] = dict(y=y, pid=pid, tau=tau)
    print('>> run', run, 'tau_mask %.4f' % tau, flush=True)
    for k in KS:
        for r in range(R):
            rng = np.random.default_rng(zlib.crc32(f'{sseed}|missing|{k}|0.0|0.0|{r}'.encode()))
            xs, ms, _, _ = corrupt_stress(x, rng, k, 0.0, 0.0)
            o = infer(model, xs, dev)
            key = f'missing|k={k}|s=0.0|bw=0.0|r={r}'
            dmatch = float(np.abs(o['diag'] - st[key + '|y_score']).max())
            pm = (o['lead'] < tau).astype(np.int8)
            post_o = infer(model, repaired(xs, o['rec'], ms), dev)['diag']
            post_p = infer(model, repaired(xs, o['rec'], pm), dev)['diag']
            res[run][(k, r)] = dict(pre=o['diag'], post_o=post_o, post_p=post_p, match=dmatch, sens=float(pm[ms == 1].mean()), frr=float(pm[ms == 0].mean()))
        pre_m = np.mean([macro_auc(y, res[run][(k, r)]['pre']) for r in range(R)])
        po_m = np.mean([macro_auc(y, res[run][(k, r)]['post_o']) for r in range(R)])
        pp_m = np.mean([macro_auc(y, res[run][(k, r)]['post_p']) for r in range(R)])
        print('>>', run, 'k=%d pre %.4f oracle %+.4f pred %+.4f sens %.3f frr %.3f (match %.1e)' % (k, pre_m, po_m - pre_m, pp_m - pre_m, np.mean([res[run][(k, r)]['sens'] for r in range(R)]), np.mean([res[run][(k, r)]['frr'] for r in range(R)]), max(res[run][(k, r)]['match'] for r in range(R))), flush=True)

rng = np.random.default_rng(0)
pid0 = res[runs[0]]['pid']; pids, inv = np.unique(pid0, return_inverse=True)
groups = [np.flatnonzero(inv == g) for g in range(len(pids))]
B = 500
samples = [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]
rows = []
for k in KS:
    def deltas(smp):
        do, dp = [], []
        for r in runs:
            y = res[r]['y'][smp]
            pre = [macro_auc(y, res[r][(k, q)]['pre'][smp]) for q in range(R)]
            do.append(np.mean([macro_auc(y, res[r][(k, q)]['post_o'][smp]) - pre[q] for q in range(R)]))
            dp.append(np.mean([macro_auc(y, res[r][(k, q)]['post_p'][smp]) - pre[q] for q in range(R)]))
        return float(np.mean(do)), float(np.mean(dp)), do, dp
    d0o, d0p, pero, perp = deltas(np.arange(len(pid0)))
    bs = np.array([deltas(s)[:2] for s in samples])
    for name, d0, col, per in [('oracle', d0o, 0, pero), ('pred', d0p, 1, perp)]:
        lo_, hi_ = np.percentile(bs[:, col], [2.5, 97.5])
        print('## proposed k=%d %s delta %+.4f [%+.4f, %+.4f] per-seed %s' % (k, name, d0, lo_, hi_, ' '.join('%+.4f' % v for v in per)), flush=True)
        rows.append(dict(k=k, mask=name, delta=d0, lo=lo_, hi=hi_, per_seed=';'.join('%.5f' % v for v in per), sens=np.mean([res[r][(k, q)]['sens'] for r in runs for q in range(R)]), frr=np.mean([res[r][(k, q)]['frr'] for r in runs for q in range(R)])))
with open(out_dir / 'T_repair_missing_pred.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrote', out_dir / 'T_repair_missing_pred.csv')
