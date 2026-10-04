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
P = 'cnn_full_leadwise_cleanfrac20'; A = 'cnn_diagonly_aug'
KS = [2, 4, 6, 8, 10]; R = 5; sseed = 2000
res = {}; block = None; blo = None
for s in range(3):
    rd = runs_dir / f'{P}_s{s}'
    cfg = json.loads((rd / 'config.json').read_text()); lock = json.loads((rd / 'lock.json').read_text())
    assert hashlib.sha256((rd / 'best.pt').read_bytes()).hexdigest() == lock['checkpoint_sha256'], rd
    tau = float(lock['tau_mask'])
    model = build(cfg['arch'], tuple(cfg['tasks']), cfg.get('lead_head', 'pooled')).to(dev)
    model.load_state_dict(torch.load(rd / 'best.pt', map_location=dev))
    idx = split_indices(df, cfg['split'])['test']; lo, hi = int(idx.min()), int(idx.max())
    if block is None:
        block = np.array(X[lo:hi + 1]); blo = lo
    xr = block[idx - blo]; mu, sd = np.array(cfg['norm_mu'], np.float32), np.array(cfg['norm_sd'], np.float32)
    x = ((xr - mu[None, :, None]) / (sd[None, :, None] + 1e-6)).astype(np.float32); y = Y[idx]; pid = df.patient_id.values[idx]
    st = np.load(rd / 'bundles' / 'test_stress.npz')
    ra = runs_dir / f'{A}_s{s}'; cfga = json.loads((ra / 'config.json').read_text()); locka = json.loads((ra / 'lock.json').read_text())
    assert cfga['split'] == cfg['split'], (cfga['split'], cfg['split'])
    ca = np.load(ra / 'bundles' / 'test_clean.npz'); assert np.array_equal(ca['y_true'].astype(np.int8), y), 'aug test order differs'
    sta = np.load(ra / 'bundles' / 'test_stress.npz')
    res[s] = dict(y=y, pid=pid)
    print('>> seed', s, 'tau_mask %.4f' % tau, 'aug checkpoint', locka.get('checkpoint_sha256', '')[:12], flush=True)
    for k in KS:
        for r in range(R):
            rng = np.random.default_rng(zlib.crc32(f'{sseed}|missing|{k}|0.0|0.0|{r}'.encode()))
            xs, ms, _, _ = corrupt_stress(x, rng, k, 0.0, 0.0)
            o = infer(model, xs, dev)
            key = f'missing|k={k}|s=0.0|bw=0.0|r={r}'
            assert float(np.abs(o['diag'] - st[key + '|y_score']).max()) < 1e-5, 'bundle mismatch'
            pm = (o['lead'] < tau).astype(np.int8)
            post_o = infer(model, repaired(xs, o['rec'], ms), dev)['diag']
            post_p = infer(model, repaired(xs, o['rec'], pm), dev)['diag']
            res[s][(k, r)] = dict(pre=o['diag'], post_o=post_o, post_p=post_p, aug=np.asarray(sta[key + '|y_score'], dtype=np.float32))
        f = lambda key: np.mean([macro_auc(y, res[s][(k, r)][key]) for r in range(R)])
        print('>> seed %d k=%d pre %.4f oracle %.4f pred %.4f aug %.4f' % (s, k, f('pre'), f('post_o'), f('post_p'), f('aug')), flush=True)

rng = np.random.default_rng(0)
pid0 = res[0]['pid']; pids, inv = np.unique(pid0, return_inverse=True)
groups = [np.flatnonzero(inv == g) for g in range(len(pids))]
B = 500
samples = [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]
rows = []
CON = [('pre_minus_aug', 'pre'), ('oracle_minus_aug', 'post_o'), ('pred_minus_aug', 'post_p')]
for k in KS:
    def deltas(smp):
        out = {c: [] for c, _ in CON}
        for s in range(3):
            y = res[s]['y'][smp]
            aug = [macro_auc(y, res[s][(k, q)]['aug'][smp]) for q in range(R)]
            for c, key in CON:
                out[c].append(np.mean([macro_auc(y, res[s][(k, q)][key][smp]) - aug[q] for q in range(R)]))
        return {c: (float(np.mean(v)), v) for c, v in out.items()}
    d0 = deltas(np.arange(len(pid0)))
    bs = [deltas(smp) for smp in samples]
    for c, _ in CON:
        arr = np.array([b[c][0] for b in bs]); lo_, hi_ = np.percentile(arr, [2.5, 97.5])
        print('## k=%d %s delta %+.4f [%+.4f, %+.4f] per-seed %s' % (k, c, d0[c][0], lo_, hi_, ' '.join('%+.4f' % v for v in d0[c][1])), flush=True)
        rows.append(dict(k=k, contrast=c, delta=d0[c][0], lo=lo_, hi=hi_, per_seed=';'.join('%.5f' % v for v in d0[c][1])))
with open(out_dir / 'T_repair_missing_vs_aug.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrote', out_dir / 'T_repair_missing_vs_aug.csv', flush=True)
