import sys, json, csv, hashlib, numpy as np, torch
from pathlib import Path
sys.path.insert(0, '/home/spath004/ecg_lock')
from data import load_labels, split_indices, CLASSES
from corruption import CorruptionProfile, corrupt_training
from models import build
from export import infer, repaired
from sklearn.metrics import roc_auc_score

def macro_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, k], p[:, k]) for k in range(y.shape[1]) if 0 < y[:, k].mean() < 1]))

root = Path('/home/spath004/cardio ML/ptbxl'); df = load_labels(root); X = np.load(root / 'cache_x100.npy', mmap_mode='r')
Y = df[CLASSES].values.astype(np.int8)
runs_dir = Path('/scratch/spath004/ecg_lock_runs'); out_dir = Path('/home/spath004/ecg_lock_results/recon_quality'); out_dir.mkdir(exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', dev, flush=True)
runs = ['cnn_full_leadwise_cleanfrac20_s0', 'cnn_full_leadwise_cleanfrac20_s1', 'cnn_full_leadwise_cleanfrac20_s2',
        'cnn_full_s0', 'cnn_full_s1', 'cnn_full_s2', 'cnn_diag_rec_s0', 'cnn_diag_rec_s1', 'cnn_diag_rec_s2',
        'cnn_full_leadwise_s0', 'cnn_full_leadwise_s1', 'cnn_full_leadwise_s2']
per_seed = {}; block = None; blo = None
KEYS = ['post_all', 'post_drop', 'post_noise', 'post_strong']
for run in runs:
    rd = runs_dir / run
    cfg = json.loads((rd / 'config.json').read_text()); lock = json.loads((rd / 'lock.json').read_text())
    sha = hashlib.sha256((rd / 'best.pt').read_bytes()).hexdigest(); assert sha == lock['checkpoint_sha256'], run
    model = build(cfg['arch'], tuple(cfg['tasks']), cfg.get('lead_head', 'pooled')).to(dev)
    model.load_state_dict(torch.load(rd / 'best.pt', map_location=dev))
    idx = split_indices(df, cfg['split'])['test']; lo, hi = int(idx.min()), int(idx.max())
    if block is None:
        block = np.array(X[lo:hi + 1]); blo = lo
    xr = block[idx - blo]; mu, sd = np.array(cfg['norm_mu'], np.float32), np.array(cfg['norm_sd'], np.float32)
    x = ((xr - mu[None, :, None]) / (sd[None, :, None] + 1e-6)).astype(np.float32); y = Y[idx]; pid = df.patient_id.values[idx]
    prof = CorruptionProfile(clean_frac=0.0, min_one_masked=True, sigma_min=0.0)
    xt, mask, ctype, sig, bw = corrupt_training(x, np.random.default_rng(2001), prof)
    b = np.load(rd / 'bundles' / 'test_traindist_oracle.npz'); assert np.array_equal(b['corruption_mask'], mask)
    o = infer(model, xt, dev); pre = o['diag']; rec = o['rec']
    dpre = float(np.abs(pre - b['pre_repair_score']).max())
    post_all = infer(model, repaired(xt, rec, mask), dev)['diag']
    dpost = float(np.abs(post_all - b['post_repair_score']).max())
    post_drop = infer(model, repaired(xt, rec, (ctype == 1).astype(np.int8)), dev)['diag']
    post_noise = infer(model, repaired(xt, rec, (ctype == 2).astype(np.int8)), dev)['diag']
    post_strong = infer(model, repaired(xt, rec, ((ctype == 2) & (sig >= 0.15)).astype(np.int8)), dev)['diag']
    per_seed[run] = dict(y=y, pid=pid, pre=pre, post_all=post_all, post_drop=post_drop, post_noise=post_noise, post_strong=post_strong)
    print('>>', run, 'bundle match pre %.2e post %.2e | pre %.4f all %.4f drop %.4f noise %.4f strong %.4f | records with dropout %d, with noise %d' % (
        dpre, dpost, macro_auc(y, pre), macro_auc(y, post_all), macro_auc(y, post_drop), macro_auc(y, post_noise), macro_auc(y, post_strong),
        int((ctype == 1).any(1).sum()), int((ctype == 2).any(1).sum())), flush=True)

def config_of(r): return r.rsplit('_s', 1)[0]
configs = []
for r in runs:
    if config_of(r) not in configs: configs.append(config_of(r))
rng = np.random.default_rng(0)
pid0 = per_seed[runs[0]]['pid']; pids, inv = np.unique(pid0, return_inverse=True)
groups = [np.flatnonzero(inv == g) for g in range(len(pids))]
B = 1000
samples = [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]
rows = []
for c in configs:
    seeds = [r for r in runs if config_of(r) == c]
    full = {r: {k: macro_auc(per_seed[r]['y'], per_seed[r][k]) for k in ['pre'] + KEYS} for r in seeds}
    boot = {k: [] for k in KEYS}
    for smp in samples:
        pre_s = {r: macro_auc(per_seed[r]['y'][smp], per_seed[r]['pre'][smp]) for r in seeds}
        for k in KEYS:
            boot[k].append(np.mean([macro_auc(per_seed[r]['y'][smp], per_seed[r][k][smp]) - pre_s[r] for r in seeds]))
    for k in KEYS:
        d = float(np.mean([full[r][k] - full[r]['pre'] for r in seeds])); lo_, hi_ = np.percentile(boot[k], [2.5, 97.5])
        ps = [round(full[r][k] - full[r]['pre'], 4) for r in seeds]
        rows.append(dict(config=c, repair=k, delta=d, ci95_lo=float(lo_), ci95_hi=float(hi_), per_seed=str(ps), pre_mean=float(np.mean([full[r]['pre'] for r in seeds]))))
        print('##', c, k, 'delta %.4f [%.4f, %.4f]' % (d, lo_, hi_), ps, flush=True)
with open(out_dir / 'T_repair_by_type.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrote', out_dir / 'T_repair_by_type.csv', flush=True)
