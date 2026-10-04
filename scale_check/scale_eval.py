"""Evaluate the width-scaled models with the prespecified test corruption and the missing-lead grid
(post hoc scale check; endpoints fixed in SCALE_PLAN.md before any run)."""
import sys, json, csv, zlib, hashlib, numpy as np, torch
from pathlib import Path
sys.path.insert(0, '/home/spath004/ecg_lock'); sys.path.insert(0, '/home/spath004/xfer')
from data import load_labels, split_indices, CLASSES
from corruption import CorruptionProfile, corrupt_training, corrupt_stress
from export import infer, repaired
from scale_common import wide_build
from sklearn.metrics import roc_auc_score

def macro_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, k], p[:, k]) for k in range(y.shape[1]) if 0 < y[:, k].mean() < 1]))

root = Path('/home/spath004/cardio ML/ptbxl'); df = load_labels(root); X = np.load(root / 'cache_x100.npy', mmap_mode='r')
Y = df[CLASSES].values.astype(np.int8)
base = Path('/scratch/spath004/ecg_scale_runs'); out_dir = Path('/home/spath004/ecg_lock_results/scale'); out_dir.mkdir(exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
CONF = {'aug': 'scale_w4_diagonly_aug', 'prop': 'scale_w4_proposed'}
KS = [4, 8, 10]; R = 5
res = {}; block = None
for s in range(3):
    res[s] = {}
    for key, name in CONF.items():
        rd = base / f'{name}_s{s}'
        cfg = json.loads((rd / 'config.json').read_text()); tl = json.loads((rd / 'train_log.json').read_text())
        assert hashlib.sha256((rd / 'best.pt').read_bytes()).hexdigest() == tl['checkpoint_sha256'], rd
        model = wide_build('cnn', tuple(cfg['tasks']), cfg['lead_head'], 4).to(dev)
        model.load_state_dict(torch.load(rd / 'best.pt', map_location=dev)); model.eval()
        idx = split_indices(df, cfg['split'])['test']; lo, hi = int(idx.min()), int(idx.max())
        if block is None:
            block = (lo, np.array(X[lo:hi + 1]))
        xr = block[1][idx - block[0]]; mu, sd = np.array(cfg['norm_mu'], np.float32), np.array(cfg['norm_sd'], np.float32)
        x = ((xr - mu[None, :, None]) / (sd[None, :, None] + 1e-6)).astype(np.float32); y = Y[idx]
        r = dict(y=y, pid=df.patient_id.values[idx], n_params=cfg['n_params'], best_epoch=tl['best_epoch'])
        r['clean'] = infer(model, x, dev)['diag']
        prof = CorruptionProfile(clean_frac=0.0, min_one_masked=True, sigma_min=0.0)
        xt, mask, ctype, sig, bw = corrupt_training(x, np.random.default_rng(2001), prof)
        o = infer(model, xt, dev); r['corr'] = o['diag']
        if 'rec' in o:
            r['rep_oracle'] = infer(model, repaired(xt, o['rec'], mask), dev)['diag']
            r['rep_drop'] = infer(model, repaired(xt, o['rec'], (ctype == 1).astype(np.int8)), dev)['diag']
        for k in KS:
            for q in range(R):
                rng = np.random.default_rng(zlib.crc32(f'2000|missing|{k}|0.0|0.0|{q}'.encode()))
                xs, ms, _, _ = corrupt_stress(x, rng, k, 0.0, 0.0)
                o = infer(model, xs, dev); r[('miss', k, q)] = o['diag']
                if 'rec' in o:
                    r[('missrep', k, q)] = infer(model, repaired(xs, o['rec'], ms), dev)['diag']
        res[s][key] = r
        print('>>', name, s, 'params', r['n_params'], 'best_ep', r['best_epoch'], 'clean %.4f corr %.4f' % (macro_auc(y, r['clean']), macro_auc(y, r['corr'])), flush=True)

pid0 = res[0]['aug']['pid']; pids, inv = np.unique(pid0, return_inverse=True)
groups = [np.flatnonzero(inv == g) for g in range(len(pids))]
rng = np.random.default_rng(0); B = 1000
samples = [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]

def contrast(name, f):
    full = np.arange(len(pid0)); d0, per = f(full)
    bs = np.array([f(smp)[0] for smp in samples]); lo_, hi_ = np.percentile(bs, [2.5, 97.5])
    print('## %s delta %+.4f [%+.4f, %+.4f] per-seed %s' % (name, d0, lo_, hi_, ' '.join('%+.4f' % v for v in per)), flush=True)
    return dict(contrast=name, delta=d0, lo=lo_, hi=hi_, per_seed=';'.join('%.5f' % v for v in per))

def mk(a_fn, b_fn):
    def f(smp):
        per = []
        for s in range(3):
            y = res[s]['aug']['y'][smp]
            per.append(a_fn(s, smp, y) - b_fn(s, smp, y))
        return float(np.mean(per)), per
    return f

A = lambda key, cfg: (lambda s, smp, y: macro_auc(y, res[s][cfg][key][smp]))
M = lambda kind, k, cfg: (lambda s, smp, y: np.mean([macro_auc(y, res[s][cfg][(kind, k, q)][smp]) for q in range(R)]))
rows = []
rows.append(contrast('clean: proposed - aug', mk(A('clean', 'prop'), A('clean', 'aug'))))
rows.append(contrast('B1-scale: corrupted, proposed - aug', mk(A('corr', 'prop'), A('corr', 'aug'))))
rows.append(contrast('oracle repair, proposed: post - pre', mk(A('rep_oracle', 'prop'), A('corr', 'prop'))))
rows.append(contrast('dropped-only repair, proposed: post - pre', mk(A('rep_drop', 'prop'), A('corr', 'prop'))))
for k in KS:
    rows.append(contrast(f'k={k}: proposed - aug', mk(M('miss', k, 'prop'), M('miss', k, 'aug'))))
    rows.append(contrast(f'k={k}: repaired proposed - aug', mk(M('missrep', k, 'prop'), M('miss', k, 'aug'))))
for cfg in ('aug', 'prop'):
    print('## means', cfg, 'clean %.4f corr %.4f' % (np.mean([macro_auc(res[s][cfg]['y'], res[s][cfg]['clean']) for s in range(3)]),
          np.mean([macro_auc(res[s][cfg]['y'], res[s][cfg]['corr']) for s in range(3)])),
          ' '.join('k%d %.4f' % (k, np.mean([macro_auc(res[s][cfg]['y'], res[s][cfg][('miss', k, q)]) for s in range(3) for q in range(R)])) for k in KS), flush=True)
with open(out_dir / 'T_scale_check.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrote', out_dir / 'T_scale_check.csv', flush=True)
