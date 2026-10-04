"""Width-scaled versions of the locked CNN models (post hoc scale check). The locked code is
imported unchanged; only the encoder, decoder and the heads that read the encoder are widened."""
import sys, torch.nn as nn
sys.path.insert(0, '/home/spath004/ecg_lock')
import models

def wide_build(arch, tasks, lead_head='pooled', W=4):
    assert arch == 'cnn' and lead_head in ('pooled', 'leadwise')
    m = models.MultiTask(arch, tasks, lead_head)
    c1, c2, c3 = 64 * W, 128 * W, 256 * W
    b = models.block
    m.enc.net = nn.Sequential(b(12, c1, 2), b(c1, c1, 1), b(c1, c2, 2), b(c2, c2, 1), b(c2, c3, 2), b(c3, c3, 1), nn.Dropout(0.15))
    m.diag = nn.Sequential(nn.Linear(c3, 128), nn.GELU(), nn.Linear(128, 5))
    if 'lead' in m.tasks and lead_head == 'pooled':
        m.lead = nn.Sequential(nn.Linear(c3, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 12))
    if 'rec' in m.tasks:
        m.dec.net = nn.Sequential(nn.ConvTranspose1d(c3, c2, 4, 2, 1), nn.GELU(), nn.ConvTranspose1d(c2, c1, 4, 2, 1), nn.GELU(),
                                  nn.ConvTranspose1d(c1, c1, 4, 2, 1), nn.GELU(), nn.Conv1d(c1, 12, 7, 1, 3))
    return m
