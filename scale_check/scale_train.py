"""Train one width-scaled configuration with the locked recipe (train.py unchanged; only build() is swapped)."""
import os, sys
sys.path.insert(0, '/home/spath004/ecg_lock'); sys.path.insert(0, '/home/spath004/xfer')
import train
from scale_common import wide_build
W = int(os.environ.get('ECG_WIDTH', '4'))
train.build = lambda arch, tasks, lead_head='pooled': wide_build(arch, tasks, lead_head, W)
if __name__ == '__main__':
    train.main()
