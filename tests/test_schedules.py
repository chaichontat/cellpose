import numpy as np
from cellpose.schedules import build_lr_schedule


def test_cosine_schedule_basic():
    n_epochs = 20
    base_lr = 0.01
    warmup = 2
    lr = build_lr_schedule(n_epochs, base_lr, schedule="cosine", warmup_epochs=warmup, hold_epochs=5, min_lr=1e-4)
    assert lr.shape == (n_epochs,)
    # warmup starts at 0 and increases
    assert lr[0] >= 0.0 and lr[warmup - 1] < base_lr
    # after warmup+hold, lr decreases towards min_lr
    assert lr[warmup] <= base_lr and lr[-1] <= lr[warmup]
    assert np.all(lr >= 0.0)


def test_step_schedule_len_and_halvings():
    # choose long schedule to trigger 5 halvings of 10 epochs each
    n_epochs = 400
    base_lr = 0.01
    lr = build_lr_schedule(n_epochs, base_lr, schedule="step", warmup_epochs=10)
    assert lr.shape == (n_epochs,)
    # last 50 epochs should show five plateau blocks with halving between blocks
    tail = lr[-50:]
    # sample block boundaries (10-epoch blocks)
    b = [tail[i * 10] for i in range(5)]
    # strictly decreasing by ~factor 2 each block
    assert b[0] > b[1] > b[2] > b[3] > b[4] >= 0.0
