import numpy as np


def build_lr_schedule(
    n_epochs: int,
    base_lr: float,
    schedule: str = "cosine",
    warmup_epochs: int = 10,
    hold_epochs: int | None = None,
    min_lr: float | None = None,
) -> np.ndarray:
    """
    Build an epoch-wise learning rate schedule.

    - cosine: linear warmup, then cosine decay to 0.
    - step: existing Cellpose style (warmup + flat + late halvings), with five halvings.

    Args:
        n_epochs: total epochs.
        base_lr: peak learning rate.
        schedule: 'cosine' or 'step'.
        warmup_epochs: linear warmup length (epochs).

    Returns:
        np.ndarray of shape (n_epochs,)
    """
    n_epochs = int(max(0, n_epochs))
    warm = int(max(0, min(warmup_epochs, n_epochs)))

    if n_epochs == 0:
        return np.zeros((0,), dtype=float)

    if schedule == "cosine":
        # default floor is 1% of base LR
        if min_lr is None:
            min_lr = 0.01 * float(base_lr)
        if warm == n_epochs:
            return np.linspace(0.0, base_lr, n_epochs, dtype=float)
        # Default hold: keep peak LR for ~50% of epochs if not specified
        if hold_epochs is None:
            hold = int(max(0, round(0.5 * n_epochs) - warm))
        else:
            hold = int(max(0, min(hold_epochs, n_epochs - warm)))
        remain = n_epochs - warm - hold
        warmup = np.linspace(0.0, base_lr, warm, endpoint=False, dtype=float)
        hold_arr = base_lr * np.ones(hold, dtype=float)
        if remain <= 0:
            return np.concatenate([warmup, hold_arr])
        # cosine from base_lr -> min_lr over 'remain' epochs
        # y = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * t))
        cos_part = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, remain)))
        decay = min_lr + (base_lr - min_lr) * cos_part
        return np.concatenate([warmup, hold_arr, decay])

    if schedule == "step":
        # replicate existing behavior but with five halvings
        LR = np.linspace(0.0, base_lr, min(10, n_epochs), dtype=float)
        if n_epochs > 10:
            LR = np.append(LR, base_lr * np.ones(n_epochs - 10, dtype=float))
        # For long runs, replace the tail with five halving blocks
        if n_epochs > 300:
            # five halvings over last 50 epochs (5x10)
            LR = LR[: n_epochs - 50]
            for _ in range(5):
                LR = np.append(LR, LR[-1] / 2.0 * np.ones(10, dtype=float))
        elif n_epochs > 100:
            # five halvings over last 25 epochs (5x5)
            LR = LR[: n_epochs - 25]
            for _ in range(5):
                LR = np.append(LR, LR[-1] / 2.0 * np.ones(5, dtype=float))
        return LR

    raise ValueError(f"unknown schedule '{schedule}'")
