from typing import TYPE_CHECKING
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

if TYPE_CHECKING:
    import lightning.pytorch as pl


class EarlyStoppingWithWarmup(EarlyStopping):
    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: float | None = None,
        divergence_threshold: float | None = None,
        check_on_train_epoch_end: bool | None = None,
        log_rank_zero_only: bool = False,
        warmup_iters: int = 100,
        warmup_epochs: int = 3,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
            log_rank_zero_only=log_rank_zero_only,
        )
        # two thresholds to have invariant to extra small datasets and larger datasets
        self.warmup_iters = warmup_iters
        self.warmup_epochs = warmup_epochs

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from lightning.pytorch.trainer.states import TrainerFn
        current_iter = trainer.current_epoch * trainer.num_training_batches
        warmup_threshold = max(self.warmup_epochs * trainer.num_training_batches, self.warmup_iters)
        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking or current_iter < warmup_threshold
