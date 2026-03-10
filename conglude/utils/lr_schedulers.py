import torch
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, _LRScheduler

   

class PlateauWithWarmup(_LRScheduler):
    """
    Learning rate scheduler that combines linear warmup with ReduceLROnPlateau.

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        Optimizer whose learning rate will be scheduled.
    warmup_steps: int
        Number of initial steps during which LR increases linearly.
    factor: float
        Multiplicative factor for LR reduction after plateau.
    patience: int
        Number of epochs with no improvement before reducing LR.
    min_lr: float
        Lower bound on the learning rate.
    mode: str
        Whether to reduce LR when a metric has stopped decreasing ('min') or increasing ('max').
    last_epoch: int
        The index of the last epoch. Default: -1.
    """

    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        warmup_steps: int = 10, 
        factor: float = 0.1, 
        patience: int = 30, 
        min_lr: float = 1e-6, 
        mode: str = "max", 
        last_epoch: int = -1
    ) -> None:
                
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # ReduceLROnPlateau scheduler (used after warmup)
        self.plateau = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr)

        # Initialize base _LRScheduler
        super().__init__(optimizer, last_epoch)


    def get_lr(
        self
    ) -> list[float]:
        """
        Compute learning rate for the current step.
        During warmup: LR increases linearly from 0 to base_lr.
        After warmup: LR is controlled by ReduceLROnPlateau.
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup scaling factor
            scale = float(self.current_step + 1) / float(max(1, self.warmup_steps))
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # After warmup, lr is controlled by ReduceLROnPlateau
            return [group['lr'] for group in self.optimizer.param_groups]


    def step(
        self, 
        metrics: float = None
    ) -> None:
        
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # During warmup, use standard _LRScheduler step logic
            return super().step()
        else:
            # After warmup, delegate scheduling to ReduceLROnPlateau
            if metrics is None:
                raise ValueError("Validation metric (e.g. val_loss) must be provided after warmup.")
            self.plateau.step(metrics)



class CosineWithWarmup(LambdaLR):
    """
    Learning rate scheduler with linear warmup followed by cosine decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_steps : int
        Number of steps for linear warmup.
    total_steps : int
        Total number of training steps (including warmup).
    """

    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        warmup_steps: int = 10, 
        total_steps: int = 500,
    ) -> None:

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        def lr_lambda(
            current_step: int
        ) -> float:
            """
            Compute multiplicative LR factor for a given step.

            During warmup: scale linearly from 0 to 1
            After warmup: apply cosine decay from 1 to 0
            """

            # Linear warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine decay phase
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))
    
        # Initialize LambdaLR with the defined schedule
        super().__init__(optimizer, lr_lambda)