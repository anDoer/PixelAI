from typing import Tuple, Optional, List

class RuntimeConfig: 
    
    # dataset configuration
    default_train_image_size: Tuple[int, int] = (24, 28)
    bucketing_resolutions: List[Tuple[int, int]] = [
        (24, 28), 
        (32, 32), 
        (48, 48), 
        (64, 64), 
        (96, 96),
        (128, 128)]
    enable_bucketing: bool = False

    # training config 
    train_batch_size: int = 8 
    num_epochs: int = 100
    max_num_samples: Optional[int] = None  # total number of samples to train on
                                           # if set, overrides the num epochs
    num_workers: int = 0
    gradient_accumulation_steps: int = 1
    mixed_precision: str = 'bf16'  # 'no', 'fp16', 'bf16'
    seed: int = 42

    learning_rate: float = 1e-4
    optimizer: str = 'adamw'  # 'adamw'
    loss_weighting_scheme: str = 'none' # 'sigma_squared', 'none'

    # lr scheduler
    num_warmup_steps: int = 1000
    scheduler_name: str = 'linear'  # 'linear', 'cosine', 'constant'
    num_scheduler_cycles: Optional[int] = None  # None for no cycles, or an integer for number of cycles

    # i/o 
    dataset_path: str = '~/Data/Datasets/AIPixel'
    save_path: str = 'output/train/'
    logging_dir: str = 'logs'

    # model config 
    patch_size: int = 1
    in_channels: int = 3
    num_layers: int = 12
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56) # the sum is supposed to be 128 to be attention_head_dim! 

    # optimizer config 
    # AdamW
    adamw_betas: Tuple[float, float] = (0.9, 0.999)
    adamw_weight_decay: float = 1e-2
    adamw_eps: float = 1e-8
    
    # Flow Matching Scheduler
    num_train_timesteps: int = 1000
    scale_shift: float = 1.0  # for FlowMatchEulerDiscreteScheduler
    # parameters for logit normal sampling
    sampling_logit_mean: float = 0.0
    sampling_logit_std: float = 1.0  # std for logit normal sampling