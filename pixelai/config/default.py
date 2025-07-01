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
    batch_size: int = 4 
    num_workers: int = 0