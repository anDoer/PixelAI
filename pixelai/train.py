import torch
import os
import math
import logging
import argparse
import shutil

from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, set_seed

from diffusers.optimization import get_scheduler
from safetensors.torch import save_file, load_file

from pixelai.utils.logging import get_logger
from pixelai.models.pixel_model import PixelTransformer
from pixelai.datasets.data_reader import get_dataloader
from pixelai.config.default import RuntimeConfig
from pixelai.models.scheduler import FlowMatchEulerDiscreteScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="PixelAI Training Script")

    parser.add_argument('--config_path', type=str)

    return parser.parse_args()

def setup_accelerator(logging_dir: Path):
    accelerator_project_config = ProjectConfiguration(project_dir=RuntimeConfig.save_path,
                                                      logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=RuntimeConfig.gradient_accumulation_steps,
        mixed_precision=RuntimeConfig.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    return accelerator
    
def prepare_hooks(accelerator,
                  logger: logging.Logger):
    
    def save_model_hook(models, weights, output_dir):
        import pdb; pdb.set_trace()
        if accelerator.is_main_process:
            logger.info(f"Saving model state to {output_dir}...")

            # Save the state of the model
            for model in models:
                if isinstance(model, PixelTransformer):
                    # Save the model state using safetensors
                    save_file(accelerator.unwrap_model(model).state_dict(), Path(output_dir, 'transformer.safetensors'))
                else:
                    logger.warning(f"Model {model} is not a PixelTransformer, skipping save.")

                weights.pop()
    
    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # remove from list so models are not loaded again
            model = models.pop()

            if isinstance(model, PixelTransformer):
                state_dict = load_file(Path(input_dir, 'transformer.safetensors'))
                model.load_state_dict(state_dict)

                del state_dict

    logger.info("Registering save and load state hooks for the model...")
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

def initialize_model(logger: logging.Logger,
                     accelerator: Accelerator,
                     weight_dtype) -> PixelTransformer:

    logger.info(f"Setting up PixelTransformer model...")
    model = PixelTransformer(
        patch_size=RuntimeConfig.patch_size,
        in_channels=RuntimeConfig.in_channels,
        num_layers=RuntimeConfig.num_layers,
        attention_head_dim=RuntimeConfig.attention_head_dim,
        num_attention_heads=RuntimeConfig.num_attention_heads,
        axes_dims_rope=RuntimeConfig.axes_dims_rope
    )

    logger.info(f"Using weight dtype: {weight_dtype}")
    model = model.to(accelerator.device, dtype=weight_dtype)

    return model

def prepare_optimizer(transformer: PixelTransformer,
                      logger: logging.Logger):

    logger.info("Preparing optimizer...")
    transformer_params = {"params": transformer.parameters(), 
                          "lr": RuntimeConfig.learning_rate,}

    parameters_to_optimize = [transformer_params]

    optimizer = torch.optim.AdamW(
        parameters_to_optimize,
        betas=RuntimeConfig.adamw_betas,
        weight_decay=RuntimeConfig.adamw_weight_decay,
        eps= RuntimeConfig.adamw_eps,
    )

    return optimizer

def get_lr_scheduler(optimizer,
                     logger: logging.Logger,
                     accelerator: Accelerator,
                     dataloader):
    
    logger.info("Preparing scheduler...")

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = RuntimeConfig.num_warmup_steps * accelerator.num_processes
    if RuntimeConfig.max_num_samples is None:
        len_train_dataloader_after_sharding = math.ceil(len(dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / RuntimeConfig.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            RuntimeConfig.num_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = RuntimeConfig.max_num_samples * accelerator.num_processes

    scheduler = get_scheduler(
        name=RuntimeConfig.scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=RuntimeConfig.num_warmup_steps,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=RuntimeConfig.num_scheduler_cycles,  # No cycles for now
    ) 

    return scheduler

def get_sigmas_for_training(scheduler: FlowMatchEulerDiscreteScheduler,
                            timesteps: torch.Tensor,
                            n_dim: int = 4):

    scheduler_sigmas = scheduler.sigmas
    scheduler_timesteps = scheduler.timesteps
    timesteps = timesteps

    # obtain the indices of sampled timesteps
    indices = [(scheduler_timesteps == t).nonzero().item() for t in timesteps]

    # obtain the respective sigmas 
    sigmas = scheduler_sigmas[indices].flatten()
    while len(sigmas.shape) < n_dim:
        sigmas = sigmas.unsqueeze(-1)

    return sigmas

def sample_timesteps_logit_normal_density(logit_mean: float,
                                          logit_std: float,
                                          batch_size: int,
                                          generator: torch.Generator,
                                          noise_scheduler: FlowMatchEulerDiscreteScheduler,
                                          ):

    # sample u from a logit-normal distribution
    u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), generator=generator)
    u = torch.nn.functional.sigmoid(u)

    # obtain the indices of sampled timesteps
    timestep_indices = (u * noise_scheduler.num_train_timesteps).long()
    sampled_timesteps = noise_scheduler.timesteps[timestep_indices]

    return sampled_timesteps

def compute_loss(model_pred: torch.Tensor,
                 target: torch.Tensor,
                 sigmas: torch.Tensor):
    
    # compute loss weighting
    if RuntimeConfig.loss_weighting_scheme == 'sigma_squared':
        loss_weight = (sigmas ** -2).float()
    else:
        loss_weight = torch.ones_like(sigmas)

    loss = loss_weight * (model_pred.float() - target.float()) ** 2
    loss = loss.reshape(loss.shape[0], -1).mean(dim=1).mean()

    return loss
    
def run_train():
    args = parse_args() 

    # TODO: load config from file and override RuntimeConfig

    logging_dir = Path(RuntimeConfig.save_path, RuntimeConfig.logging_dir)
    accelerator = setup_accelerator(logging_dir)

    if accelerator.is_main_process:
        logging_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__, 
                        log_level=logging.DEBUG, 
                        log_to_file=True, 
                        log_file_path=f"{logging_dir}/pixelai_train.log")

    logger.info(accelerator.state)
    logger.info(f"Setting runtime seed to {RuntimeConfig.seed}")
    set_seed(RuntimeConfig.seed)
    
    # determine the weight dtype based on mixed precision setting
    weight_dtype = torch.float32
    if RuntimeConfig.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif RuntimeConfig.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16

    # prepare hooks for saving and loading model state
    prepare_hooks(accelerator,
                  logger=logger)

    # initialize the transformer model
    transformer = initialize_model(logger, accelerator, weight_dtype=weight_dtype)

    # Prepare Optimizer
    optimizer = prepare_optimizer(transformer, logger)

    # Prepare Dataset
    logger.info("Preparing dataset...")
    dataloader = get_dataloader()

    # Prepare lr scheduler
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, accelerator=accelerator, logger=logger, dataloader=dataloader)

    # prepare for training 
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    # Recalculate the total numer of training steps
    num_steps_per_epoch = math.ceil(len(dataloader) / RuntimeConfig.gradient_accumulation_steps)
    if RuntimeConfig.max_num_samples is None:
        RuntimeConfig.max_num_samples = RuntimeConfig.num_epochs * num_steps_per_epoch

    RuntimeConfig.num_epochs = math.ceil(RuntimeConfig.max_num_samples / num_steps_per_epoch)

    total_batch_size = (RuntimeConfig.train_batch_size *
                        accelerator.num_processes *
                        RuntimeConfig.gradient_accumulation_steps) 

    logger.info("****** Running training ******")
    logger.info(f"  Num Epochs = {RuntimeConfig.num_epochs}")
    logger.info(f"  Num Steps per Epoch = {num_steps_per_epoch}")
    logger.info(f"  Total Batch Size = {total_batch_size}")
    logger.info(f"  Max Num Samples = {RuntimeConfig.max_num_samples}")
    logger.info(f"  Gradient Accumulation Steps = {RuntimeConfig.gradient_accumulation_steps}")
    logger.info("******************************")

    global_step = 0
    first_epoch = 0

    if RuntimeConfig.resume_from is not None:
        if RuntimeConfig.resume_from != 'last':  
            ckpt = Path(RuntimeConfig.resume_from)
        else:
            existing_ckpts = sorted(Path(RuntimeConfig.save_path, 'checkpoints').glob('checkpoint-*'))
            ckpt = Path('last') if len(existing_ckpts) == 0 else existing_ckpts[-1]

        ckpt_path = os.path.join(RuntimeConfig.save_path, 'checkpoints', ckpt.name)
        if not os.path.exists(ckpt_path):

            logger.warning(f"Checkpoint {ckpt_path} does not exist, starting from scratch.")
            global_step = 0
        else:
            logger.info(f"Loading checkpoint from {ckpt_path}...")
            accelerator.load_state(ckpt_path)
            global_step = int(ckpt.name.split('-')[1])
            first_epoch = global_step // num_steps_per_epoch

    initial_global_step = global_step
    
    progress_bar = tqdm(
        range(0, RuntimeConfig.max_num_samples),
        initial=initial_global_step,
        desc=f'Epoch [{first_epoch}/{RuntimeConfig.num_epochs}] Steps',
        disable=not accelerator.is_main_process,
    )

    # initialize the scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=RuntimeConfig.num_train_timesteps,
                                                shift=RuntimeConfig.scale_shift)

    generator = torch.Generator().manual_seed(RuntimeConfig.seed)

    for epoch in range(first_epoch, RuntimeConfig.num_epochs):
        transformer.train()

        progress_bar.set_description(f'Epoch [{epoch}/{RuntimeConfig.num_epochs}] Steps')

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(transformer):
                images = batch['images']

                image_sizes = batch['image_sizes'].to(dtype=weight_dtype)
                original_sizes = batch['original_sizes'].to(dtype=weight_dtype)
                image_offsets = batch['image_offsets'].to(dtype=weight_dtype)

                # TODO: encode this as additional condition similar to what has been done in SDXL
                #       or think about a better method!
                image_size_condition = torch.cat([image_sizes, original_sizes, image_offsets], dim=-1)

                # prepare the image ids
                image_ids = PixelTransformer._prepare_latent_image_ids(
                    height=images.shape[2],
                    width=images.shape[3],
                    device=accelerator.device,
                    dtype=weight_dtype
                )

                # sample random noise
                noise = torch.randn(images.shape, generator=generator)

                # sample random timesteps 
                timesteps = sample_timesteps_logit_normal_density(
                   logit_mean=RuntimeConfig.sampling_logit_mean,
                   logit_std=RuntimeConfig.sampling_logit_std,
                   batch_size=images.shape[0],
                   generator=generator, 
                   noise_scheduler=scheduler,
                )

                # obtain corresponding sigmas
                sigmas = get_sigmas_for_training(
                    scheduler=scheduler,
                    timesteps=timesteps,
                    n_dim=images.ndim
                )

                # move to gpu
                sigmas = sigmas.to(accelerator.device)
                noise = noise.to(accelerator.device)

                # add noie following flow matching
                noisy_input = ( 1.0 - sigmas) * images + sigmas * noise
                noisy_input = noisy_input.to(dtype=weight_dtype)
                
                # flow matching target
                target = noise - images
                
                # pack
                noisy_input_packed = PixelTransformer._pack_latents(
                    latents=noisy_input,
                    patch_size=RuntimeConfig.patch_size
                )

                model_pred = transformer(
                    hidden_states=noisy_input_packed,
                    timestep=timesteps.to(accelerator.device),
                    img_ids=image_ids)

                # unpack
                model_pred = PixelTransformer._unpack_latents(
                    latents=model_pred,
                    height=images.shape[2],
                    width=images.shape[3],
                    patch_size=RuntimeConfig.patch_size
                )

                loss = compute_loss(
                    model_pred=model_pred,
                    target=target,
                    sigmas=sigmas)

                accelerator.backward(loss)

                if accelerator.sync_gradients and RuntimeConfig.gradient_clipping is not None:
                    accelerator.clip_grad_norm_(transformer.parameters(), 
                                                max_norm=RuntimeConfig.gradient_clipping)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % RuntimeConfig.checkpointing_interval == 0:
                        if RuntimeConfig.checkpoint_total_limit is not None:
                            checkpoint_dirs = Path(RuntimeConfig.save_path, 'checkpoints').glob('checkpoint-*')
                            checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.name.split('-')[1]))

                            if len(checkpoint_dirs) >= RuntimeConfig.checkpoint_total_limit:
                                num_to_remove = len(checkpoint_dirs) - RuntimeConfig.checkpoint_total_limit + 1
                                ckpts_to_remove = checkpoint_dirs[:num_to_remove]

                                logger.info(f"{len(checkpoint_dirs)} checkpoints found, removing {num_to_remove} oldest checkpoints.")
                                logger.info(f"removed checkpoints: {', '.join(ckpts_to_remove)}")

                                for ckpt_path in ckpts_to_remove:
                                    shutil.rmtree(ckpt_path)
                        
                        save_path = Path(RuntimeConfig.save_path, 'checkpoints', f'checkpoint-{global_step}')
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")


            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= RuntimeConfig.max_num_samples:
                logger.info("Reached max number of samples, stopping training.")
                break

        # we finished an epoch, we can run validation now
        if accelerator.is_main_process:
            # TODO: implement and run the validation pipeline!
            pass

    logger.info("Training finished.")

    # TODO: Save the final model
    # TODO: run inference

    accelerator.end_training()


    ## Debugging
    #test_input = torch.randn(1, 3, 24, 28).to(accelerator.device, dtype=weight_dtype)
    #test_timestep = torch.tensor([0], dtype=torch.long).to(accelerator.device)

    #image_ids = model._prepare_latent_image_ids(24, 28, 
    #                                            device=accelerator.device,
    #                                            dtype=weight_dtype)

    ## ToDO: Move inside model??
    #packed_input = model._pack_latents(test_input,
    #                                   patch_size=model.patch_size)

    #test = model(hidden_states=packed_input,
    #             timestep=test_timestep,
    #             img_ids=image_ids)
    #test = model._unpack_latents(test,
    #                             height=24,
    #                             width=28,
    #                             patch_size=model.patch_size)

    #import pdb; pdb.set_trace()



if __name__ == '__main__':
    run_train()