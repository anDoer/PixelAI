import numpy as np
import torch
import torchvision.transforms as transforms
import pytest

from PIL import Image
from pixelai.models.scheduler import FlowMatchEulerDiscreteScheduler

@pytest.fixture
def test_image():
    test_image = Image.open('data/science.png').convert('RGB')  

    return test_image

@pytest.fixture
def transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])


def test_sample_image(test_image, transform):
    image_tensor = transform(test_image).unsqueeze(0)
    noise = torch.randn_like(image_tensor)

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1
    )

    # set the scheduler to the inference steps
    scheduler.set_timesteps(num_inference_steps=50)
    scheduler.set_begin_index(1)

    begin_index = 1
    vt = noise - image_tensor
    
    # add noise
    sample = (1 - scheduler.sigmas[begin_index]) * image_tensor + scheduler.sigmas[begin_index] * noise

    # denoising 
    sigmas = scheduler.sigmas

    for i in range(1, len(scheduler.timesteps)):
        sigma = sigmas[begin_index]
        sigma_prev = sigmas[begin_index + 1]

        dt = (sigma_prev - sigma)
        sample = sample + dt * vt
        begin_index += 1

    sample = sample.clamp(-1, 1)  # Clamp the output to valid image range

    assert torch.allclose(sample, image_tensor, atol=1e-5), "Sampled image does not match the original image within tolerance."