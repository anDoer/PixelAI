import torch

def default_collate(batch):

    images = list()
    image_sizes = list()
    original_sizes = list()
    image_offsets = list()

    for item in batch:
        images.append(item['image'])
        image_sizes.append(item['image_size'])
        original_sizes.append(item['original_size'])
        image_offsets.append(item['image_offsets'])

    output = {
        'images': torch.stack(images, dim=0),
        'image_sizes': torch.stack(image_sizes, dim=0),
        'original_sizes': torch.stack(original_sizes, dim=0),
        'image_offsets': torch.stack(image_offsets, dim=0),
    }
    return output