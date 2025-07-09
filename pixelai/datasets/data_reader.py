import os
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from pixelai.datasets.dataset_utils import cluster_image_sizes, plot_cluster_statistics
from pixelai.datasets.augmentation.augmentation_utils import resize_with_aspect_ratio
from pixelai.datasets.collate import default_collate

from pixelai.config.default import RuntimeConfig

class Dataset(Dataset):
    def __init__(self, 
                 data_path: str):
        self.data_path = Path(data_path)
        
        self.samples = list()
        self.data = dict()
        self.game_to_class_idx = dict()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.load_data()

    def load_data(self):

        game_folder_idx = 0
        for game_folder in self.data_path.expanduser().iterdir():
            if not game_folder.is_dir():
                continue

            game_name = game_folder.name
            if game_name not in self.data:
                self.data[game_name] = dict()
                self.game_to_class_idx[game_name] = game_folder_idx
                game_folder_idx += 1

            for char_folder in game_folder.iterdir():
                if not char_folder.is_dir():
                    continue

                char_name = char_folder.name
                if char_name not in self.data[game_name]:
                    self.data[game_name][char_name] = list()

                for img_file in char_folder.glob('*.png'):
                    sample = {
                        'game': game_name,
                        'character': char_name,
                        'image_path': img_file,
                        'data_idx': len(self.samples), # Unique index for the sample
                        'game_class_idx': self.game_to_class_idx[game_name],
                    }

                    self.samples.append(sample)
                    self.data[game_name][char_name].append(sample)
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # load image 
        image = Image.open(sample['image_path']).convert('RGB')
        resized_image, resize_meta = resize_with_aspect_ratio(image,
                                                              target_size=RuntimeConfig.default_train_image_size,
                                                              fill_value=(0, 0, 0))
        image_tensor = self.transform(resized_image)

        train_item = {
            'image': image_tensor,
            'image_size': torch.tensor(resize_meta['target_size']),
            'original_size': torch.tensor(resize_meta['original_size']),
            'image_offsets': torch.tensor(resize_meta['offsets']),
        }

        return train_item

    def _get_statistics(self, return_stats: bool = False) -> Tuple[List[int], Dict[int, List[int]], Dict[str, Dict]]:
        """
        Collect statistics about the dataset, specifically image sizes.
        Args:
            return_stats (bool): If True, return detailed statistics about the clusters.
        Returns:    
            Tuple containing:
                - List of representative sizes for each cluster
                - Mapping of cluster sizes to indices of images in those clusters
                - Statistics about the clusters if return_stats is True
        
        """ 
        stats = {
            "image_sizes": [],
        }

        for game, chars in self.data.items():
            print(f'Game: {game}')
            for char, samples in chars.items():
                for sample in samples:
                    img = Image.open(sample['image_path'])
                    width, height = img.size 
                    stats["image_sizes"].append((width, height))

        image_sizes = stats["image_sizes"]

        cluster_sizes, cluster_mapping, cluster_stats = cluster_image_sizes(
            image_sizes=image_sizes,
            n_bins=10,
            return_stats=return_stats
        ) 

        return cluster_sizes, cluster_mapping, cluster_stats
        
def get_dataloader():
    dataset = Dataset(data_path=RuntimeConfig.dataset_path)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=RuntimeConfig.train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=RuntimeConfig.num_workers,
        collate_fn=default_collate
    )

    return dataloader

if __name__ == '__main__':
    dataset = Dataset(data_path='/home/doering/Data/Datasets/AIPixel')

    _, _, stats = dataset._get_statistics(return_stats=True)
    plot_cluster_statistics(stats, 
                            output_folder='output/dataset_stats/', 
                            filename='cluster_statistics.png')

    #from torch.utils.data import DataLoader
    #dataloader = DataLoader(dataset, 
    #                        batch_size=RuntimeConfig.batch_size, 
    #                        shuffle=True,
    #                        pin_memory=True,
    #                        num_workers=RuntimeConfig.num_workers,
    #                        collate_fn=default_collate)