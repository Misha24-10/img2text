import os
import json
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from src.config import ModelUpdate
from src.model import feature_extractor, text_tokenizer


class CocoDataset(Dataset):
    def __init__(self, coco_dir, annotation_file, split="train", column_anotation="caption"):
        self.column_anotation = column_anotation
        if split not in ["train", "val"]:
            raise ValueError("Split must be 'train' or 'val'")

        self.coco_dir = coco_dir
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)['annotations']
        self.image_dir = os.path.join(coco_dir, f'{split}2017')

        temp = pd.DataFrame(self.data).groupby("image_id").agg(
            {'id': lambda x: list(x), self.column_anotation: lambda x: list(x)}
        ).reset_index()

        self.processed_data = [vals.to_dict() for _, vals in temp.iterrows()]
        
    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        ann = self.processed_data[idx]
        image_path = os.path.join(self.image_dir, f'{ann["image_id"]:012d}.jpg')
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        features = feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze()
        annotation = ann[self.column_anotation][np.random.randint(len(ann[self.column_anotation]))]
        text_tokens = text_tokenizer(annotation, padding="max_length", max_length=ModelUpdate.max_length, truncation=True)

        return {
            "labels": text_tokens["input_ids"],
            "pixel_values": features,
            "gt_annotations": ann[self.column_anotation],
        }

class CustomCollator:
    def __call__(self, features):
        labels = torch.tensor([f["labels"] for f in features])
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        gt_annotations = [f["gt_annotations"] for f in features]

        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'gt_annotations': gt_annotations
        }