import json
import os

import kornia as K
import numpy as np
from einops import rearrange
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms.functional import pil_to_tensor


def RC3D(path_to_dataset, use_gt_depth):
    parts = ["part1", "part2", "part3", "part4"]
    datasets = [SubDataset(os.path.join(path_to_dataset, part), use_gt_depth) for part in parts]
    return ConcatDataset(datasets)


class SubDataset(Dataset):
    def __init__(self, path_to_dataset, use_gt_depth):
        self.path_to_dataset = path_to_dataset
        with open(os.path.join(self.path_to_dataset, "coco_annotations.json"), "r") as j:
            annotations_json = json.loads(j.read())
        self.image1 = [x["file_name"] for x in annotations_json["images"][0::3]]
        self.image2 = [x["file_name"] for x in annotations_json["images"][2::3]]
        self.annotations1 = [x["bbox"] for x in annotations_json["annotations"][0::3]]
        self.annotations2 = [x["bbox"] for x in annotations_json["annotations"][2::3]]
        self.use_gt_depth = use_gt_depth

    def read_image_as_tensor(self, path_to_image):
        """
        Returms a normalised RGB image as tensor.
        """
        with open(path_to_image, "rb") as file:
            pil_image = Image.open(file).convert("RGB")
            image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor.squeeze()
    
    def read_depth_as_tensor(self, path_to_image):
        """
        Returms a normalised RGB image as tensor.
        """
        with open(path_to_image, "rb") as file:
            pil_image = Image.open(file)
            image_as_tensor = pil_to_tensor(pil_image).float()
        return image_as_tensor.squeeze()

    def __len__(self):
        """
        Returns the number of testing images.
        """
        return len(self.image1)

    def __getitem__(self, item_index):
        image1_as_tensor = self.read_image_as_tensor(os.path.join(self.path_to_dataset, self.image1[item_index]))
        image2_as_tensor = self.read_image_as_tensor(os.path.join(self.path_to_dataset, self.image2[item_index]))
        target_bbox_1 = self.annotations1[item_index]
        target_bbox_1[2] += target_bbox_1[0]
        target_bbox_1[3] += target_bbox_1[1]
        target_bbox_2 = self.annotations2[item_index]
        target_bbox_2[2] += target_bbox_2[0]
        target_bbox_2[3] += target_bbox_2[1]

        data = {
            "image1": image1_as_tensor,
            "image2": image2_as_tensor,
            "registration_strategy": "3d",
            "target1": [target_bbox_1],
            "target2": [target_bbox_2],
        }

        if self.use_gt_depth:
            depth1 = self.read_depth_as_tensor(os.path.join(self.path_to_dataset, f"depth_{self.image1[item_index].replace('.jpg', '')}.png"))
            depth2 = self.read_depth_as_tensor(os.path.join(self.path_to_dataset, f"depth_{self.image2[item_index].replace('.jpg', '')}.png"))
            data["depth1"] = depth1
            data["depth2"] = depth2
        
        return data

def get_easy_dict_from_yaml_file(path_to_yaml_file):
    """
    Reads a yaml and returns it as an easy dict.
    """
    import yaml
    from easydict import EasyDict

    with open(path_to_yaml_file, "r") as stream:
        yaml_file = yaml.safe_load(stream)
    return EasyDict(yaml_file)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = RC3D(path_to_dataset="path", use_gt_depth=True)
    import matplotlib.pyplot as plt

    def collate_fn(batch):
        keys = batch[0].keys()
        collated_dictionary = {}
        for key in keys:
            collated_dictionary[key] = []
            for batch_item in batch:
                collated_dictionary[key].append(batch_item[key])
            if "target" in key or "registration" in key:
                continue
            collated_dictionary[key] = rearrange(collated_dictionary[key], "... -> ...")
        return collated_dictionary

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        image1 = batch["image1"]
        image2 = batch["image2"]
        figure, subplots = plt.subplots(1, 2)
        subplots[0].imshow(K.tensor_to_image((image1 * 255)).astype(np.uint8))
        subplots[1].imshow(K.tensor_to_image((image2 * 255)).astype(np.uint8))
        plt.savefig("rc3d_check.png", bbox_inches="tight")
        break