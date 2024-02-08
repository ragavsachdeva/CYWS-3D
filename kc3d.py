import os
import pickle

import imageio.v2 as imageio
import kornia as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from etils import epath
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
from torchvision.ops import masks_to_boxes

class KC3D(Dataset):
    def __init__(self, path_to_dataset, use_ground_truth_registration):
        self.path_to_dataset = path_to_dataset
        self.data = self.get_data_info(path_to_dataset)
        len_train = len(self.data["train"])
        len_val = len(self.data["val"])
        len_test = len(self.data["test"])
        self.use_ground_truth_registration = use_ground_truth_registration
        self.annotation_indices = np.arange(len_train + len_val, len_train + len_val + len_test)
        self.data = np.array(self.data["test"])

    def get_data_info(self, path_to_dataset):
        train_val_test_split_file_path = os.path.join(path_to_dataset, "data_split.pkl")
        if os.path.exists(train_val_test_split_file_path):
            with open(train_val_test_split_file_path, "rb") as file:
                return pickle.load(file)

    def read_image_as_tensor(self, path_to_image):
        """
        Returms a normalised RGB image as tensor.
        """
        with open(path_to_image, "rb") as file:
            pil_image = Image.open(file).convert("RGB")
            image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor.squeeze()

    def read_gt_depth_from_tiff(self, path_to_depth):
        """
        Returns the depth map as a float tensor.
        """
        filename = epath.Path(path_to_depth)
        img = imageio.imread(filename.read_bytes(), format="tiff")
        if img.ndim == 2:
            img = img[:, :, None]
        return K.image_to_tensor(img).float().squeeze()

    def __len__(self):
        """
        Returns the number of images.
        """
        return len(self.data)

    def get_target_annotation_mask(self, mask_path):
        with open(mask_path, "rb") as file:
            pil_image = Image.open(file)
            mask_as_np_array = np.array(pil_image)
        return K.image_to_tensor(mask_as_np_array).float()
    
    def get_target_bboxes_from_mask(self, mask_as_tensor):
        if len(mask_as_tensor.shape) == 2:
            mask_as_tensor = rearrange(mask_as_tensor, "h w -> 1 h w")
        bboxes = masks_to_boxes(mask_as_tensor)
        return bboxes

    def __getitem__(self, item_index):
        scene = self.data[item_index]
        image1_as_tensor = self.read_image_as_tensor(os.path.join(self.path_to_dataset, scene["image1"]))
        image2_as_tensor = self.read_image_as_tensor(os.path.join(self.path_to_dataset, scene["image2"]))
        depth1_as_tensor = self.read_gt_depth_from_tiff(os.path.join(self.path_to_dataset, scene["depth1"]))
        depth2_as_tensor = self.read_gt_depth_from_tiff(os.path.join(self.path_to_dataset, scene["depth2"]))

        target_mask_1 = self.get_target_annotation_mask(os.path.join(self.path_to_dataset, scene["mask1"]))
        target_mask_2 = self.get_target_annotation_mask(os.path.join(self.path_to_dataset, scene["mask2"]))
        target_bbox_1 = self.get_target_bboxes_from_mask(target_mask_1)
        target_bbox_2 = self.get_target_bboxes_from_mask(target_mask_2)

        data = {
            "image1": image1_as_tensor,
            "image2": image2_as_tensor,
            "depth1": depth1_as_tensor,
            "depth2": depth2_as_tensor,
            "registration_strategy": "3d",
            "target1": target_bbox_1.tolist(),
            "target2": target_bbox_2.tolist(),
        }

        if self.use_ground_truth_registration:
            metadata = np.load(
                os.path.join(
                    self.path_to_dataset,
                    "_".join(scene["image1"].split(".")[0].split("_")[:3]) + ".npy",
                ),
                allow_pickle=True,
            ).item()

            for key, value in metadata.items():
                if key == "intrinsics":
                    data["intrinsics1"] = torch.Tensor(value)
                    data["intrinsics2"] = torch.Tensor(value)
                elif key == "position_before":
                    data["position1"] = torch.Tensor(value)
                elif key == "position_after":
                    data["position2"] = torch.Tensor(value)
                elif key == "rotation_before":
                    data["rotation1"] = torch.Tensor(value)
                elif key == "rotation_after":
                    data["rotation2"] = torch.Tensor(value)

        return data

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from modules.registeration_module import FeatureRegisterationModule
    dataset = KC3D(path_to_dataset="path_to_dataset", use_ground_truth_registration=True)
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
    dfrm = FeatureRegisterationModule(None)
    for batch in dataloader:
        image1 = batch["image1"]
        image2 = batch["image2"]
        image1_warped_onto_image2, image2_warped_onto_image1, _, _ = dfrm.register_3d_features(batch, image1, image2)
        figure, subplots = plt.subplots(1, 4)
        subplots[0].imshow(K.tensor_to_image((image1 * 255)).astype(np.uint8))
        subplots[1].imshow(K.tensor_to_image((image2 * 255)).astype(np.uint8))
        subplots[2].imshow(K.tensor_to_image((image1_warped_onto_image2 * 255)).astype(np.uint8))
        subplots[3].imshow(K.tensor_to_image((image2_warped_onto_image1 * 255)).astype(np.uint8))
        plt.savefig("kc3d_check.png", bbox_inches="tight")
        break