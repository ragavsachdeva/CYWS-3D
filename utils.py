import kornia as K
from etils import epath
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import imageio.v2 as imageio
import torch
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch

def create_batch_from_metadata(metadata):
    list_of_items = metadata["batch"]
    batch = {}
    all_keys = [
        "image1",
        "image2",
        "depth1",
        "depth2",
        "intrinsics1",
        "intrinsics2",
        "position1",
        "position2",
        "rotation1",
        "rotation2",
        "transfm2d_1_to_2",
        "transfm2d_2_to_1",
        "registration_strategy"
    ]
    for key in all_keys:
        batch[key] = []
    for item in list_of_items:
        for key in all_keys:
            value = item.get(key, None)
            if value is None:
                batch[key].append(None)
                continue
            if "image" in key:
                value = read_image_as_tensor(value)
            if "depth" in key:
                value = read_depth_as_tensor(value)
            for k in ["position","rotation","intrinsics","transfm2d"]:
                if k in key:
                    value = torch.tensor(np.load(value))
            batch[key].append(value)
    _sanity_test_batch(batch, list_of_items)
    return batch

def read_image_as_tensor(path_to_image):
    assert path_to_image is not None
    with open(path_to_image, "rb") as file:
        pil_image = Image.open(file).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
    return image_as_tensor

def read_depth_as_tensor(path_to_depth):
    assert path_to_depth is not None
    if ".tiff" in path_to_depth:
        return _read_depth_from_tiff(path_to_depth)
    return _read_depth_from_png(path_to_depth)

@torch.no_grad()
def fill_in_the_missing_information(batch, depth_predictor, correspondence_extractor):
    for i in range(len(batch["image1"])):
        if batch["registration_strategy"][i] == "3d":
            assert (batch["depth1"][i] is None) == (batch["depth2"][i] is None)
            if batch["depth1"][i] is None and batch["depth2"][i] is None:
                batch["depth1"][i] = depth_predictor.infer(batch["image1"][i].unsqueeze(0)).squeeze()
                batch["depth2"][i] = depth_predictor.infer(batch["image2"][i].unsqueeze(0)).squeeze()
    batch = correspondence_extractor(batch)
    return batch

def prepare_batch_for_model(batch):
    nearest_resize = K.augmentation.Resize((224,224), resample=0, align_corners=None, keepdim=True)
    bicubic_resize = K.augmentation.Resize((224,224), resample=2, keepdim=True)
    for i in range(len(batch["image1"])):
        original_hw1 = batch["image1"][i].shape[-2:]
        original_hw2 = batch["image2"][i].shape[-2:]
        batch["image1"][i] = bicubic_resize(normalise_image(batch["image1"][i]))
        batch["image2"][i] = bicubic_resize(normalise_image(batch["image2"][i]))
        if batch["depth1"][i] is not None:
            original_depth_hw1 = batch["depth1"][i].shape[-2:]
            batch["depth1"][i] = nearest_resize(batch["depth1"][i])
        if batch["depth2"][i] is not None:
            original_depth_hw2 = batch["depth2"][i].shape[-2:]
            batch["depth2"][i] = nearest_resize(batch["depth2"][i])
        if batch["intrinsics1"][i] is not None:
            assert original_hw1 == original_depth_hw1
            transformation = nearest_resize.transform_matrix.squeeze()
            transformation = convert_kornia_transformation_matrix_to_normalised_coordinates(transformation, original_hw1, (224, 224))
            batch["intrinsics1"][i] = transformation @ batch["intrinsics1"][i]
        if batch["intrinsics2"][i] is not None:
            assert original_hw2 == original_depth_hw2
            transformation = nearest_resize.transform_matrix.squeeze()
            transformation = convert_kornia_transformation_matrix_to_normalised_coordinates(transformation, original_hw2, (224, 224))
            batch["intrinsics2"][i] = transformation @ batch["intrinsics2"][i]
    for keys in ["image1", "image2"]:
        batch[keys] = torch.stack(batch[keys])
    batch["query_metadata"] = [
        {
            "pad_shape": (224, 224),
            "border": np.array([0, 0, 0, 0]),
            "batch_input_shape": (224, 224),
        }
    ] * len(batch["image1"])
    return batch

def normalise_image(img_as_tensor):
    imagenet_normalisation = K.enhance.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = rearrange(img_as_tensor, "c h w -> 1 c h w")
    img = imagenet_normalisation(img)
    return img.squeeze()

def undo_imagenet_normalization(image_as_tensor):
    """
    Undo the imagenet normalization.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_as_tensor = image_as_tensor * std + mean
    return image_as_tensor

def convert_kornia_transformation_matrix_to_normalised_coordinates(matrix, original_hw, new_hw):
    scale_up = torch.Tensor([[original_hw[1], 0, 0], [0, original_hw[0], 0], [0, 0, 1]])
    scale_down = torch.Tensor([[1 / new_hw[1], 0, 0], [0, 1 / new_hw[0], 0], [0, 0, 1]])
    return scale_down @ matrix @ scale_up

def _read_depth_from_png(path_to_depth):
    with open(path_to_depth, "rb") as file:
        pil_image = Image.open(path_to_depth)
        image_as_tensor = pil_to_tensor(pil_image).float()
    if image_as_tensor.ndim == 3:
        image_as_tensor = image_as_tensor.squeeze(0)
    return image_as_tensor

def _read_depth_from_tiff(path_to_depth):
    filename = epath.Path(path_to_depth)
    img = imageio.imread(filename.read_bytes(), format="tiff")
    if img.ndim == 2:
        img = img[:, :, None]
    return K.image_to_tensor(img).float().squeeze()

def visualise_predictions(
    left_image,
    right_image,
    left_predicted_bboxes,
    right_predicted_bboxes,
    save_path="./results.png"
):
    TARGET_COLOUR = "#1E88E5"
    PREDICTED_COLOUR = "#FFC107"
    figure, plot = plt.subplots(1, 2)
    plot[0].imshow(K.tensor_to_image(left_image))
    for bbox in left_predicted_bboxes:
        bbox[[0,2]] = bbox[[0, 2]] * (left_image.shape[-1] / 224)
        bbox[[1,3]] = bbox[[1, 3]] * (left_image.shape[-2] / 224)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        rect = patches.Rectangle(
            bbox[:2], w, h, linewidth=2, edgecolor=PREDICTED_COLOUR, facecolor="none"
        )
        plot[0].add_patch(rect)
    plot[0].axis("off")

    plot[1].imshow(K.tensor_to_image(right_image))
    for bbox in right_predicted_bboxes:
        bbox[[0,2]] = bbox[[0, 2]] * (right_image.shape[-1] / 224)
        bbox[[1,3]] = bbox[[1, 3]] * (right_image.shape[-2] / 224)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        rect = patches.Rectangle(
            bbox[:2], w, h, linewidth=2, edgecolor=PREDICTED_COLOUR, facecolor="none"
        )
        plot[1].add_patch(rect)
    plot[1].axis("off")
    figure.savefig(save_path, bbox_inches="tight")

def plot_correspondences(source_image, target_image, source_points, target_points, save_path="./correspondences.png"):
    """
    Helper function to plot correspondences.
    """
    fig, axarr = plt.subplots(1,2)
    if torch.is_tensor(source_image):
        source_image = K.tensor_to_image(source_image)
    if torch.is_tensor(target_image):
        target_image = K.tensor_to_image(target_image)
    axarr[0].imshow(source_image)
    axarr[1].imshow(target_image)

    source_points = source_points * torch.tensor([source_image.shape[1], source_image.shape[0]])
    target_points = target_points * torch.tensor([target_image.shape[1], target_image.shape[0]])

    for i, (pt_q, pt_t) in enumerate(zip(source_points, target_points)):
            col = (np.random.random(), np.random.random(), np.random.random())
            con = ConnectionPatch(pt_t, pt_q,
                                  coordsA='data', coordsB='data',
                                  axesA=axarr[1], axesB=axarr[0],
                                  color='g', linewidth=0.5)
            axarr[1].add_artist(con)
            axarr[0].plot(pt_q[0], pt_q[1], c=col, marker='x')
            axarr[1].plot(pt_t[0], pt_t[1], c=col, marker='x')
    plt.savefig(save_path, bbox_inches="tight")

def _sanity_test_batch(batch, list_of_items):
    keys_and_their_existance = [
        {
            "keys": ["image1", "image2", "registration_strategy"],
            "possible_values": [
                [True, True, True]
            ],
        },
        {
            "keys": ["depth1", "depth2", "intrinsics1", "intrinsics2", "position1", "position2", "rotation1", "rotation2"],
            "possible_values": [
                [True, True, True, True, True, True, True, True],
                [False, False, False, False, False, False, False, False],
                [True, True, False, False, False, False, False, False],
            ]
        },
        {
            "keys": ["transfm2d_1_to_2", "transfm2d_2_to_1"],
            "possible_values": [
                [True, True],
                [False, False],
            ]
        }
    ]
    for i in range(len(list_of_items)):
        for dict in keys_and_their_existance:
            keys = dict["keys"]
            keys_exist = [batch[key][i] is not None for key in keys]
            assert keys_exist in dict["possible_values"]
