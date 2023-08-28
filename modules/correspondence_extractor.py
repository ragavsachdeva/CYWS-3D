import torch
import torch.nn as nn
import kornia as K
import numpy as np
from SuperGluePretrainedNetwork.models.matching import Matching
from modules.geometry import transform_points, convert_image_coordinates_to_world, sample_depth_for_given_points
import torch.nn.functional as F

class CorrespondenceExtractor(nn.Module):
    def __init__(self, nms_radius=4, keypoint_threshold=0.005, max_keypoints=1024, superglue="indoor", sinkhorn_iterations=20, match_threshold=0.2, resize=640):
        super().__init__()
        config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
        self._matching = Matching(config).eval()
        self._resize = K.augmentation.Resize(resize, side="long")

    @torch.no_grad()
    def forward(self, batch):
        batch_points1 = []
        batch_points2 = []
        for i in range(len(batch["image1"])):
            if batch["registration_strategy"][i] == "identity" or batch["intrinsics1"][i] is not None or batch["transfm2d_1_to_2"][i] is not None:
                # No need to compute correspondences
                batch_points1.append(None)
                batch_points2.append(None)
                continue
            inp1 = K.color.rgb_to_grayscale(batch["image1"][i]).unsqueeze(0)
            inp2 = K.color.rgb_to_grayscale(batch["image2"][i]).unsqueeze(0)
            inp1 = self._resize(inp1)
            inp2 = self._resize(inp2)
            pred = self._matching({'image0': inp1, 'image1': inp2})
            kpts1, kpts2 = pred['keypoints0'][0], pred['keypoints1'][0]
            matches, conf = pred['matches0'][0], pred['matching_scores0'][0]
            scale_1 = torch.tensor(inp1.shape[-2:]).flip(dims=(0,))
            scale_2 = torch.tensor(inp2.shape[-2:]).flip(dims=(0,))
            kpts1 /= scale_1
            kpts2 /= scale_2
            valid = matches != -1
            conf = conf[valid]
            kpts1 = kpts1[valid]
            kpts2 = kpts2[matches[valid]]
            conf, sort_idx = conf.sort(descending=True)
            kpts1 = kpts1[sort_idx]
            kpts2 = kpts2[sort_idx]
            kpts1, kpts2 = filter_out_bad_correspondences_using_ransac(batch["registration_strategy"][i], kpts1, kpts2, batch["depth1"][i], batch["depth2"][i])
            batch_points1.append(kpts1)
            batch_points2.append(kpts2)
        batch["points1"] = batch_points1
        batch["points2"] = batch_points2
        return batch


def inliers_using_ransac(X, Y, n_iters=500):
    best_inliers = None
    best_fit_error = None
    threshold = torch.median(torch.abs(Y - torch.median(Y)))
    for _ in range(n_iters):
        # estimate transformation
        sample_indices = np.random.choice(np.arange(X.shape[0]), size=min(50, X.shape[0]), replace=False)
        sample_X = X[sample_indices]
        sample_Y = Y[sample_indices]
        X_ = F.pad(sample_X, (0, 1), value=1)
        Y_ = F.pad(sample_Y, (0, 1), value=1)
        X_pinv = torch.linalg.pinv(X_)
        M = torch.einsum("ij,jk->ki", X_pinv, Y_)
        # find inliers
        X_warped = transform_points(M, X)
        fit_error = torch.sum(torch.abs(X_warped - Y), dim=1)
        inliers = (fit_error < threshold).nonzero().squeeze()
        fit_error_of_inliers = fit_error[inliers].sum()
        if (best_fit_error is None or fit_error_of_inliers < best_fit_error) and torch.numel(inliers) >= 10:
            best_fit_error = fit_error_of_inliers
            best_inliers = (fit_error < threshold).nonzero().squeeze()
    if best_inliers is None:
        return torch.ones(X.shape[0]).bool()
    return best_inliers


def filter_out_bad_correspondences_using_ransac(registration_strategy, points1, points2, depth1=None, depth2=None):
    if registration_strategy == "3d":
        assert depth1 is not None and depth2 is not None
        X = convert_image_coordinates_to_world(
            image_coords=points1.unsqueeze(0),
            depth=sample_depth_for_given_points(depth1.unsqueeze(0), points1.unsqueeze(0)),
            K_inv=torch.eye(3).type_as(points1).unsqueeze(0),
            Rt=torch.eye(4).type_as(points1).unsqueeze(0),
        ).squeeze(0)
        Y = convert_image_coordinates_to_world(
            image_coords=points2.unsqueeze(0),
            depth=sample_depth_for_given_points(depth2.unsqueeze(0), points2.unsqueeze(0)),
            K_inv=torch.eye(3).type_as(points2).unsqueeze(0),
            Rt=torch.eye(4).type_as(points2).unsqueeze(0),
        ).squeeze(0)
    elif registration_strategy == "2d":
        X = points1
        Y = points2
    else:
        raise NotImplementedError()
    inliers = inliers_using_ransac(X, Y)
    points1 = points1[inliers]
    points2 = points2[inliers]
    return points1, points2