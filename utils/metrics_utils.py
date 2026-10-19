#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import lpips
from utils.system_utils import save_ply
from skimage.metrics import structural_similarity
from chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from chamfer.fscore import fscore


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f"PSNR = {self.measure():.6f}"


class RMSEMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        rmse = (truths - preds) ** 2
        rmse = np.sqrt(rmse.mean())

        self.V += rmse
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "RMSE"), self.measure(), global_step)

    def report(self):
        return f"RMSE = {self.measure():.6f}"


class MAEMeter:
    def __init__(self, intensity_inv_scale=1.0):
        self.V = 0
        self.N = 0
        self.intensity_inv_scale = intensity_inv_scale

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # Mean Absolute Error
        mae = np.abs(
            truths * self.intensity_inv_scale - preds * self.intensity_inv_scale
        ).mean()

        self.V += mae
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "MAE"), self.measure(), global_step)

    def report(self):
        return f"MAE = {self.measure():.6f}"


class DepthMeter:
    def __init__(self, scale, lpips_fn=None):
        self.V = []
        self.N = 0
        self.scale = scale
        self.lpips_fn = lpips.LPIPS(net='alex').eval()

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths, valid_mask=None):
        preds = preds / self.scale
        truths = truths / self.scale
        if valid_mask is not None and torch.is_tensor(valid_mask):
            valid_mask = valid_mask.detach().cpu().numpy()
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        depth_error = self.compute_depth_errors(truths, preds, valid_mask=valid_mask)

        depth_error = list(depth_error)
        self.V.append(depth_error)
        self.N += 1

    def compute_depth_errors(
            self, gt, pred, min_depth=1e-6, max_depth=80, valid_mask=None,
    ):
        pred[pred < min_depth] = min_depth
        pred[pred > max_depth] = max_depth
        gt[gt < min_depth] = min_depth
        gt[gt > max_depth] = max_depth

        if valid_mask is not None:
            vm = valid_mask.astype(bool)
            if vm.shape != gt.shape:
                vm = np.broadcast_to(vm, gt.shape)
            if vm.any():
                rmse = np.sqrt(((gt - pred)[vm] ** 2).mean())
                medae = np.median(np.abs((gt - pred)[vm]))
            else:
                rmse = 0.0
                medae = 0.0
        else:
            rmse = (gt - pred) ** 2
            rmse = np.sqrt(rmse.mean())
            medae = np.median(np.abs(gt - pred))

        if gt.shape[-2] >= 32:
            lpips_loss = self.lpips_fn(torch.from_numpy(pred).squeeze(0),
                                       torch.from_numpy(gt).squeeze(0), normalize=True).item()
        else:
            lpips_loss = 1

        ssim = structural_similarity(
            pred.squeeze(0), gt.squeeze(0), data_range=np.max(gt) - np.min(gt)
        )

        if valid_mask is not None and vm.any():
            psnr = 10 * np.log10(max_depth ** 2 / np.mean(((pred - gt)[vm]) ** 2))
        else:
            psnr = 10 * np.log10(max_depth ** 2 / np.mean((pred - gt) ** 2))

        return rmse, medae, lpips_loss, ssim, psnr

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(
            os.path.join(prefix, f"depth error{suffix}"), self.measure()[0], global_step
        )

    def report(self):
        return f"Depth_error = {self.measure()}"


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def clear(self):
        self.V = 0
        self.N = 0

    # def prepare_inputs(self, *inputs):
    #     outputs = []
    #     for i, inp in enumerate(inputs):
    #         inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
    #         inp = inp.to(self.device)
    #         outputs.append(inp)
    #     return outputs

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)
        ssim = structural_similarity(
            preds.squeeze(0).squeeze(-1), truths.squeeze(0).squeeze(-1)
        )

        # preds, truths = self.prepare_inputs(
        #     preds, truths)  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        # ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f"SSIM = {self.measure():.6f}"


class PointsMeter:
    def __init__(self, scale, vfov, hfov=None):
        self.V = []
        self.N = 0
        self.scale = scale
        self.vfov = vfov
        # Stitched 360° default; pass args.hfov for cam_num=1 single-half RV.
        self.hfov = tuple(hfov) if hfov is not None else (-180, 180)
        self.near = 0.2
        self.far = 80

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def pano_to_lidar(self, range_image):
        # TODO: 确认下这行到底要不要
        range_image[range_image > self.far] = 0  # self.far
        panorama_height, panorama_width = range_image.shape[-2:]
        theta, phi = torch.meshgrid(torch.arange(panorama_height, device=range_image.device),
                                    torch.arange(panorama_width, device=range_image.device), indexing="ij")

        vertical_degree_range = self.vfov[1] - self.vfov[0]
        theta = (90 - self.vfov[1] + theta / panorama_height * vertical_degree_range) * torch.pi / 180

        horizontal_degree_range = self.hfov[1] - self.hfov[0]
        phi = (self.hfov[0] + phi / panorama_width * horizontal_degree_range) * torch.pi / 180

        dx = torch.sin(theta) * torch.sin(phi)
        dz = torch.sin(theta) * torch.cos(phi)
        dy = -torch.cos(theta)

        directions = torch.stack([dx, dy, dz], dim=0)
        directions = F.normalize(directions, dim=0)

        points_xyz = directions * range_image
        points_xyz = points_xyz.reshape(3, -1).permute(1, 0)
        points_xyz = points_xyz[points_xyz.norm(dim=1) > self.near]

        return points_xyz

    def _clip_points(self, points):
        points = points / self.scale
        norm = points.norm(dim=1)
        return points[(norm > self.near) & (norm <= self.far)]

    def update(self, preds, truths, pred_points=None, gt_points=None):
        """``pred_points`` / ``gt_points`` (N,3, scaled units) override the uniform
        pano back-projection, e.g. ring-RV slots whose rays are not on the grid."""
        # preds, truths = self.prepare_inputs(
        #     preds, truths
        # )  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        chamLoss = chamfer_3DDist()
        if pred_points is not None:
            pred_lidar = self._clip_points(pred_points)
        else:
            pred_lidar = self.pano_to_lidar(preds[0] / self.scale)
        if gt_points is not None:
            gt_lidar = self._clip_points(gt_points)
        else:
            gt_lidar = self.pano_to_lidar(truths[0] / self.scale)
        if len(pred_lidar) == 0 or len(gt_lidar) == 0:
            return
        # pred_lidar = self.discard_outliers(pred_lidar, 1.0)

        dist1, dist2, idx1, idx2 = chamLoss(pred_lidar[None, ...], gt_lidar[None, ...])
        chamfer_dis = dist1.mean() + dist2.mean()
        threshold = 0.05  # monoSDF
        f_score, precision, recall = fscore(dist1, dist2, threshold)
        f_score = f_score.cpu()[0]

        # save_ply(pred_lidar, "origin.ply")
        # drop_lidar = self.visualize_outliers(pred_lidar, dist1, 0.98)
        # save_ply(drop_lidar, "drop.ply")

        self.V.append([chamfer_dis.cpu(), f_score])

        self.N += 1

    @staticmethod
    def visualize_outliers(points, dist, threshold):
        dist_threshold = torch.quantile(dist, threshold)
        filtered_points = points[(dist <= dist_threshold)[0]]
        return filtered_points

    def measure(self):
        # return self.V / self.N
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "CD"), self.measure()[0], global_step)

    def report(self):
        return f"CD f-score = {self.measure()}"


class RaydropMeter:
    def __init__(self, ratio=0.5):
        self.V = []
        self.N = 0
        self.ratio = ratio

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths, valid_mask=None):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        if valid_mask is not None:
            if torch.is_tensor(valid_mask):
                valid_mask = valid_mask.detach().cpu().numpy()
            vm = valid_mask.astype(bool)
            if vm.shape != preds.shape:
                vm = np.broadcast_to(vm, preds.shape)
            preds = preds[vm]
            truths = truths[vm]
        results = []

        if preds.size == 0:
            self.V.append([0.0, 1.0, 1.0])
            self.N += 1
            return

        rmse = (truths - preds) ** 2
        rmse = np.sqrt(rmse.mean())
        results.append(rmse)

        preds_mask = np.where(preds > self.ratio, 1, 0)
        acc = (preds_mask == truths).mean()
        results.append(acc)

        TP = np.sum((truths == 1) & (preds_mask == 1))
        FP = np.sum((truths == 0) & (preds_mask == 1))
        TN = np.sum((truths == 0) & (preds_mask == 0))
        FN = np.sum((truths == 1) & (preds_mask == 0))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        results.append(f1)

        self.V.append(results)
        self.N += 1

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(os.path.join(prefix, "raydrop error"), self.measure()[0], global_step)

    def report(self):
        return f"Rdrop_error (RMSE, Acc, F1) = {self.measure()}"


class IntensityMeter:
    def __init__(self, scale, lpips_fn=None):
        self.V = []
        self.N = 0
        self.scale = scale
        self.lpips_fn = lpips.LPIPS(net='alex').eval()

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths, valid_mask=None):
        preds = preds / self.scale
        truths = truths / self.scale
        if valid_mask is not None and torch.is_tensor(valid_mask):
            valid_mask = valid_mask.detach().cpu().numpy()
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        intensity_error = self.compute_intensity_errors(truths, preds, valid_mask=valid_mask)

        intensity_error = list(intensity_error)
        self.V.append(intensity_error)
        self.N += 1

    def compute_intensity_errors(
            self, gt, pred, min_intensity=1e-6, max_intensity=1.0, valid_mask=None,
    ):
        pred[pred < min_intensity] = min_intensity
        pred[pred > max_intensity] = max_intensity
        gt[gt < min_intensity] = min_intensity
        gt[gt > max_intensity] = max_intensity

        if valid_mask is not None:
            vm = valid_mask.astype(bool)
            if vm.shape != gt.shape:
                vm = np.broadcast_to(vm, gt.shape)
            if vm.any():
                rmse = np.sqrt(((gt - pred)[vm] ** 2).mean())
                medae = np.median(np.abs((gt - pred)[vm]))
                psnr = 10 * np.log10(max_intensity ** 2 / np.mean(((pred - gt)[vm]) ** 2))
            else:
                rmse, medae, psnr = 0.0, 0.0, 0.0
        else:
            rmse = (gt - pred) ** 2
            rmse = np.sqrt(rmse.mean())
            medae = np.median(np.abs(gt - pred))
            psnr = 10 * np.log10(max_intensity ** 2 / np.mean((pred - gt) ** 2))

        if gt.shape[-2] >= 32:
            lpips_loss = self.lpips_fn(torch.from_numpy(pred).squeeze(0),
                                       torch.from_numpy(gt).squeeze(0), normalize=True).item()
        else:
            lpips_loss = 1

        ssim = structural_similarity(
            pred.squeeze(0), gt.squeeze(0), data_range=np.max(gt) - np.min(gt)
        )

        return rmse, medae, lpips_loss, ssim, psnr

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(
            os.path.join(prefix, f"intensity error{suffix}"), self.measure()[0], global_step
        )

    def report(self):
        return f"Inten_error = {self.measure()}"
