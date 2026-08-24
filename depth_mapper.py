import cv2
import torch
import numpy as np


class DepthMapper:
    """
    Wraps a pretrained MiDaS monocular depth model.

    STATUS: live, real (not LiDAR -- relative/inverse depth from a single
    RGB frame). Say this out loud in the demo: it's genuine depth estimation,
    just not metric/absolute distance.
    """

    def __init__(self, model_type="MiDaS_small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # First run downloads weights -- needs internet once, then cached.
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

    def predict(self, frame_bgr):
        """Return a raw depth map (H x W float array). Higher value = closer."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy()

    def colorize(self, depth_map):
        """Normalize + apply a colormap so depth is visible as an image."""
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min > 1e-6:
            normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth_map)
        normalized_uint8 = (normalized * 255).astype(np.uint8)
        return cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_MAGMA)

    def center_proximity(self, depth_map, box_frac=0.2):
        """
        0-1 'closeness' score for a box in the center of frame.
        Used for the obstacle-alert stretch goal: something looming in
        the middle of the walking path should push this toward 1.0.
        """
        h, w = depth_map.shape
        bh, bw = int(h * box_frac), int(w * box_frac)
        cy, cx = h // 2, w // 2
        region = depth_map[cy - bh:cy + bh, cx - bw:cx + bw]

        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min < 1e-6:
            return 0.0
        region_norm = (region.mean() - depth_min) / (depth_max - depth_min)
        return float(np.clip(region_norm, 0.0, 1.0))
