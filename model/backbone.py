import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _3DBackbone(nn.Module):
	"""
	A lightweight voxel-based 3D backbone that maps raw point clouds to dense 3D features.

	Contract
	- input:  points [B, N, 3] in camera/world coordinates (x, y, z)
	- output: feats  [B, C, D, H, W]
		D/H/W correspond to (z, y, x) respectively, aligning with ROI_3d which expects
		sampling order grid_z -> dim=2 (D), grid_y -> dim=3 (H), grid_x -> dim=4 (W).
		This also treats D as the forward depth axis if your camera uses z-forward.

	Pretrained
	- Will try to partially load weights from pv_rcnn_8369.pth (keys under 'model_state.backbone_3d.*').
	  The checkpoint stores 3D conv weights as [kD, kH, kW, inC, outC], which are transposed to
	  PyTorch format [outC, inC, kD, kH, kW]. Any missing layers (e.g., final projection to C=768)
	  are randomly initialized and loaded with strict=False.

	Notes
	- Voxelization produces 4 input channels per voxel: [mean_x, mean_y, mean_z, occupancy]
	  inside a fixed 3D range. This matches the first conv expecting in_channels=4 from the checkpoint.
	- Default grid resolution is 16^3 to be consistent with the rest of the pipeline.
	- If your camera uses x-forward, set d_axis='x' when constructing to make D align with x-depth.
	"""

	def __init__(
		self,
		grid_size: Tuple[int, int, int] = (16, 16, 16),
		space_range: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((0.0, -2.0, -1.0), (10.0, 2.0, 1.0)),
		out_channels: int = 768,
		d_axis: str = 'x',
		pretrained_path: str = 'pv_rcnn_8369.pth',
		load_pretrained: bool = True,
	):
		super().__init__()
		assert d_axis in ('x', 'y', 'z'), "d_axis must be one of {'x','y','z'}"

		self.grid_D, self.grid_H, self.grid_W = grid_size
		# Fixed range consistent with other parts of the repo (see corr_* definitions)
		(self.x_min, self.y_min, self.z_min), (self.x_max, self.y_max, self.z_max) = space_range
		self.d_axis = d_axis  # which axis maps to D dim
		self.out_channels = out_channels

		# Backbone blocks (match checkpoint naming & shapes)
		# conv_input: in=4 -> 16, k=3
		self.conv_input = nn.Sequential(
			nn.Conv3d(4, 16, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm3d(16),
			nn.ReLU(inplace=True),
		)

		# conv1: one block 16 -> 16
		self.conv1 = nn.Sequential(
			nn.Sequential(
				nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm3d(16),
				nn.ReLU(inplace=True),
			)
		)

		# conv2: three blocks, 16->32 then keep 32
		self.conv2 = nn.Sequential(
			nn.Sequential(
				nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm3d(32),
				nn.ReLU(inplace=True),
			),
			nn.Sequential(
				nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm3d(32),
				nn.ReLU(inplace=True),
			),
			nn.Sequential(
				nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm3d(32),
				nn.ReLU(inplace=True),
			),
		)

		# conv3: three blocks, 32->64 then keep 64
		self.conv3 = nn.Sequential(
			nn.Sequential(
				nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm3d(64),
				nn.ReLU(inplace=True),
			),
			nn.Sequential(
				nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm3d(64),
				nn.ReLU(inplace=True),
			),
			nn.Sequential(
				nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm3d(64),
				nn.ReLU(inplace=True),
			),
		)

		# conv4: three blocks, keep 64
		self.conv4 = nn.Sequential(
			nn.Sequential(
				nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm3d(64),
				nn.ReLU(inplace=True),
			),
			nn.Sequential(
				nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm3d(64),
				nn.ReLU(inplace=True),
			),
			nn.Sequential(
				nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm3d(64),
				nn.ReLU(inplace=True),
			),
		)

		# conv_out: 64 -> 128 with kernel (3,1,1) as suggested by checkpoint shape
		self.conv_out = nn.Sequential(
			nn.Conv3d(64, 128, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
			nn.BatchNorm3d(128),
			nn.ReLU(inplace=True),
		)

		# Project to desired output channels for fusion with 2D backbone (e.g., 768 for DINOv2-B/14)
		self.proj = nn.Conv3d(128, self.out_channels, kernel_size=1, bias=True)

		if load_pretrained and pretrained_path is not None:
			self._load_pretrained_from_pvrcnn(pretrained_path)

	# --------------------------- public API ---------------------------
	def forward(self, points: torch.Tensor) -> torch.Tensor:
		"""
		points: [B, N, 3] (x, y, z)
		returns: [B, C, D, H, W]
		"""
		assert points.dim() == 3 and points.size(-1) == 3, 'Expect [B, N, 3]'
		B = points.size(0)

		# 1) voxelize raw points into 4-channel grid [B, 4, D, H, W]
		vox = self._points_to_voxels(points, grid_size=(self.grid_D, self.grid_H, self.grid_W))

		# 2) 3D CNN backbone
		x = self.conv_input(vox)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv_out(x)
		x = self.proj(x)  # [B, out_channels, D, H, W]
		return x

	# --------------------------- helpers ---------------------------
	@torch.no_grad()
	def _points_to_voxels(self, pts: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
		"""
		Scatter points to a dense voxel grid.

		Feature per voxel (4 channels): [mean_x, mean_y, mean_z, occupancy]

		Axis mapping to grid dims [D,H,W]:
			D <- d_axis (default 'z'), H <- 'y', W <- remaining axis ('x').
		"""
		B, N, _ = pts.shape

		# Clamp to fixed range
		x = pts[..., 0].clamp(self.x_min, self.x_max)
		y = pts[..., 1].clamp(self.y_min, self.y_max)
		z = pts[..., 2].clamp(self.z_min, self.z_max)

		# Choose axis order for (D,H,W)
		if self.d_axis == 'z':
			d_vals, h_vals, w_vals = z, y, x
			d_min, d_max = self.z_min, self.z_max
			h_min, h_max = self.y_min, self.y_max
			w_min, w_max = self.x_min, self.x_max
		elif self.d_axis == 'x':
			d_vals, h_vals, w_vals = x, y, z
			d_min, d_max = self.x_min, self.x_max
			h_min, h_max = self.y_min, self.y_max
			w_min, w_max = self.z_min, self.z_max
		else:  # 'y'
			d_vals, h_vals, w_vals = y, z, x
			d_min, d_max = self.y_min, self.y_max
			h_min, h_max = self.z_min, self.z_max
			w_min, w_max = self.x_min, self.x_max

		D, H, W = grid_size

		def to_idx(v, vmin, vmax, size):
			# map to [0, size-1]
			denom = max(vmax - vmin, 1e-6)
			scaled = (v - vmin) / denom
			idx = torch.clamp((scaled * (size - 1)).long(), 0, size - 1)
			return idx

		d_idx = to_idx(d_vals, d_min, d_max, D)
		h_idx = to_idx(h_vals, h_min, h_max, H)
		w_idx = to_idx(w_vals, w_min, w_max, W)

		lin = (d_idx * (H * W) + h_idx * W + w_idx)  # [B, N]

		device = pts.device
		vox_sum_x = torch.zeros(B, D * H * W, device=device, dtype=pts.dtype)
		vox_sum_y = torch.zeros_like(vox_sum_x)
		vox_sum_z = torch.zeros_like(vox_sum_x)
		vox_cnt = torch.zeros_like(vox_sum_x)

		# scatter add per batch
		batch_ids = torch.arange(B, device=device).view(B, 1).expand(B, N)
		vox_sum_x.index_put_((batch_ids, lin), x, accumulate=True)
		vox_sum_y.index_put_((batch_ids, lin), y, accumulate=True)
		vox_sum_z.index_put_((batch_ids, lin), z, accumulate=True)
		vox_cnt.index_put_((batch_ids, lin), torch.ones_like(x), accumulate=True)

		# avoid div by zero
		eps = 1e-6
		mean_x = vox_sum_x / (vox_cnt + eps)
		mean_y = vox_sum_y / (vox_cnt + eps)
		mean_z = vox_sum_z / (vox_cnt + eps)
		occ = (vox_cnt > 0).to(pts.dtype)

		# reshape to [B, C=4, D, H, W]
		mean_x = mean_x.view(B, D, H, W)
		mean_y = mean_y.view(B, D, H, W)
		mean_z = mean_z.view(B, D, H, W)
		occ = occ.view(B, D, H, W)

		feats = torch.stack([mean_x, mean_y, mean_z, occ], dim=1)  # [B,4,D,H,W]
		return feats

	def _load_pretrained_from_pvrcnn(self, ckpt_path: str):
		try:
			sd = torch.load(ckpt_path, map_location='cpu')
			if isinstance(sd, dict) and 'model_state' in sd:
				sd = sd['model_state']
			# filter backbone_3d.*
			b3d = {k[len('backbone_3d.') :]: v for k, v in sd.items() if k.startswith('backbone_3d.')}

			mapped = {}
			for k, v in b3d.items():
				if not isinstance(v, torch.Tensor):
					continue
				# conv weights in ckpt are [kD,kH,kW,inC,outC] -> transpose to [outC,inC,kD,kH,kW]
				if k.endswith('.weight') and v.dim() == 5:
					v = v.permute(4, 3, 0, 1, 2).contiguous()
				mapped[k] = v

			# load with strict=False
			missing, unexpected = self.load_state_dict(mapped, strict=False)
			# Optional: print short summary during development (suppressed in production)
			if len(missing) > 0 or len(unexpected) > 0:
				# Register buffers for visibility but don't raise
				self.register_buffer('_pretrained_missing', torch.tensor([len(missing)]), persistent=False)
				self.register_buffer('_pretrained_unexpected', torch.tensor([len(unexpected)]), persistent=False)
		except Exception as e:
			# If anything goes wrong, keep randomly initialized weights
			self.register_buffer('_pretrained_error', torch.tensor([1]), persistent=False)
			# You can uncomment the next line for debugging:
			# print(f"[3DBackbone] Failed to load pretrained from {ckpt_path}: {e}")


__all__ = ['_3DBackbone']




