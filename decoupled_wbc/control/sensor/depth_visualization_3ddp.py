"""
Depth visualization inspired by 3D-Diffusion-Policy (YanjieZe/3D-Diffusion-Policy).
Reference: https://github.com/YanjieZe/3D-Diffusion-Policy
Uses normalized 3D coordinates (u, v, depth) as RGB for smooth, continuous coloring.
"""

import numpy as np
import cv2


def depth_to_rgb_coord_colormap(
    depth: np.ndarray,
    near_mm: float = 250,
    far_mm: float = 4000,
    invalid_color: tuple = (0, 0, 0),
) -> np.ndarray:
    """
    3D-Diffusion-Policy style: color by normalized (u, v, depth) as RGB.
    Creates smooth, continuous gradients - no horizontal bands.
    Each pixel colored by its position in image + depth space.

    Args:
        depth: (H, W) uint16 depth in mm
        near_mm, far_mm: depth range for normalization
        invalid_color: (B, G, R) for invalid pixels

    Returns:
        (H, W, 3) BGR uint8 image
    """
    h, w = depth.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (depth > 0) & (depth < 65535)
    if not np.any(valid):
        out[:] = invalid_color
        return out

    # Normalize u (col), v (row), z (depth) to [0, 1]
    u = np.tile(np.linspace(0, 1, w), (h, 1))
    v = np.tile(np.linspace(0, 1, h), (w, 1)).T
    depth_f = depth.astype(np.float32)
    depth_clipped = np.clip(depth_f, near_mm, far_mm)
    z = np.where(valid, (depth_clipped - near_mm) / (far_mm - near_mm), 0)

    # Use (u, v, z) as (R, G, B) - 3D-Diffusion-Policy pointcloud style
    r = (u * 255).astype(np.uint8)
    g = (v * 255).astype(np.uint8)
    b = (z * 255).astype(np.uint8)

    out[valid, 0] = b[valid]  # B
    out[valid, 1] = g[valid]  # G
    out[valid, 2] = r[valid]  # R

    return out


def depth_to_jet_smooth(
    depth: np.ndarray,
    near_mm: float = 250,
    far_mm: float = 4000,
) -> np.ndarray:
    """
    Smooth depth colormap: normalize depth to [0,1] and apply JET.
    Simple, no histogram equalization - consistent coloring.
    """
    out = np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    valid = (depth > 0) & (depth < 65535)
    if not np.any(valid):
        return out
    depth_clipped = np.clip(depth.astype(np.float32), near_mm, far_mm)
    depth_norm = np.where(valid, (depth_clipped - near_mm) / (far_mm - near_mm) * 255, 0).astype(np.uint8)
    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    out[valid] = colored[valid]
    return out


def depth_to_turbo_smooth(
    depth: np.ndarray,
    near_mm: float = 250,
    far_mm: float = 4000,
) -> np.ndarray:
    """Smooth TURBO colormap - perceptually uniform, less banding."""
    out = np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    valid = (depth > 0) & (depth < 65535)
    if not np.any(valid):
        return out
    depth_clipped = np.clip(depth.astype(np.float32), near_mm, far_mm)
    depth_norm = np.where(valid, (depth_clipped - near_mm) / (far_mm - near_mm) * 255, 0).astype(np.uint8)
    try:
        colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    except cv2.error:
        colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    out[valid] = colored[valid]
    return out


def depth_to_visualization(
    depth: np.ndarray,
    style: str = "3ddp",
    near_mm: float = 250,
    far_mm: float = 4000,
) -> np.ndarray:
    """
    Main entry: convert depth to BGR visualization.

    Styles:
        "3ddp"  - 3D-Diffusion-Policy style (u,v,depth as RGB) - smooth, no bands
        "jet"   - Simple JET colormap
        "turbo" - TURBO colormap (smooth)
    """
    if style == "3ddp":
        return depth_to_rgb_coord_colormap(depth, near_mm, far_mm)
    elif style == "turbo":
        return depth_to_turbo_smooth(depth, near_mm, far_mm)
    else:
        return depth_to_jet_smooth(depth, near_mm, far_mm)


if __name__ == "__main__":
    # Quick test
    depth = np.random.randint(300, 2000, (480, 640), dtype=np.uint16)
    depth[:, :100] = 0  # invalid
    out = depth_to_visualization(depth, style="3ddp")
    print(f"Output shape: {out.shape}")
    cv2.imwrite("/tmp/depth_3ddp_test.png", out)
    print("Saved /tmp/depth_3ddp_test.png")
