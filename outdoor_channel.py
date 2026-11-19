"""Select the visible surface connected to the camera's near-ground region."""

import cv2
import numpy as np
from PIL import Image

from live_overlay import render_regions


# Cityscapes IDs differ from the original WalkScapes model.
ROAD = 0
SIDEWALK = 1
OBSTACLES = (4, 5, 11, 12, 13, 14, 15, 16, 17, 18)


def channel_input(image: Image.Image) -> Image.Image:
    scale = min(1, 768 / max(image.size))
    size = tuple(max(32, round(value * scale)) for value in image.size)
    return image.resize(size, Image.Resampling.BILINEAR)


def channel_regions(labels: np.ndarray, confidence: np.ndarray):
    height, width = labels.shape
    nearby = np.zeros_like(labels, dtype=bool)
    nearby[int(height * 0.80):int(height * 0.96),
           int(width * 0.35):int(width * 0.65)] = True
    nearby &= confidence >= 0.65
    votes = [
        float(confidence[nearby & (labels == label)].sum())
        for label in (ROAD, SIDEWALK)
    ]
    obstacles = np.isin(labels, OBSTACLES) & (confidence >= 0.40)
    minimum_votes = max(50, height * width * 0.005)
    if max(votes) < minimum_votes or max(votes) < 0.70 * sum(votes):
        return np.zeros_like(nearby), obstacles
    surface_label = (ROAD, SIDEWALK)[int(np.argmax(votes))]
    surface = (labels == surface_label) & (confidence >= 0.55)
    # A margin inside the predicted boundary keeps the overlay off curb edges.
    surface = cv2.erode(surface.astype(np.uint8), np.ones((3, 3), np.uint8))
    count, components = cv2.connectedComponents(surface, connectivity=8)
    anchored = np.unique(components[nearby & (labels == surface_label)])
    seeds = np.bincount(
        components[(labels == surface_label) & (confidence >= 0.80)], minlength=count,
    )
    keep = anchored[(anchored != 0) & (seeds[anchored] >= 50)]
    return np.isin(components, keep), obstacles


def render_channel(image: Image.Image, labels: np.ndarray, confidence: np.ndarray):
    ground, obstacles = channel_regions(labels, confidence)
    ground = trim_channel(image, ground)
    return render_regions(image, ground, obstacles)


def trim_channel(image: Image.Image, ground: np.ndarray) -> np.ndarray:
    """Only remove candidate pixels using near-ground appearance and visible edges."""
    original = ground
    height, width = ground.shape
    scale = min(1, 768 / max(height, width))
    if scale < 1:
        width, height = round(width * scale), round(height * scale)
        ground = cv2.resize(ground.astype(np.uint8), (width, height),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    rgb = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))
    seed = np.zeros_like(ground)
    seed[int(height * 0.80):int(height * 0.96),
         int(width * 0.43):int(width * 0.57)] = True
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    samples = lab[seed & ground]
    if len(samples) >= 100:
        center = np.median(samples, axis=0)
        limits = np.maximum(
            np.percentile(np.abs(samples - center), 90, axis=0) * 2,
            np.array((40, 7, 9)),
        )
        ground = ground & np.all(np.abs(lab - center) <= limits, axis=2)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 100)
    edges[:int(height * 0.40)] = 0
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=45,
        minLineLength=int(height * 0.18), maxLineGap=15,
    )
    boundaries = {}
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if abs(y2 - y1) < height * 0.15:
                continue
            slope = (x2 - x1) / (y2 - y1)
            intercept = x1 - slope * y1
            lower = slope * height * 0.90 + intercept
            upper = slope * height * 0.40 + intercept
            if 0 < lower < width * 0.47 and lower < upper < width * 0.80:
                side, score = "left", lower
            elif width * 0.53 < lower < width and width * 0.20 < upper < lower:
                side, score = "right", -lower
            else:
                continue
            if side not in boundaries or score > boundaries[side][0]:
                boundaries[side] = (score, slope, intercept)

    yy, xx = np.mgrid[:height, :width]
    for side, (_, slope, intercept) in boundaries.items():
        edge_x = slope * np.maximum(yy, height * 0.40) + intercept
        if side == "left":
            ground &= xx >= edge_x + 2
        else:
            ground &= xx <= edge_x - 2

    _, components = cv2.connectedComponents(ground.astype(np.uint8))
    anchors = np.unique(components[seed & ground])
    result = np.isin(components, anchors[anchors != 0]).astype(np.uint8)
    result = cv2.resize(result, (original.shape[1], original.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    return result & original
