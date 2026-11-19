"""Render the live view without treating a semantic label as a route clearance."""

import cv2
import numpy as np
from PIL import Image


SIDEWALK = 1
ROAD = 0
OBSTACLE_LABELS = (3, 6, 7)  # Obstacle, person, vehicle (including bicycle).
# NVIDIA's ADE20K label IDs.
INDOOR_FLOOR = (3, 28)  # Floor and rug.
INDOOR_STAIRS = (53, 59, 96, 121)  # Stairs, stairway, escalator, step.
INDOOR_OBSTACLES = (
    10, 12, 14, 15, 19, 23, 24, 30, 31, 33, 35, 38, 41, 42, 44, 45,
    50, 55, 58, 62, 64, 69, 75, 78, 93, 95, 97, 110, 111, 112, 115,
    116, 127, 138,
)
GREEN = (42, 224, 157)
ORANGE = (255, 115, 61)
BLUE = (48, 136, 255)
CYAN = (42, 190, 232)


def analysis_image(image: Image.Image) -> Image.Image:
    scale = min(640 / max(image.size), (384 * 640 / (image.width * image.height)) ** 0.5)
    size = tuple(max(32, round(dimension * scale)) for dimension in image.size)
    return image.resize(size, Image.Resampling.BILINEAR)


def ground_connected(ground: np.ndarray, margin: int) -> np.ndarray:
    ground = cv2.erode(ground.astype(np.uint8), np.ones((margin, margin), dtype=np.uint8))
    _, components = cv2.connectedComponents(ground)
    height, width = ground.shape
    # Keep components that reach the lower, central part of the view.
    near = components[height * 4 // 5:, width // 4:width * 3 // 4]
    ids = np.unique(near)
    return np.isin(components, ids[ids != 0])


def live_regions(labels: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidate = ground_connected((labels == SIDEWALK) & (confidence >= 0.75), 5)
    obstacle = np.isin(labels, OBSTACLE_LABELS) & (confidence >= 0.55)
    return candidate, obstacle


def outdoor_regions(labels: np.ndarray, confidence: np.ndarray, surface_confidence: np.ndarray):
    surface = np.isin(labels, (ROAD, SIDEWALK)) & (surface_confidence >= 0.65)
    # A shared surface confidence avoids holes where road/sidewalk labels compete.
    connected = ground_connected(surface, 3)
    sidewalk = connected & (labels == SIDEWALK)
    # Repair only small gaps still classified as sidewalk in this frame.
    support = (labels == SIDEWALK) & (surface_confidence >= 0.50)
    for _ in range(2):
        sidewalk = cv2.dilate(sidewalk.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & support
    road = connected & (labels == ROAD)
    obstacle = np.isin(labels, OBSTACLE_LABELS) & (confidence >= 0.55)
    return sidewalk, obstacle, road


def indoor_input(image: Image.Image, enhance: bool) -> tuple[Image.Image, np.ndarray]:
    scale = 512 / max(image.size)
    size = tuple(max(32, round(dimension * scale)) for dimension in image.size)
    frame = image.resize(size, Image.Resampling.BILINEAR)
    rgb = np.asarray(frame)
    clipped = rgb.min(axis=2) >= 247
    if not enhance:
        return frame, clipped
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    light = lab[:, :, 0]
    if np.mean(light < 90) >= 0.20:
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
        lab[:, :, 0] = cv2.addWeighted(light, 0.35, enhanced, 0.65, 0)
        frame = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    return frame, clipped


def indoor_regions(labels: np.ndarray, confidence: np.ndarray, clipped: np.ndarray):
    floor = np.isin(labels, INDOOR_FLOOR) & (confidence >= 0.60) & ~clipped
    obstacle = np.isin(labels, INDOOR_OBSTACLES) & (confidence >= 0.40)
    stairs = np.isin(labels, INDOOR_STAIRS) & (confidence >= 0.40)
    return ground_connected(floor, 3), obstacle, stairs


def render_regions(image: Image.Image, candidate: np.ndarray, obstacle: np.ndarray,
                   stairs: np.ndarray | None = None, road: np.ndarray | None = None) -> Image.Image:
    result = image.copy()
    for region, color in ((candidate, GREEN), (road, CYAN), (obstacle, ORANGE), (stairs, BLUE)):
        if region is None:
            continue
        mask = Image.fromarray(region.astype(np.uint8) * 255).resize(
            image.size, Image.Resampling.NEAREST,
        )
        tinted = Image.blend(image, Image.new("RGB", image.size, color), 0.42)
        result.paste(tinted, mask=mask)
    return result


def render_live(image: Image.Image, labels: np.ndarray, confidence: np.ndarray) -> Image.Image:
    return render_regions(image, *live_regions(labels, confidence))


def render_outdoor(image: Image.Image, labels: np.ndarray, confidence: np.ndarray,
                   surface_confidence: np.ndarray) -> Image.Image:
    sidewalk, obstacles, road = outdoor_regions(labels, confidence, surface_confidence)
    return render_regions(image, sidewalk, obstacles, road=road)
