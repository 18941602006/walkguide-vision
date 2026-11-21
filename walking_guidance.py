"""Plan a short image-space direction line inside the current candidate mask."""

import cv2
import numpy as np
from PIL import Image


def direction_path(ground: np.ndarray):
    height, width = ground.shape
    scale = min(1, 160 / width, 200 / height)
    size = (round(width * scale), round(height * scale))
    # Require the complete source cell to belong to the candidate region.
    mask = cv2.resize(ground.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.999
    clearance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    allowed = mask & (clearance >= 2.5)
    h, w = mask.shape
    center = (w - 1) / 2
    start_y = round(h * 0.90)
    positions = np.arange(w)
    starts = allowed[start_y] & (np.abs(positions - center) <= w * 0.15)
    if not starts.any():
        return [], "stop"
    costs = np.where(starts, np.abs(positions - center) * 0.1, np.inf)
    parents = []
    last_y = start_y
    for y in range(start_y - 1, round(h * 0.35), -1):
        options = []
        for offset in range(-2, 3):
            previous = np.roll(costs, offset) + abs(offset) * 0.25
            if offset < 0:
                previous[offset:] = np.inf
            elif offset > 0:
                previous[:offset] = np.inf
            options.append(previous)
        options = np.stack(options)
        choice = options.argmin(axis=0)
        next_costs = options.min(axis=0) + 1 / (clearance[y] + 0.5)
        next_costs[~allowed[y]] = np.inf
        if not np.isfinite(next_costs).any():
            break
        parents.append(positions - (choice - 2))
        costs = next_costs
        last_y = y
    if start_y - last_y < h * 0.18:
        return [], "stop"
    x = int(np.argmin(costs + np.abs(positions - center) * 0.05))
    points = [(x, last_y)]
    for parent, y in zip(reversed(parents), range(last_y + 1, start_y + 1)):
        x = int(parent[x])
        points.append((x, y))
    points.reverse()
    lookahead = points[min(len(points) - 1, round(h * 0.25))]
    offset = lookahead[0] - points[0][0]
    hint = "left" if offset < -w * 0.07 else "right" if offset > w * 0.07 else "straight"
    if last_y > h * 0.55:
        hint = "stop"
    result = [(round((x + 0.5) * width / w), round((y + 0.5) * height / h)) for x, y in points]
    return result, hint


def render_guidance(image: Image.Image, ground: np.ndarray):
    points, hint = direction_path(ground)
    if not points:
        return image, hint
    height, width = ground.shape
    scale = np.array((image.width / width, image.height / height))
    line = np.round(np.asarray(points) * scale).astype(np.int32)
    overlay = np.asarray(image).copy()
    thickness = max(2, round(image.width / 180))
    cv2.polylines(overlay, [line], False, (20, 55, 95), thickness + 4, cv2.LINE_AA)
    cv2.polylines(overlay, [line], False, (255, 255, 255), thickness, cv2.LINE_AA)
    if hint == "stop":
        cv2.drawMarker(overlay, tuple(line[-1]), (255, 65, 65), cv2.MARKER_TILTED_CROSS,
                       markerSize=thickness * 5, thickness=thickness)
    else:
        arrow_start = line[max(0, len(line) - 12)]
        cv2.arrowedLine(overlay, tuple(arrow_start), tuple(line[-1]), (255, 255, 255),
                        thickness, cv2.LINE_AA, tipLength=0.35)
    valid = np.asarray(Image.fromarray(ground).resize(image.size, Image.Resampling.NEAREST))
    overlay[~valid] = np.asarray(image)[~valid]
    return Image.fromarray(overlay), hint
