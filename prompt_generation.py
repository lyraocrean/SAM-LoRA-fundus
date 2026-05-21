import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage import measure, morphology

Point = Tuple[int, int]


@dataclass
class PromptSet:
    center: Point
    edge: List[Point]
    negative: List[Point]

    @property
    def all_positive(self) -> List[Point]:
        return [self.center] + self.edge


@dataclass
class BoxPromptSet:
    boxes_xyxy: List[Tuple[int, int, int, int]]


def _ensure_bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype != np.bool_:
        mask = mask > 0
    return mask.astype(bool)


def _largest_component(mask: np.ndarray) -> Tuple[np.ndarray, bool]:
    labels = measure.label(mask, connectivity=1)
    if labels.max() == 0:
        return mask, False
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return labels == keep, True


def _euclidean_far_enough(candidate: Point, existing: Sequence[Point], min_distance: float) -> bool:
    for p in existing:
        dy = candidate[0] - p[0]
        dx = candidate[1] - p[1]
        if (dy * dy + dx * dx) ** 0.5 < min_distance:
            return False
    return True


def _sample_centers_with_min_distance(
    candidates: Sequence[Point],
    count: int,
    min_distance: float,
    rng: np.random.Generator,
) -> List[Point]:
    if not candidates:
        return []
    order = rng.permutation(len(candidates))
    picked: List[Point] = []
    for idx in order:
        cand = candidates[idx]
        if _euclidean_far_enough(cand, picked, min_distance):
            picked.append(cand)
            if len(picked) == count:
                return picked
    for idx in order:
        cand = candidates[idx]
        if cand not in picked:
            picked.append(cand)
            if len(picked) == count:
                break
    return picked


def _select_points_from_mask(
    mask: np.ndarray, desired: int, rng: np.random.Generator, exclude: Sequence[Point] = ()
) -> List[Point]:
    coords = np.argwhere(mask)
    coords = [tuple(c) for c in coords if tuple(c) not in exclude]
    if not coords:
        return []
    if len(coords) <= desired:
        order = rng.permutation(len(coords))
        return [coords[i] for i in order]
    indices = rng.choice(len(coords), size=desired, replace=False)
    return [coords[i] for i in indices]


def _sample_negative_points(
    background_mask: np.ndarray,
    count: int,
    min_distance: float,
    avoid: Sequence[Point],
    rng: np.random.Generator,
) -> List[Point]:
    candidates = [tuple(c) for c in np.argwhere(background_mask)]
    if not candidates:
        return []

    order = rng.permutation(len(candidates))
    picked: List[Point] = []
    for idx in order:
        cand = candidates[idx]
        if _euclidean_far_enough(cand, picked + list(avoid), min_distance):
            picked.append(cand)
            if len(picked) == count:
                break

    # Fallback to fill quota even if spacing is tight.
    while len(picked) < count and candidates:
        cand = candidates[rng.integers(0, len(candidates))]
        if cand not in picked:
            picked.append(cand)
    return picked


def _square_box_from_center(
    center_rc: Point,
    box_size: int,
    h: int,
    w: int,
) -> Tuple[int, int, int, int]:
    y, x = center_rc
    half = max(int(box_size) // 2, 1)
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w - 1, x + half)
    y2 = min(h - 1, y + half)
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return (x1, y1, x2, y2)


def _skeleton_endpoints(skeleton: np.ndarray) -> List[Point]:
    skel = skeleton.astype(np.uint8)
    if skel.sum() == 0:
        return []
    kernel = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
        dtype=np.uint8,
    )
    neighbor_count = cv2.filter2D(skel, ddepth=cv2.CV_16S, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    endpoint_mask = (skel == 1) & (neighbor_count == 1)
    return [tuple(c) for c in np.argwhere(endpoint_mask)]


def generate_box_prompts_from_mask(
    mask: np.ndarray,
    min_boxes: int = 3,
    max_boxes: int = 4,
    box_size: int = 96,
    min_component_area: int = 50,
    min_center_distance: int = 40,
    seed: Optional[int] = None,
) -> BoxPromptSet:
    """
    Generate small square boxes that cover vessel regions from a binary mask.
    Boxes are returned in (x1, y1, x2, y2) format.
    """
    if min_boxes < 1 or max_boxes < 1:
        raise ValueError("min_boxes and max_boxes must be >= 1.")
    if min_boxes > max_boxes:
        raise ValueError("min_boxes must be <= max_boxes.")

    rng = np.random.default_rng(seed)
    bool_mask = _ensure_bool_mask(mask)
    labels = measure.label(bool_mask, connectivity=1)
    if labels.max() == 0:
        raise ValueError("Mask is empty; cannot sample box prompts.")

    comps = measure.regionprops(labels)
    comps = [c for c in comps if c.area >= min_component_area]
    if not comps:
        comps = measure.regionprops(labels)

    target_count = int(rng.integers(min_boxes, max_boxes + 1))
    centers: List[Point] = []
    for comp in comps:
        comp_mask = labels == comp.label
        skeleton = morphology.skeletonize(comp_mask)
        skel_points = [tuple(c) for c in np.argwhere(skeleton)]
        if not skel_points:
            skel_points = [tuple(c) for c in np.argwhere(comp_mask)]
        if not skel_points:
            continue
        per_comp = max(1, target_count // max(len(comps), 1))
        centers.extend(
            _sample_centers_with_min_distance(
                skel_points,
                count=per_comp,
                min_distance=min_center_distance,
                rng=rng,
            )
        )

    if len(centers) < target_count:
        all_skel = [tuple(c) for c in np.argwhere(morphology.skeletonize(bool_mask))]
        if not all_skel:
            all_skel = [tuple(c) for c in np.argwhere(bool_mask)]
        centers.extend(
            _sample_centers_with_min_distance(
                all_skel,
                count=target_count - len(centers),
                min_distance=min_center_distance,
                rng=rng,
            )
        )

    h, w = bool_mask.shape
    boxes: List[Tuple[int, int, int, int]] = []
    used = set()
    for c in centers:
        b = _square_box_from_center(c, box_size=box_size, h=h, w=w)
        if b in used:
            continue
        used.add(b)
        boxes.append(b)
        if len(boxes) == target_count:
            break

    if len(boxes) < min_boxes:
        vessel_points = [tuple(c) for c in np.argwhere(bool_mask)]
        extra_centers = _sample_centers_with_min_distance(
            vessel_points,
            count=min_boxes - len(boxes),
            min_distance=min_center_distance,
            rng=rng,
        )
        for c in extra_centers:
            b = _square_box_from_center(c, box_size=box_size, h=h, w=w)
            if b not in used:
                used.add(b)
                boxes.append(b)

    return BoxPromptSet(boxes_xyxy=boxes[:max_boxes])


def generate_terminal_box_prompts_from_mask(
    mask: np.ndarray,
    min_boxes: int = 3,
    max_boxes: int = 4,
    box_size: int = 96,
    min_component_area: int = 50,
    min_center_distance: int = 40,
    terminal_radius: int = 8,
    seed: Optional[int] = None,
) -> BoxPromptSet:
    """
    Generate square boxes with terminal-priority sampling.
    Terminal candidates are skeleton endpoints and their local neighborhoods.
    """
    if min_boxes < 1 or max_boxes < 1:
        raise ValueError("min_boxes and max_boxes must be >= 1.")
    if min_boxes > max_boxes:
        raise ValueError("min_boxes must be <= max_boxes.")

    rng = np.random.default_rng(seed)
    bool_mask = _ensure_bool_mask(mask)
    labels = measure.label(bool_mask, connectivity=1)
    if labels.max() == 0:
        raise ValueError("Mask is empty; cannot sample box prompts.")

    comps = measure.regionprops(labels)
    comps = [c for c in comps if c.area >= min_component_area]
    if not comps:
        comps = measure.regionprops(labels)

    target_count = int(rng.integers(min_boxes, max_boxes + 1))
    centers: List[Point] = []
    all_terminal_candidates: List[Point] = []
    radius = max(int(terminal_radius), 1)

    for comp in comps:
        comp_mask = labels == comp.label
        skeleton = morphology.skeletonize(comp_mask)
        endpoints = _skeleton_endpoints(skeleton)
        if not endpoints:
            continue

        endpoint_mask = np.zeros_like(skeleton, dtype=bool)
        ys, xs = zip(*endpoints)
        endpoint_mask[np.array(ys), np.array(xs)] = True
        terminal_region = morphology.binary_dilation(endpoint_mask, footprint=morphology.disk(radius))
        terminal_candidates = [tuple(c) for c in np.argwhere(terminal_region & skeleton)]
        if not terminal_candidates:
            terminal_candidates = endpoints
        all_terminal_candidates.extend(terminal_candidates)

        per_comp = max(1, target_count // max(len(comps), 1))
        centers.extend(
            _sample_centers_with_min_distance(
                terminal_candidates,
                count=per_comp,
                min_distance=min_center_distance,
                rng=rng,
            )
        )

    if len(centers) < target_count:
        global_skeleton = morphology.skeletonize(bool_mask)
        global_endpoints = _skeleton_endpoints(global_skeleton)
        global_terminal_candidates: List[Point] = []
        if global_endpoints:
            endpoint_mask = np.zeros_like(global_skeleton, dtype=bool)
            ys, xs = zip(*global_endpoints)
            endpoint_mask[np.array(ys), np.array(xs)] = True
            terminal_region = morphology.binary_dilation(endpoint_mask, footprint=morphology.disk(radius))
            global_terminal_candidates = [tuple(c) for c in np.argwhere(terminal_region & global_skeleton)]
            if not global_terminal_candidates:
                global_terminal_candidates = global_endpoints
        elif all_terminal_candidates:
            global_terminal_candidates = all_terminal_candidates

        centers.extend(
            _sample_centers_with_min_distance(
                global_terminal_candidates,
                count=target_count - len(centers),
                min_distance=min_center_distance,
                rng=rng,
            )
        )

    h, w = bool_mask.shape
    boxes: List[Tuple[int, int, int, int]] = []
    used = set()
    for c in centers:
        b = _square_box_from_center(c, box_size=box_size, h=h, w=w)
        if b in used:
            continue
        used.add(b)
        boxes.append(b)
        if len(boxes) == target_count:
            break

    if len(boxes) < min_boxes:
        vessel_points = [tuple(c) for c in np.argwhere(bool_mask)]
        extra_centers = _sample_centers_with_min_distance(
            vessel_points,
            count=min_boxes - len(boxes),
            min_distance=min_center_distance,
            rng=rng,
        )
        for c in extra_centers:
            b = _square_box_from_center(c, box_size=box_size, h=h, w=w)
            if b not in used:
                used.add(b)
                boxes.append(b)

    return BoxPromptSet(boxes_xyxy=boxes[:max_boxes])


def encode_boxes_as_sam_points(
    boxes_xyxy: Sequence[Tuple[int, int, int, int]]
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Encode each box as two SAM prompt points:
    top-left corner with label 2, bottom-right corner with label 3.
    """
    coords_xy: List[Tuple[int, int]] = []
    labels: List[int] = []
    for x1, y1, x2, y2 in boxes_xyxy:
        coords_xy.append((x1, y1))
        labels.append(2)
        coords_xy.append((x2, y2))
        labels.append(3)
    return coords_xy, labels


def generate_prompts_from_mask(
    mask: np.ndarray,
    edge_distance_thresh: int = 4,
    negative_min_distance: int = 5,
    seed: Optional[int] = None,
) -> PromptSet:
    """
    Scheme 1: 1 center point (deep inside vessel), 2 edge points near fine vessels, 2 negatives.
    Returns points in (row, col) order relative to the input mask.
    """
    rng = np.random.default_rng(seed)
    bool_mask = _ensure_bool_mask(mask)
    vessel_mask, has_component = _largest_component(bool_mask)
    if not has_component:
        raise ValueError("Mask is empty; cannot sample prompt points.")

    distance_map = distance_transform_edt(vessel_mask)
    center: Point = tuple(np.unravel_index(np.argmax(distance_map), distance_map.shape))

    skeleton = morphology.skeletonize(vessel_mask)
    boundary = vessel_mask & (~binary_erosion(vessel_mask))
    distance_to_boundary = distance_transform_edt(~boundary)
    edge_candidates = skeleton & (distance_to_boundary <= edge_distance_thresh)

    edge_points = _select_points_from_mask(edge_candidates, desired=2, rng=rng, exclude=(center,))
    if len(edge_points) < 2:
        boundary_points = _select_points_from_mask(boundary, desired=2 - len(edge_points), rng=rng, exclude=(center,))
        edge_points.extend(boundary_points)
    if len(edge_points) < 2:
        filler = _select_points_from_mask(vessel_mask, desired=2 - len(edge_points), rng=rng, exclude=(center,))
        edge_points.extend(filler)

    negative_points = _sample_negative_points(
        ~vessel_mask, count=2, min_distance=negative_min_distance, avoid=[center, *edge_points], rng=rng
    )

    return PromptSet(center=center, edge=edge_points[:2], negative=negative_points)


def _ensure_rgb_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        return image[..., :3]
    return image


def visualize_prompts(
    image: np.ndarray,
    mask: np.ndarray,
    prompts: PromptSet,
    save_path: str | None = None,
    show: bool = True,
    title: str | None = None,
):
    image_rgb = _ensure_rgb_image(image)
    mask_vis = mask if mask.ndim == 2 else mask[..., 0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[1].imshow(mask_vis, cmap="gray")
    axes[1].set_title("Mask")
    axes[2].imshow(image_rgb)
    axes[2].imshow(mask_vis, cmap="gray", alpha=0.4)
    axes[2].set_title("Mask + Prompts")

    def _scatter(points: Sequence[Point], marker: str, label: str):
        if not points:
            return
        ys, xs = zip(*points)
        axes[2].scatter(xs, ys, marker=marker, color="red", s=30, label=label)

    _scatter([prompts.center], marker="o", label="center +")
    _scatter(prompts.edge, marker="s", label="edge +")
    _scatter(prompts.negative, marker="x", label="negative -")

    axes[2].legend(loc="lower right", fontsize="small")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _load_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask from {path}")
    if mask.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if mask.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        mask = cv2.cvtColor(mask, code)
    return mask


def _load_image_or_default(image_path: Optional[str], mask: np.ndarray) -> np.ndarray:
    if image_path is None:
        scaled = (mask.astype(np.float32) / (mask.max() + 1e-6) * 255).astype(np.uint8)
        return np.stack([scaled] * 3, axis=-1)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image from {image_path}")
    if image.ndim == 3:
        code = cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB
        image = cv2.cvtColor(image, code)
    return image


def main():
    parser = argparse.ArgumentParser(description="Generate prompt points from a binary mask (Scheme 1).")
    parser.add_argument("--mask", required=True, help="Path to binary mask (ground truth).")
    parser.add_argument("--image", default=None, help="Optional path to the corresponding image for visualization.")
    parser.add_argument("--save", default=None, help="Path to save the visualization figure.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--edge_distance", type=int, default=4, help="Max distance to boundary for edge points.")
    parser.add_argument("--neg_min_distance", type=int, default=5, help="Min distance for negative points to others.")
    args = parser.parse_args()

    mask = _load_mask(args.mask)
    prompts = generate_prompts_from_mask(
        mask=mask,
        edge_distance_thresh=args.edge_distance,
        negative_min_distance=args.neg_min_distance,
        seed=args.seed,
    )
    image = _load_image_or_default(args.image, mask)

    visualize_prompts(
        image=image,
        mask=mask,
        prompts=prompts,
        save_path=args.save,
        show=not args.save,
        title="Prompt Sampling (Scheme 1)",
    )

    print(
        f"center: {prompts.center}, "
        f"edge: {prompts.edge}, "
        f"negative: {prompts.negative}, "
        f"saved_to: {args.save or 'displayed'}"
    )


if __name__ == "__main__":
    main()
