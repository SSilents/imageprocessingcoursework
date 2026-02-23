import argparse, os, cv2
import numpy as np


def order_quad_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def warp_foreground_to_edges(img, threshold=16):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    foreground = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.10 * (w * h):
        return img

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    src = order_quad_points(box)
    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_mask = cv2.threshold(warped_gray, threshold, 255, cv2.THRESH_BINARY)[1]
    if cv2.countNonZero(warped_mask) < 0.60 * (w * h):
        return img

    return warped


def clean_black_edge_rim(img, black_thresh=12, edge_band=24):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    near_black = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)[1]

    edge_mask = np.zeros((h, w), dtype=np.uint8)
    edge_mask[:edge_band, :] = 255
    edge_mask[h - edge_band:, :] = 255
    edge_mask[:, :edge_band] = 255
    edge_mask[:, w - edge_band:] = 255

    rim_mask = cv2.bitwise_and(near_black, edge_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rim_mask = cv2.dilate(rim_mask, kernel, iterations=1)

    if cv2.countNonZero(rim_mask) == 0:
        return img

    return cv2.inpaint(img, rim_mask, 3, cv2.INPAINT_TELEA)


def increase_sharpness(img, amount=1.2, blur_kernel=(0, 0), sigma=1.2):
    if amount <= 0:
        return img

    blurred = cv2.GaussianBlur(img, blur_kernel, sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def repair_defects(img, dark_thresh=15, bright_thresh=245, ksize=3, method=cv2.INPAINT_NS, radius=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dark_missing = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)[1]
    bright_missing = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_or(dark_missing, bright_missing)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=1)

    return cv2.inpaint(img, mask, radius, method)


def balance_colour(img, clip_limit=2.0, tile_grid=(8, 8), a_gain=1.25, b_gain=1.3):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l2 = clahe.apply(l)

    a2 = np.clip((a.astype(np.float32) - 128.0) * a_gain + 128.0, 0, 255).astype(np.uint8)
    b2 = np.clip((b.astype(np.float32) - 128.0) * b_gain + 128.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l2, a2, b2)), cv2.COLOR_LAB2BGR)


def denoise_image(
    img,
    bilateral_d=9,
    bilateral_sigma_color=75,
    bilateral_sigma_space=75,
    nlm_h=10,
    nlm_h_color=10,
    nlm_template_window=7,
    nlm_search_window=21,
):
    img = cv2.bilateralFilter(img, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    img = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        nlm_h,
        nlm_h_color,
        nlm_template_window,
        nlm_search_window,
    )
    return img


def apply_processing_pipeline(img, params):
    pipeline = [
        lambda img: warp_foreground_to_edges(img, threshold=params.warp_threshold),
        lambda img: clean_black_edge_rim(img, black_thresh=params.black_thresh, edge_band=params.edge_band),
        lambda img: denoise_image(
            img,
            bilateral_d=params.bilateral_d,
            bilateral_sigma_color=params.bilateral_sigma_color,
            bilateral_sigma_space=params.bilateral_sigma_space,
            nlm_h=params.nlm_h,
            nlm_h_color=params.nlm_h_color,
            nlm_template_window=params.nlm_template_window,
            nlm_search_window=params.nlm_search_window,
        ),
        lambda img: repair_defects(
            img,
            dark_thresh=params.dark_thresh,
            bright_thresh=params.bright_thresh,
            ksize=params.mask_ksize,
            method=cv2.INPAINT_NS if params.inpaint_method == "ns" else cv2.INPAINT_TELEA,
            radius=params.inpaint_radius,
        ),
        lambda img: balance_colour(
            img,
            clip_limit=params.clahe_clip,
            tile_grid=(params.clahe_tile, params.clahe_tile),
            a_gain=params.a_gain,
            b_gain=params.b_gain,
        ),
        lambda img: increase_sharpness(img, amount=params.sharpness, sigma=params.sharp_sigma),
    ]

    for step in pipeline:
        img = step(img)

    return img

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--output", default="results")
parser.add_argument("--warp-threshold", type=int, default=8)
parser.add_argument("--black-thresh", type=int, default=12)
parser.add_argument("--edge-band", type=int, default=24)
parser.add_argument("--dark-thresh", type=int, default=15)
parser.add_argument("--bright-thresh", type=int, default=245)
parser.add_argument("--mask-ksize", type=int, default=3)
parser.add_argument("--clahe-clip", type=float, default=2.0)
parser.add_argument("--clahe-tile", type=int, default=8)
parser.add_argument("--a-gain", type=float, default=1.25)
parser.add_argument("--b-gain", type=float, default=1.3)
parser.add_argument("--bilateral-d", type=int, default=9)
parser.add_argument("--bilateral-sigma-color", type=int, default=75)
parser.add_argument("--bilateral-sigma-space", type=int, default=75)
parser.add_argument("--nlm-h", type=float, default=10)
parser.add_argument("--nlm-h-color", type=float, default=10)
parser.add_argument("--nlm-template-window", type=int, default=7)
parser.add_argument("--nlm-search-window", type=int, default=21)
parser.add_argument("--inpaint-method", choices=["ns", "telea"], default="ns")
parser.add_argument("--inpaint-radius", type=float, default=3)
parser.add_argument("--sharpness", type=float, default=2.0)
parser.add_argument("--sharp-sigma", type=float, default=1.2)
args = parser.parse_args()

input_path = args.path
if not os.path.isabs(input_path):
    base_dir = os.path.dirname(__file__)
    input_path = os.path.join(base_dir, input_path)

if not os.path.isdir(input_path):
    raise FileNotFoundError(f"Directory not found: {input_path}")

results_path = args.output
if not os.path.isabs(results_path):
    results_path = os.path.join(os.path.dirname(__file__), results_path)
os.makedirs(results_path,exist_ok=True)

for name in os.listdir(input_path):
    im_path = os.path.join(input_path, name)
    img = cv2.imread(im_path)
    if img is not None:
        output_path = os.path.join(results_path,name)
        img = apply_processing_pipeline(img, args)
        cv2.imwrite(output_path,img)
        print(f"Processed: {name}")
