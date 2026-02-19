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

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

input_path = args.path
if not os.path.isabs(input_path):
    base_dir = os.path.dirname(__file__)
    input_path = os.path.join(base_dir, input_path)

if not os.path.isdir(input_path):
    raise FileNotFoundError(f"Directory not found: {input_path}")

results_path = os.path.join(os.path.dirname(__file__),"results")
os.makedirs(results_path,exist_ok=True)

for name in os.listdir(input_path):
    im_path = os.path.join(input_path, name)
    img = cv2.imread(im_path)
    if img is not None:
        warped = warp_foreground_to_edges(img, threshold=8)
        warped = clean_black_edge_rim(warped, black_thresh=12, edge_band=24)
        output_path = os.path.join(results_path,name)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # 1) initial defect candidates (tune thresholds for your data)
        dark_missing = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)[1]
        bright_missing = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.bitwise_or(dark_missing, bright_missing)

        # 2) clean mask (remove specks, connect nearby gaps)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.dilate(mask, k, iterations=1)

        # 3) infill
        filled = cv2.inpaint(warped, mask, 3, cv2.INPAINT_TELEA)
        
        # 4) balance colour
        lab = cv2.cvtColor(filled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)

        a_gain = 1.25
        b_gain = 1.3
        a2 = np.clip((a.astype(np.float32) - 128.0) * a_gain + 128.0, 0, 255).astype(np.uint8)
        b2 = np.clip((b.astype(np.float32) - 128.0) * b_gain + 128.0, 0, 255).astype(np.uint8)
        balanced = cv2.cvtColor(cv2.merge((l2, a2, b2)), cv2.COLOR_LAB2BGR)

        # 5) remove other noise
        bilater = cv2.bilateralFilter(balanced, 9, 75, 75)
        nonLocalDeNoise = cv2.fastNlMeansDenoisingColored(bilater, None, 10, 10, 7, 21)

        cv2.imwrite(output_path,nonLocalDeNoise)
        print(f"Processed: {name}")
