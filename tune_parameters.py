import argparse
import csv
import itertools
import os
import re
import shutil
import subprocess
import sys


ACCURACY_PATTERN = re.compile(r"Accuracy is\s+([0-9]*\.?[0-9]+)")


def parse_list(values, cast_fn):
    return [cast_fn(value.strip()) for value in values.split(",") if value.strip()]


def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout


def extract_accuracy(output):
    match = ACCURACY_PATTERN.search(output)
    if not match:
        raise ValueError(f"Could not parse accuracy from output:\n{output}")
    return float(match.group(1))


def evaluate_combo(python_exec, base_dir, input_path, results_path, model_path, combo):
    if os.path.isdir(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path, exist_ok=True)

    process_cmd = [
        python_exec,
        os.path.join(base_dir, "main.py"),
        input_path,
        "--output",
        results_path,
        "--warp-threshold",
        str(combo["warp_threshold"]),
        "--black-thresh",
        str(combo["black_thresh"]),
        "--edge-band",
        str(combo["edge_band"]),
        "--dark-thresh",
        str(combo["dark_thresh"]),
        "--bright-thresh",
        str(combo["bright_thresh"]),
        "--mask-ksize",
        str(combo["mask_ksize"]),
        "--clahe-clip",
        str(combo["clahe_clip"]),
        "--clahe-tile",
        str(combo["clahe_tile"]),
        "--a-gain",
        str(combo["a_gain"]),
        "--b-gain",
        str(combo["b_gain"]),
        "--bilateral-d",
        str(combo["bilateral_d"]),
        "--bilateral-sigma-color",
        str(combo["bilateral_sigma_color"]),
        "--bilateral-sigma-space",
        str(combo["bilateral_sigma_space"]),
        "--nlm-h",
        str(combo["nlm_h"]),
        "--nlm-h-color",
        str(combo["nlm_h_color"]),
        "--nlm-template-window",
        str(combo["nlm_template_window"]),
        "--nlm-search-window",
        str(combo["nlm_search_window"]),
        "--inpaint-method",
        str(combo["inpaint_method"]),
        "--inpaint-radius",
        str(combo["inpaint_radius"]),
        "--sharpness",
        str(combo["sharpness"]),
        "--sharp-sigma",
        str(combo["sharp_sigma"]),
    ]

    run_command(process_cmd)

    classify_cmd = [
        python_exec,
        os.path.join(base_dir, "image_processing_files", "classify.py"),
        "--data",
        results_path,
        "--model",
        model_path,
    ]
    classify_output = run_command(classify_cmd)
    accuracy = extract_accuracy(classify_output)
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Grid search for image preprocessing parameters")
    parser.add_argument("--input", default="image_processing_files/xray_images")
    parser.add_argument("--results", default="results")
    parser.add_argument("--model", default="image_processing_files/classifier.model")
    parser.add_argument("--csv", default="results/parameter_search_results.csv")

    parser.add_argument("--warp-thresholds", default="6,8")
    parser.add_argument("--black-thresholds", default="12")
    parser.add_argument("--edge-bands", default="24")
    parser.add_argument("--dark-thresholds", default="12,15")
    parser.add_argument("--bright-thresholds", default="240,245")
    parser.add_argument("--mask-ksizes", default="3")
    parser.add_argument("--clahe-clips", default="1.8,2.2")
    parser.add_argument("--clahe-tiles", default="8")
    parser.add_argument("--a-gains", default="1.25")
    parser.add_argument("--b-gains", default="1.3")
    parser.add_argument("--bilateral-d-values", default="9")
    parser.add_argument("--bilateral-sigma-colors", default="75")
    parser.add_argument("--bilateral-sigma-spaces", default="75")
    parser.add_argument("--nlm-h-values", default="10")
    parser.add_argument("--nlm-h-color-values", default="10")
    parser.add_argument("--nlm-template-windows", default="7")
    parser.add_argument("--nlm-search-windows", default="21")
    parser.add_argument("--inpaint-methods", default="ns")
    parser.add_argument("--inpaint-radii", default="3")
    parser.add_argument("--sharpness-values", default="1.6,2.0")
    parser.add_argument("--sharp-sigmas", default="1.2")

    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    python_exec = sys.executable

    input_path = args.input if os.path.isabs(args.input) else os.path.join(base_dir, args.input)
    results_path = args.results if os.path.isabs(args.results) else os.path.join(base_dir, args.results)
    model_path = args.model if os.path.isabs(args.model) else os.path.join(base_dir, args.model)
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(base_dir, args.csv)

    warp_thresholds = parse_list(args.warp_thresholds, int)
    black_thresholds = parse_list(args.black_thresholds, int)
    edge_bands = parse_list(args.edge_bands, int)
    dark_thresholds = parse_list(args.dark_thresholds, int)
    bright_thresholds = parse_list(args.bright_thresholds, int)
    mask_ksizes = parse_list(args.mask_ksizes, int)
    clahe_clips = parse_list(args.clahe_clips, float)
    clahe_tiles = parse_list(args.clahe_tiles, int)
    a_gains = parse_list(args.a_gains, float)
    b_gains = parse_list(args.b_gains, float)
    bilateral_d_values = parse_list(args.bilateral_d_values, int)
    bilateral_sigma_colors = parse_list(args.bilateral_sigma_colors, int)
    bilateral_sigma_spaces = parse_list(args.bilateral_sigma_spaces, int)
    nlm_h_values = parse_list(args.nlm_h_values, float)
    nlm_h_color_values = parse_list(args.nlm_h_color_values, float)
    nlm_template_windows = parse_list(args.nlm_template_windows, int)
    nlm_search_windows = parse_list(args.nlm_search_windows, int)
    inpaint_methods = parse_list(args.inpaint_methods, str)
    inpaint_radii = parse_list(args.inpaint_radii, float)
    sharpness_values = parse_list(args.sharpness_values, float)
    sharp_sigmas = parse_list(args.sharp_sigmas, float)

    combinations = list(
        itertools.product(
            warp_thresholds,
            black_thresholds,
            edge_bands,
            dark_thresholds,
            bright_thresholds,
            mask_ksizes,
            clahe_clips,
            clahe_tiles,
            a_gains,
            b_gains,
            bilateral_d_values,
            bilateral_sigma_colors,
            bilateral_sigma_spaces,
            nlm_h_values,
            nlm_h_color_values,
            nlm_template_windows,
            nlm_search_windows,
            inpaint_methods,
            inpaint_radii,
            sharpness_values,
            sharp_sigmas,
        )
    )

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    best_accuracy = -1.0
    best_combo = None
    rows = []

    print(f"Testing {len(combinations)} combinations...")

    for index, values in enumerate(combinations, start=1):
        combo = {
            "warp_threshold": values[0],
            "black_thresh": values[1],
            "edge_band": values[2],
            "dark_thresh": values[3],
            "bright_thresh": values[4],
            "mask_ksize": values[5],
            "clahe_clip": values[6],
            "clahe_tile": values[7],
            "a_gain": values[8],
            "b_gain": values[9],
            "bilateral_d": values[10],
            "bilateral_sigma_color": values[11],
            "bilateral_sigma_space": values[12],
            "nlm_h": values[13],
            "nlm_h_color": values[14],
            "nlm_template_window": values[15],
            "nlm_search_window": values[16],
            "inpaint_method": values[17],
            "inpaint_radius": values[18],
            "sharpness": values[19],
            "sharp_sigma": values[20],
        }

        accuracy = evaluate_combo(python_exec, base_dir, input_path, results_path, model_path, combo)

        row = {
            **combo,
            "accuracy": accuracy,
        }
        rows.append(row)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_combo = combo

        print(
            f"[{index}/{len(combinations)}] "
            f"acc={accuracy:.4f} | "
            f"warp={combo['warp_threshold']} black={combo['black_thresh']} edge={combo['edge_band']} "
            f"dark={combo['dark_thresh']} bright={combo['bright_thresh']} "
            f"clahe={combo['clahe_clip']} sharp={combo['sharpness']}"
        )

    fieldnames = [
        "warp_threshold",
        "black_thresh",
        "edge_band",
        "dark_thresh",
        "bright_thresh",
        "mask_ksize",
        "clahe_clip",
        "clahe_tile",
        "a_gain",
        "b_gain",
        "bilateral_d",
        "bilateral_sigma_color",
        "bilateral_sigma_space",
        "nlm_h",
        "nlm_h_color",
        "nlm_template_window",
        "nlm_search_window",
        "inpaint_method",
        "inpaint_radius",
        "sharpness",
        "sharp_sigma",
        "accuracy",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nBest combination:")
    print(best_combo)
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Saved all results to: {csv_path}")


if __name__ == "__main__":
    main()
