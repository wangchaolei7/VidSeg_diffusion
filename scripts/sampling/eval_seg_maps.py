import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from scripts.sampling.dataset_specs import build_dataset_spec, list_sequences, resolve_gt_mask_path
from scripts.sampling.metrics_ovdg import OVDGMetrics


def _frame_sort_key(filename: str) -> Tuple[int, str]:
    stem = os.path.splitext(filename)[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def _load_frame_names(color_dir: str) -> List[str]:
    frame_files = [f for f in os.listdir(color_dir) if f.endswith(".png") or f.endswith(".jpg")]
    frame_files = sorted(frame_files, key=_frame_sort_key)
    return [os.path.splitext(f)[0] for f in frame_files]


def _load_cityscapes_gt_bases(mask_dir: str, mask_suffix: str, mask_ext: str) -> List[str]:
    if not mask_dir or not os.path.isdir(mask_dir):
        return []
    suffix = f"{mask_suffix}{mask_ext}"
    base_names = []
    for fname in os.listdir(mask_dir):
        if not fname.endswith(suffix):
            continue
        base_names.append(fname[: -len(suffix)])
    return sorted(base_names, key=_frame_sort_key)


def _load_gray_image(img_path: str, size: Tuple[int, int]) -> Optional[np.ndarray]:
    if not os.path.isfile(img_path):
        return None
    img = Image.open(img_path).convert("L")
    img = img.resize(size, Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _resolve_pred_dir(output_root: str, exp_name: str, pred_folder: str, modulate_lambda_start: float) -> Optional[str]:
    seg_root = os.path.join(output_root, exp_name, pred_folder)
    if not os.path.isdir(seg_root):
        return None
    expected = os.path.join(seg_root, f"{0:06d}_l_{modulate_lambda_start}")
    if os.path.isdir(expected):
        return expected
    candidates = [d for d in os.listdir(seg_root) if os.path.isdir(os.path.join(seg_root, d))]
    if not candidates:
        return None
    target = f"_l_{modulate_lambda_start}"
    for name in candidates:
        if target in name:
            return os.path.join(seg_root, name)
    return os.path.join(seg_root, sorted(candidates)[0])


def _load_target_size(color_dir: str, frame_name: str) -> Optional[Tuple[int, int]]:
    for ext in (".jpg", ".png"):
        img_path = os.path.join(color_dir, f"{frame_name}{ext}")
        if os.path.isfile(img_path):
            with Image.open(img_path) as img:
                width, height = img.size
            return (width, height)
    return None


def _maybe_resize_label(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if arr.shape[1] == size[0] and arr.shape[0] == size[1]:
        return arr
    img = Image.fromarray(arr.astype(np.uint8))
    img = img.resize(size, Image.NEAREST)
    return np.array(img)


def _write_metrics_outputs(output_root: str, metrics: OVDGMetrics) -> None:
    summary = metrics.compute_summary()

    def to_percent(val):
        return round(float(val) * 100.0, 2) if val is not None and np.isfinite(val) else None

    summary_out = {
        "mIoU": to_percent(summary["mIoU"]),
        "mAcc": to_percent(summary["mAcc"]),
        "aAcc": to_percent(summary["aAcc"]),
    }
    for n in metrics.mvc_n_list:
        summary_out[f"mVC{n}"] = to_percent(summary.get(f"mVC{n}", float("nan")))
        summary_out[f"mVC{n}_videos"] = summary.get(f"mVC{n}_videos", 0)
    summary_out["num_classes"] = metrics.num_classes
    summary_out["ignore_index"] = metrics.ignore_index
    summary_out["tc_stride"] = metrics.tc_stride

    summary_path = os.path.join(output_root, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(json_dumps(summary_out))

    per_class_path = os.path.join(output_root, "metrics_per_class.csv")
    per_class_iou = summary["per_class_iou"]
    per_class_acc = summary["per_class_acc"]
    with open(per_class_path, "w", encoding="utf-8") as handle:
        handle.write("class_id,iou,acc\n")
        for idx in range(metrics.num_classes):
            iou_val = per_class_iou[idx]
            acc_val = per_class_acc[idx]
            iou_out = to_percent(iou_val) if np.isfinite(iou_val) else ""
            acc_out = to_percent(acc_val) if np.isfinite(acc_val) else ""
            handle.write(f"{idx},{iou_out},{acc_out}\n")

    log_path = os.path.join(output_root, "metrics_log.txt")
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(json_dumps(summary_out))
        handle.write("\n")


def json_dumps(payload: dict) -> str:
    items = []
    for key, value in payload.items():
        if value is None:
            items.append(f'  "{key}": null')
        elif isinstance(value, (int, float)):
            items.append(f'  "{key}": {value}')
        else:
            items.append(f'  "{key}": "{value}"')
    return "{\n" + ",\n".join(items) + "\n}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../dataset/vspw/VSPW_480p/data")
    parser.add_argument("--split_file_path", type=str, default="../dataset/vspw/VSPW_480p/val.txt")
    parser.add_argument(
        "--dataset",
        type=str,
        default="apollo",
        choices=["vspw", "apollo", "camvid", "cityscapes_origin", "cityscapes_corruptions"],
    )
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--color_root", type=str, default=None)
    parser.add_argument("--mask_root", type=str, default=None)
    parser.add_argument("--mask_suffix", type=str, default="")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument(
        "--corruption",
        type=str,
        default=None,
        choices=["fog", "frost", "snow", "spatter"],
        help="corruption type for cityscapes_corruptions",
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        default=None,
        help="comma-separated corruptions for cityscapes_corruptions",
    )
    parser.add_argument("--pred_folder", type=str, default="segmentation_map_raw")
    parser.add_argument("--modulate_lambda_start", type=float, default=50.0)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--tc_stride", type=int, default=4)
    parser.add_argument("--mvc_n", type=str, default="8,16")
    parser.add_argument("--cityscapes_sparse_mvc", action="store_true", help="use cityscapes sparse mVC")
    parser.add_argument("--citys_sim_thresh", type=float, default=20.0, help="cityscapes static pixel threshold")
    args = parser.parse_args()

    mvc_n = [int(x) for x in args.mvc_n.split(",") if x.strip()]
    if args.dataset == "cityscapes_corruptions" and args.corruptions:
        corruption_list = [x.strip() for x in args.corruptions.split(",") if x.strip()]
    elif args.dataset == "cityscapes_corruptions":
        if not args.corruption:
            raise ValueError("corruption is required for cityscapes_corruptions")
        corruption_list = [args.corruption]
    else:
        corruption_list = [None]

    num_classes = args.num_classes
    if num_classes is None:
        num_classes = 15 if args.dataset in ["apollo", "camvid", "cityscapes_origin", "cityscapes_corruptions"] else 124

    for corruption in corruption_list:
        if args.dataset == "cityscapes_corruptions":
            args.corruption = corruption

        output_root = args.output_root
        if output_root is None:
            if args.dataset == "apollo":
                output_root = "/data1/wangcl/project/VidSeg/apollo"
            elif args.dataset == "camvid":
                output_root = "/data1/wangcl/project/VidSeg/camvid"
            elif args.dataset == "cityscapes_origin":
                output_root = "/data1/wangcl/project/VidSeg/cityscapes_origin"
            elif args.dataset == "cityscapes_corruptions":
                output_root = os.path.join(
                    "/data1/wangcl/project/VidSeg/cityscapes_corruptions",
                    args.corruption,
                )
            else:
                output_root = "/data1/wangcl/project/VidSeg"

        spec = build_dataset_spec(args)
        sequences = list_sequences(spec)
        if not sequences:
            raise ValueError(f"No sequences found under {spec.color_root}")

        metrics = OVDGMetrics(
            num_classes=num_classes,
            ignore_index=args.ignore_index,
            mvc_n=mvc_n,
            tc_stride=args.tc_stride,
            citys_sim_thresh=args.citys_sim_thresh,
        )

        use_citys_sparse = args.cityscapes_sparse_mvc or args.dataset in [
            "cityscapes_origin",
            "cityscapes_corruptions",
        ]

        seq_iter = tqdm(sequences, desc="sequences", unit="seq")
        for exp_name, color_dir, mask_dir in seq_iter:
            pred_dir = _resolve_pred_dir(output_root, exp_name, args.pred_folder, args.modulate_lambda_start)
            if pred_dir is None:
                continue
            frame_names = _load_frame_names(color_dir)
            gt_bases = set()
            if args.dataset in ["cityscapes_origin", "cityscapes_corruptions"]:
                gt_bases = set(_load_cityscapes_gt_bases(mask_dir, spec.mask_suffix, spec.mask_ext))
            preds = []
            gts = []
            imgs = []
            ref_gt = None
            ref_img = None
            missing_pred = 0
            frame_iter = tqdm(frame_names, desc=exp_name, unit="frame", leave=False)
            for frame_name in frame_iter:
                pred_path = os.path.join(pred_dir, f"{frame_name}.png")
                if not os.path.isfile(pred_path):
                    missing_pred += 1
                    continue
                pred = np.array(Image.open(pred_path))
                target_size = _load_target_size(color_dir, frame_name)
                if target_size is None:
                    gt_path = resolve_gt_mask_path(mask_dir, frame_name, spec.mask_suffix, spec.mask_ext)
                    if gt_path:
                        with Image.open(gt_path) as gt_img:
                            target_size = gt_img.size
                if target_size is not None:
                    pred = _maybe_resize_label(pred, target_size)
                gt = None
                if frame_name.endswith("_leftImg8bit"):
                    frame_base = frame_name[: -len("_leftImg8bit")]
                else:
                    frame_base = frame_name
                if not gt_bases or frame_base in gt_bases:
                    gt_path = resolve_gt_mask_path(mask_dir, frame_name, spec.mask_suffix, spec.mask_ext)
                    if gt_path:
                        gt = np.array(Image.open(gt_path))
                        if target_size is not None and (gt.shape[1] != target_size[0] or gt.shape[0] != target_size[1]):
                            gt = _maybe_resize_label(gt, target_size)
                img_gray = None
                if use_citys_sparse:
                    size = (pred.shape[1], pred.shape[0])
                    img_gray = _load_gray_image(os.path.join(color_dir, f"{frame_name}.png"), size)
                    if img_gray is None:
                        img_gray = _load_gray_image(os.path.join(color_dir, f"{frame_name}.jpg"), size)
                    if ref_gt is None and gt is not None and img_gray is not None:
                        ref_gt = gt
                        ref_img = img_gray
                preds.append(pred)
                gts.append(gt)
                imgs.append(img_gray)
            if preds:
                metrics.update_video(
                    preds,
                    gts,
                    imgs=imgs,
                    ref_gt=ref_gt,
                    ref_img=ref_img,
                    use_citys_sparse=use_citys_sparse,
                )
            if missing_pred > 0:
                print(f"Missing {missing_pred}/{len(frame_names)} predictions for {exp_name}")

        _write_metrics_outputs(output_root, metrics)


if __name__ == "__main__":
    main()
