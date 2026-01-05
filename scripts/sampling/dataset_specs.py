from dataclasses import dataclass
from typing import List, Optional, Tuple
import os


@dataclass
class DatasetSpec:
    name: str
    color_root: str
    mask_root: str
    mask_suffix: str
    mask_ext: str
    output_root: Optional[str] = None
    split_file: Optional[str] = None


def _read_split_file(split_file: str) -> List[str]:
    exp_names: List[str] = []
    with open(split_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            exp_names.append(line.split()[0])
    return exp_names


def _vspw_sort_key(name: str):
    try:
        return (0, int(name.split(".")[0].split("_")[0]))
    except ValueError:
        return (1, name)


def build_dataset_spec(args) -> DatasetSpec:
    dataset = getattr(args, "dataset", "vspw")
    if dataset == "apollo":
        dataset_root = args.dataset_root or "/home/wangcl/data/open_video_DGSS/ApolloScape"
        color_root = args.color_root or os.path.join(dataset_root, "val", "ColorImage")
        mask_root = args.mask_root or os.path.join(dataset_root, "val", "15Label")
        mask_suffix = args.mask_suffix if args.mask_suffix != "" else "_bin"
        split_file = getattr(args, "split_file_path", None)
    elif dataset == "camvid":
        dataset_root = args.dataset_root or "/home/wangcl/data/open_video_DGSS/CamVid"
        color_root = args.color_root or os.path.join(dataset_root, "val", "images")
        mask_root = args.mask_root or os.path.join(dataset_root, "val", "15labels")
        mask_suffix = args.mask_suffix if args.mask_suffix != "" else "_L"
        split_file = getattr(args, "split_file_path", None)
    else:
        color_root = args.color_root or args.dataset_path
        mask_root = args.mask_root or args.dataset_path
        mask_suffix = args.mask_suffix
        split_file = args.split_file_path

    return DatasetSpec(
        name=dataset,
        color_root=color_root,
        mask_root=mask_root,
        mask_suffix=mask_suffix,
        mask_ext=args.mask_ext,
        output_root=args.output_root,
        split_file=split_file,
    )


def list_sequences(spec: DatasetSpec) -> List[Tuple[str, str, str]]:
    if spec.split_file and os.path.isfile(spec.split_file):
        exp_names = _read_split_file(spec.split_file)
    else:
        exp_names = [
            name
            for name in os.listdir(spec.color_root)
            if os.path.isdir(os.path.join(spec.color_root, name))
        ]

    if spec.name == "vspw":
        exp_names.sort(key=_vspw_sort_key)
        return [
            (
                exp_name,
                os.path.join(spec.color_root, exp_name, "origin"),
                os.path.join(spec.mask_root, exp_name, "mask"),
            )
            for exp_name in exp_names
        ]

    exp_names = sorted(exp_names)
    return [
        (
            exp_name,
            os.path.join(spec.color_root, exp_name),
            os.path.join(spec.mask_root, exp_name),
        )
        for exp_name in exp_names
    ]


def resolve_gt_mask_path(mask_dir: Optional[str], frame_name: str, suffix: str, ext: str) -> Optional[str]:
    if not mask_dir:
        return None
    gt_mask_path = os.path.join(mask_dir, f"{frame_name}{suffix}{ext}")
    if not os.path.isfile(gt_mask_path):
        return None
    return gt_mask_path
