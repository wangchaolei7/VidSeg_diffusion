from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


class OVDGMetrics:
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        mvc_n: Sequence[int] = (8, 16),
        tc_stride: int = 4,
    ) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.tc_stride = max(1, int(tc_stride))
        self.mvc_n_list = sorted({max(1, int(x)) for x in mvc_n})

        self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.mvc_sum = {n: 0.0 for n in self.mvc_n_list}
        self.mvc_cnt = {n: 0 for n in self.mvc_n_list}

    def update_confusion(self, pred: np.ndarray, gt: np.ndarray) -> None:
        if pred.ndim == 3:
            pred = pred[:, :, 0]
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        pred = pred.astype(np.int64)
        gt = gt.astype(np.int64)
        mask = (
            (gt != self.ignore_index)
            & (gt >= 0)
            & (gt < self.num_classes)
            & (pred >= 0)
            & (pred < self.num_classes)
        )
        if not np.any(mask):
            return
        pred = pred[mask]
        gt = gt[mask]
        idx = self.num_classes * gt + pred
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self.conf_mat += binc.reshape(self.num_classes, self.num_classes)

    def _compute_vc_dense(self, preds: List[np.ndarray], gts: List[np.ndarray], n: int) -> Optional[float]:
        length = len(preds)
        if length < n:
            return None
        vals = []
        for start in range(0, length - n + 1):
            pred_win = preds[start : start + n]
            gt_win = gts[start : start + n]

            gt0 = gt_win[0]
            gt_equal = np.ones(gt0.shape, dtype=bool)
            gt_valid = np.ones(gt0.shape, dtype=bool)
            for gt in gt_win:
                gt_valid &= (gt != self.ignore_index)
            for gt in gt_win[1:]:
                gt_equal &= (gt == gt0)
            gt_common = gt_equal & gt_valid
            denom = int(gt_common.sum())
            if denom == 0:
                continue

            pred0 = pred_win[0]
            pred_equal = np.ones(pred0.shape, dtype=bool)
            for pred in pred_win[1:]:
                pred_equal &= (pred == pred0)

            num = int((gt_common & pred_equal & (pred0 == gt0)).sum())
            vals.append(num / denom)
        if not vals:
            return None
        return float(np.mean(vals))

    def _compute_vc_sparse_valid_windows(
        self, preds: List[np.ndarray], gts: List[Optional[np.ndarray]], n: int
    ) -> Optional[float]:
        length = len(preds)
        if length < n:
            return None
        vals = []
        for start in range(0, length - n + 1):
            gt_win = gts[start : start + n]
            if any(x is None for x in gt_win):
                continue
            vc = self._compute_vc_dense(
                preds[start : start + n],
                [x for x in gt_win if x is not None],
                n,
            )
            if vc is not None:
                vals.append(vc)
        if not vals:
            return None
        return float(np.mean(vals))

    def update_video(self, preds: List[np.ndarray], gts: List[Optional[np.ndarray]]) -> None:
        if len(preds) != len(gts):
            raise ValueError("preds and gts must have same length")

        s = self.tc_stride
        preds_small = []
        gts_small: List[Optional[np.ndarray]] = []
        for pred, gt in zip(preds, gts):
            pred_small = pred[::s, ::s] if s > 1 else pred
            preds_small.append(pred_small)
            if gt is None:
                gts_small.append(None)
            else:
                gt_small = gt[::s, ::s] if s > 1 else gt
                gts_small.append(gt_small)
                self.update_confusion(pred, gt)

        dense_gt = all(x is not None for x in gts_small)
        for n in self.mvc_n_list:
            vc_val = None
            if dense_gt:
                vc_val = self._compute_vc_dense(preds_small, [x for x in gts_small if x is not None], n)
            else:
                vc_val = self._compute_vc_sparse_valid_windows(preds_small, gts_small, n)
            if vc_val is not None and not np.isnan(vc_val):
                self.mvc_sum[n] += float(vc_val)
                self.mvc_cnt[n] += 1

    def compute_summary(self) -> Dict[str, object]:
        conf = self.conf_mat.astype(np.float64)
        gt_sum = conf.sum(axis=1)
        pred_sum = conf.sum(axis=0)
        intersect = np.diag(conf)
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = intersect / gt_sum
            iou = intersect / (gt_sum + pred_sum - intersect)

        total = conf.sum()
        a_acc = float(intersect.sum() / total) if total > 0 else float("nan")

        valid_mask = ~(np.isnan(acc) | (iou == 0))
        if valid_mask.any():
            m_acc = float(np.nanmean(acc[valid_mask]))
            m_iou = float(np.nanmean(iou[valid_mask]))
        else:
            m_acc = float("nan")
            m_iou = float("nan")

        mvc_summary = {}
        for n in self.mvc_n_list:
            cnt = self.mvc_cnt[n]
            val = self.mvc_sum[n] / cnt if cnt > 0 else float("nan")
            mvc_summary[f"mVC{n}"] = val
            mvc_summary[f"mVC{n}_videos"] = int(cnt)

        return {
            "aAcc": a_acc,
            "mAcc": m_acc,
            "mIoU": m_iou,
            "per_class_iou": iou,
            "per_class_acc": acc,
            **mvc_summary,
        }

    def save_npz(self, path: str) -> None:
        np.savez(
            path,
            conf_mat=self.conf_mat,
            mvc_sum=np.array([self.mvc_sum[n] for n in self.mvc_n_list], dtype=np.float64),
            mvc_cnt=np.array([self.mvc_cnt[n] for n in self.mvc_n_list], dtype=np.int64),
            mvc_n=np.array(self.mvc_n_list, dtype=np.int64),
        )

    @staticmethod
    def merge_parts(
        part_paths: Iterable[str],
        num_classes: int,
        ignore_index: int,
        mvc_n: Sequence[int],
        tc_stride: int,
    ) -> "OVDGMetrics":
        merged = OVDGMetrics(num_classes, ignore_index=ignore_index, mvc_n=mvc_n, tc_stride=tc_stride)
        for path in part_paths:
            data = np.load(path)
            merged.conf_mat += data["conf_mat"]
            part_n = data.get("mvc_n", np.array(list(mvc_n), dtype=np.int64)).tolist()
            part_sum = data["mvc_sum"].tolist()
            part_cnt = data["mvc_cnt"].tolist()
            for n, s, c in zip(part_n, part_sum, part_cnt):
                if n in merged.mvc_sum:
                    merged.mvc_sum[n] += float(s)
                    merged.mvc_cnt[n] += int(c)
        return merged
