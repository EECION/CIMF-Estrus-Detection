import os
import re
from typing import Dict, List, Optional


class LogParser:
    AUC_PATTERN = re.compile(r"(?:val[_\s]?)?auc[=:\s]+([\d.]+)", re.IGNORECASE)
    ACC_PATTERN = re.compile(
        r"(?:val[_\s]?)?(?:acc(?:uracy)?)[=:\s]+([\d.]+)",
        re.IGNORECASE,
    )
    EPOCH_PATTERN = re.compile(r"epoch[\s]+(\d+)", re.IGNORECASE)
    LOSS_PATTERN = re.compile(r"(?:val[_\s]?)?loss[=:\s]+([\d.]+)", re.IGNORECASE)
    SENSITIVITY_PATTERN = re.compile(
        r"(?:val[_\s]?)?sensitivity[=:\s]+([\d.]+)",
        re.IGNORECASE,
    )
    SPECIFICITY_PATTERN = re.compile(
        r"(?:val[_\s]?)?specificity[=:\s]+([\d.]+)",
        re.IGNORECASE,
    )

    def parse_line(self, line: str) -> Dict[str, Optional[float]]:
        result = {
            "epoch": None,
            "accuracy": None,
            "auc": None,
            "loss": None,
            "sensitivity": None,
            "specificity": None,
        }
        epoch_match = self.EPOCH_PATTERN.search(line)
        if epoch_match:
            result["epoch"] = float(epoch_match.group(1))
        acc_match = self.ACC_PATTERN.search(line)
        if acc_match:
            result["accuracy"] = float(acc_match.group(1))
        auc_match = self.AUC_PATTERN.search(line)
        if auc_match:
            result["auc"] = float(auc_match.group(1))
        loss_match = self.LOSS_PATTERN.search(line)
        if loss_match:
            result["loss"] = float(loss_match.group(1))
        sens_match = self.SENSITIVITY_PATTERN.search(line)
        if sens_match:
            result["sensitivity"] = float(sens_match.group(1))
        spec_match = self.SPECIFICITY_PATTERN.search(line)
        if spec_match:
            result["specificity"] = float(spec_match.group(1))
        return result

    def parse_file(self, filepath: str) -> List[Dict[str, Optional[float]]]:
        records = []
        if not os.path.isfile(filepath):
            return records
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parsed = self.parse_line(line)
                if any(v is not None for k, v in parsed.items() if k != "epoch"):
                    records.append(parsed)
        return records

    def extract_best_metrics(self, records: List[Dict[str, Optional[float]]]) -> Dict[str, float]:
        best = {"accuracy": 0.0, "auc": 0.0}
        for rec in records:
            acc = rec.get("accuracy")
            auc = rec.get("auc")
            if acc is not None and acc > best["accuracy"]:
                best["accuracy"] = acc
            if auc is not None and auc > best["auc"]:
                best["auc"] = auc
        return best
