import os
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from .estrus_stages import EstrusStageRegistry


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _extract_subject_id(stem: str) -> str:
    parts = re.split(r"[_\-\.]", stem)
    if parts:
        return parts[0]
    return stem


class CIMFDualModalDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        eeg_transform: Optional[Callable] = None,
        audio_transform: Optional[Callable] = None,
        fold_idx: int = 0,
        total_folds: int = 5,
        image_size: int = 224,
        pairing_strategy: str = "simple",
        validation_mode: str = "classification",
    ):
        self.data_root = data_root
        self.split = split
        self.eeg_transform = eeg_transform
        self.audio_transform = audio_transform
        self.fold_idx = int(fold_idx)
        self.total_folds = int(total_folds)
        self.image_size = image_size
        self.pairing_strategy = pairing_strategy
        self.validation_mode = validation_mode
        self.stage_names = EstrusStageRegistry.get_class_names()
        self.samples: List[Tuple[str, str, int]] = []
        self._build_index()

    def _list_images(self, directory: str) -> Dict[str, str]:
        mapping = {}
        if not os.path.isdir(directory):
            return mapping
        for fname in os.listdir(directory):
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                stem = os.path.splitext(fname)[0]
                mapping[stem] = os.path.join(directory, fname)
        return mapping

    def _build_index(self):
        eeg_root = os.path.join(self.data_root, "eeg")
        audio_root = os.path.join(self.data_root, "audio")
        paired_by_subject: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        for stage_idx, stage_name in enumerate(self.stage_names):
            eeg_dir = os.path.join(eeg_root, stage_name)
            audio_dir = os.path.join(audio_root, stage_name)
            eeg_files = self._list_images(eeg_dir)
            audio_files = self._list_images(audio_dir)
            common_stems = sorted(set(eeg_files.keys()) & set(audio_files.keys()))
            for stem in common_stems:
                subject_id = _extract_subject_id(stem)
                paired_by_subject[subject_id].append(
                    (eeg_files[stem], audio_files[stem], stage_idx)
                )
        subjects = sorted(paired_by_subject.keys())
        if not subjects:
            self.samples = []
            return
        fold_buckets: List[List[str]] = [[] for _ in range(self.total_folds)]
        for idx, subject in enumerate(subjects):
            fold_buckets[idx % self.total_folds].append(subject)
        test_subjects = set(fold_buckets[self.fold_idx % self.total_folds])
        for subject, entries in paired_by_subject.items():
            in_test = subject in test_subjects
            if self.split == "train" and not in_test:
                self.samples.extend(entries)
            elif self.split in ("val", "test") and in_test:
                self.samples.extend(entries)
        if self.pairing_strategy == "balanced" and self.split == "train" and self.samples:
            by_class: Dict[int, List[Tuple[str, str, int]]] = defaultdict(list)
            for item in self.samples:
                by_class[item[2]].append(item)
            max_count = max(len(v) for v in by_class.values())
            balanced = []
            for cls in range(len(self.stage_names)):
                items = by_class.get(cls, [])
                if not items:
                    continue
                while len(items) < max_count:
                    items = items + items
                balanced.extend(items[:max_count])
            self.samples = balanced

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        eeg_path, audio_path, label = self.samples[index]
        eeg_img = Image.open(eeg_path).convert("RGB")
        audio_img = Image.open(audio_path).convert("RGB")
        if self.eeg_transform:
            eeg_img = self.eeg_transform(eeg_img)
        if self.audio_transform:
            audio_img = self.audio_transform(audio_img)
        data = {"eeg": eeg_img, "audio": audio_img}
        return data, label
