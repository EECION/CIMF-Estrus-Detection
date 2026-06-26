import os
import sys
import time
import math
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, Callable
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from torchvision import transforms

try:
    from ..utils.dual_modal_dataset import CIMFDualModalDataset
    from ..models.cimf_model import CIMFModel
    from ..utils.training_utils import MetricsCalculator
    from ..utils.logger import get_logger
    from ..utils.estrus_stages import EstrusStageRegistry
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import GUI trainer modules: {e}")
    print(f"Python path: {sys.path}")
    traceback.print_exc()
    IMPORTS_AVAILABLE = False

from .checkpoint_manager import CheckpointManager

NUM_CLASSES = 3
TOTAL_FOLDS = 5


def calculate_metrics(outputs, targets):
    with torch.no_grad():
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('predictions', outputs))
        else:
            logits = outputs
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == targets).float().mean().item()
        return {
            'accuracy': accuracy * 100.0,
            'loss': F.cross_entropy(logits, targets).item(),
        }


class TrainingWorker(QThread):

    epoch_finished = pyqtSignal(int, dict)
    training_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int, str)
    log_message = pyqtSignal(str)

    def __init__(self, config: Dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.logger = get_logger()
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'loss_components': {},
        }
        self.epoch_records = []
        self.is_running = False
        self.should_stop = False
        self.stop_timeout_timer = None
        self.checkpoint_manager = CheckpointManager()
        self.is_data_parallel = False

    def _resolve_fold_idx(self) -> int:
        test_fold = self.config.get('test_fold')
        if test_fold is None:
            test_fold = self.config.get('test_class', 0)
        if isinstance(test_fold, str):
            stripped = test_fold.strip()
            if stripped.upper().startswith('F'):
                try:
                    return int(stripped[1:])
                except ValueError:
                    pass
            try:
                return int(stripped)
            except ValueError:
                return 0
        return int(test_fold)

    def setup_model(self):
        try:
            if not IMPORTS_AVAILABLE:
                raise ImportError("Required training modules are unavailable")

            self.logger.info("Creating CIMF model...", "TrainingWorker")
            backbone = self.config.get('backbone', 'ResNet50-512')
            self.model = CIMFModel(
                backbone=backbone,
                num_classes=NUM_CLASSES,
                lambda_in=self.config.get('lambda_in', 1.0),
                lambda_cross=self.config.get('lambda_cross', 1.0),
                lambda_hybrid=self.config.get('lambda_hybrid', 1.0),
                lambda_single=self.config.get('lambda_single', 1.0),
                lambda_causal=self.config.get('lambda_causal', 1.0),
                lambda_cls=self.config.get('lambda_cls', 1.0),
                gamma=self.config.get('gamma', 1.0),
                tau=self.config.get('tau', 0.4),
            )
            self.model = self.model.to(self.device)

            if self.config.get('use_pretrained', True):
                from ..utils.model_utils import ModelUtils
                model_utils = ModelUtils()
                weights_path = self.config.get('weights_path', '')
                if weights_path and os.path.exists(weights_path):
                    self.logger.info(f"Loading weights from {weights_path}", "TrainingWorker")
                    success = model_utils.load_pretrained_weights(
                        self.model, backbone, weights_path=weights_path
                    )
                else:
                    self.logger.info("Attempting to load pretrained backbone weights...", "TrainingWorker")
                    success = model_utils.load_pretrained_weights(
                        self.model,
                        backbone,
                        weights_dir=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'weights'
                        ),
                    )
                if success:
                    self.logger.success("Pretrained weights loaded", "TrainingWorker")
                else:
                    self.logger.warning("Pretrained weights unavailable, using random init", "TrainingWorker")

            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                self.logger.info(
                    f"Detected {torch.cuda.device_count()} GPUs, enabling DataParallel",
                    "TrainingWorker",
                )
                self.model = nn.DataParallel(self.model)
                self.is_data_parallel = True
            else:
                self.is_data_parallel = False
                if torch.cuda.is_available():
                    self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}", "TrainingWorker")
                else:
                    self.logger.info("Using CPU training", "TrainingWorker")

        except Exception as e:
            self.logger.error(f"Model creation failed: {e}", "TrainingWorker")
            self._cleanup_resources()
            raise RuntimeError(f"Failed to create model: {e}")

    def setup_data(self):
        try:
            if not IMPORTS_AVAILABLE:
                raise ImportError("Required training modules are unavailable")

            self.logger.info("Setting up data loaders...", "TrainingWorker")
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            fold_idx = self._resolve_fold_idx()
            self.logger.info(f"Cross-subject fold index: {fold_idx}", "TrainingWorker")

            train_dataset = CIMFDualModalDataset(
                data_root=self.config.get('data_path', './data'),
                split='train',
                eeg_transform=train_transform,
                audio_transform=train_transform,
                fold_idx=fold_idx,
                total_folds=TOTAL_FOLDS,
                image_size=224,
                pairing_strategy='balanced',
            )

            val_dataset = CIMFDualModalDataset(
                data_root=self.config.get('data_path', './data'),
                split='val',
                eeg_transform=val_transform,
                audio_transform=val_transform,
                fold_idx=fold_idx,
                total_folds=TOTAL_FOLDS,
                image_size=224,
                pairing_strategy='simple',
                validation_mode='classification',
            )

            if len(val_dataset) == 0:
                self.logger.warning("Validation set empty, using train subset", "TrainingWorker")
                train_size = len(train_dataset)
                val_size = max(50, int(train_size * 0.2))
                indices = np.random.permutation(train_size)[:val_size]
                val_dataset = Subset(train_dataset, indices)

            batch_size = self.config.get('batch_size', 128)
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            self.logger.success(
                f"Data loaders ready - train: {len(train_dataset)}, val: {len(val_dataset)}",
                "TrainingWorker",
            )

        except Exception as e:
            self.logger.error(f"Data loader setup failed: {e}", "TrainingWorker")
            self._cleanup_resources()
            raise RuntimeError(f"Failed to create data loaders: {e}")

    def setup_optimizer(self):
        try:
            self.logger.info("Setting up optimizer...", "TrainingWorker")
            param_groups = self._get_parameter_groups()
            optimizer_type = self.config.get('optimizer', 'AdamW')
            lr = self.config.get('lr', 1e-4)
            weight_decay = self.config.get('weight_decay', 5e-3)

            if optimizer_type == 'Adam':
                self.optimizer = optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
            elif optimizer_type == 'SGD':
                self.optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9, weight_decay=weight_decay)
            elif optimizer_type == 'AdamW':
                self.optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_type}")

            scheduler_type = self.config.get('scheduler', 'CosineAnnealingLR')
            epochs = self.config.get('epochs', 50)
            scheduler_params = self.config.get('scheduler_params', {})

            if scheduler_type == 'StepLR':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_params.get('step_size', 20),
                    gamma=scheduler_params.get('gamma', 0.1),
                )
            elif scheduler_type in ('CosineAnnealingLR', 'Cosine'):
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=epochs,
                    eta_min=scheduler_params.get('eta_min', 0),
                )
            elif scheduler_type == 'ReduceLROnPlateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=scheduler_params.get('factor', 0.1),
                    patience=scheduler_params.get('patience', self.config.get('patience', 20) // 2),
                    verbose=True,
                )
            else:
                self.scheduler = None

            self.logger.success(
                f"Optimizer ready: {optimizer_type} lr={lr} weight_decay={weight_decay}",
                "TrainingWorker",
            )

        except Exception as e:
            self.logger.error(f"Optimizer setup failed: {e}", "TrainingWorker")
            self._cleanup_resources()
            raise RuntimeError(f"Failed to create optimizer: {e}")

    def _get_parameter_groups(self):
        decay_params = []
        no_decay_params = []
        weight_decay = self.config.get('weight_decay', 5e-3)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith('.bias') or 'norm' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

    def load_checkpoint(self):
        base_dir = self.config.get('checkpoint_path', './checkpoints')
        try:
            self._ensure_artifact_dirs()
            check_dir = getattr(self, '_artifact_dirs', {}).get('Checkpoints', base_dir)
        except Exception:
            check_dir = base_dir
        ck_file = self.config.get('checkpoint_file')
        if ck_file:
            checkpoint_path = os.path.join(check_dir, ck_file)
        else:
            cand = os.path.join(check_dir, 'latest.pth')
            if not os.path.exists(cand):
                cand = os.path.join(base_dir, 'latest.pth')
            checkpoint_path = cand

        if os.path.exists(checkpoint_path):
            try:
                checkpoint_info = self.checkpoint_manager.load_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    map_location=str(self.device),
                )
                if checkpoint_info:
                    self.current_epoch = checkpoint_info.get('epoch', 0)
                    metrics = checkpoint_info.get('metrics', {})
                    self.best_val_acc = metrics.get('val_acc', 0.0)
                    self.best_epoch = metrics.get('best_epoch', 0)
                    self.logger.info(f"Checkpoint loaded: {checkpoint_path}", "TrainingWorker")
                    self.logger.info(f"Resuming from epoch {self.current_epoch + 1}", "TrainingWorker")
                else:
                    self.logger.error(f"Failed to load checkpoint: {checkpoint_path}", "TrainingWorker")
            except Exception as e:
                self.logger.error(f"Checkpoint load error: {e}", "TrainingWorker")

    def _extract_batch_loss(self, model_output):
        if 'total_loss' in model_output:
            total_loss_batch = model_output['total_loss']
            losses = {k: v for k, v in model_output.items() if k.startswith('loss') and k != 'total_loss'}
        else:
            losses = {k: v for k, v in model_output.items() if k.startswith('loss')}
            total_loss_batch = sum(losses.values()) if losses else torch.tensor(0.0, device=self.device)
        return total_loss_batch, losses

    def train_epoch(self) -> Dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        loss_components = {
            'in_modal': 0.0,
            'cross_modal': 0.0,
            'hybrid': 0.0,
            'single': 0.0,
            'causal': 0.0,
            'classification': 0.0,
        }
        fine_loss_sums = {
            'loss_in': 0.0,
            'loss_cross_m1': 0.0,
            'loss_cross_m2': 0.0,
            'loss_hybrid': 0.0,
            'loss_single_m1': 0.0,
            'loss_single_m2': 0.0,
            'loss_causal_m1': 0.0,
            'loss_causal_m2': 0.0,
            'loss_cls': 0.0,
        }
        last_progress_batch = 0
        last_progress_time = time.time()
        seen_batches = 0
        self._train_stall_count = 0
        num_batches = len(self.train_loader)
        self.accumulation_steps = int(self.config.get('accumulation_steps', 4))
        self.grad_scale_factor = 1.0 / self.accumulation_steps

        if num_batches == 0:
            self.logger.warning("Train loader empty, skipping epoch", "TrainingWorker")
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'loss_components': {k: 0.0 for k in loss_components},
            }

        try:
            self.optimizer.zero_grad()
        except Exception:
            pass

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            if self.should_stop:
                remaining = (batch_idx + 1) % self.accumulation_steps
                if remaining != 0:
                    try:
                        if self.config.get('grad_clip', 1.0) > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    except Exception as e:
                        self.logger.error(f"Gradient flush on stop failed: {e}", "TrainingWorker")
                break

            now_ts = time.time()
            if now_ts - last_progress_time > 60:
                self._train_stall_count = getattr(self, '_train_stall_count', 0) + 1
                self.logger.info(
                    f"Train batch stall {int(now_ts - last_progress_time)}s at batch {batch_idx}",
                    "TrainingWorker",
                )
                last_progress_time = now_ts
                if self._train_stall_count >= 2:
                    self.logger.warning("Train stall limit reached, ending epoch early", "TrainingWorker")
                    break

            try:
                if isinstance(data, dict):
                    data = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in data.items()}
                elif hasattr(data, 'to'):
                    data = data.to(self.device)
                targets = targets.to(self.device)
            except Exception as e:
                self.logger.error(f"Device transfer failed: {e}", "TrainingWorker")
                continue

            if isinstance(data, dict) and 'eeg' in data and 'audio' in data:
                data_dict = {'eeg': data['eeg'], 'audio': data['audio']}
                model_output = self.model(data_dict, targets)
                outputs = model_output['logits']
                total_loss_batch, losses = self._extract_batch_loss(model_output)
                if torch.isnan(total_loss_batch).any() or torch.isinf(total_loss_batch).any():
                    self.logger.warning("Unstable loss detected, skipping batch", "TrainingWorker")
                    self.optimizer.zero_grad()
                    continue
            else:
                input_data = data.get('eeg', list(data.values())[0]) if isinstance(data, dict) else data
                model_output = self.model({'eeg': input_data, 'audio': input_data})
                outputs = model_output['logits']
                losses = {'loss_cls': F.cross_entropy(outputs, targets)}
                total_loss_batch = losses['loss_cls']

            raw_total_loss = (
                total_loss_batch.mean()
                if (total_loss_batch.dim() > 0 and total_loss_batch.numel() > 1)
                else total_loss_batch
            )
            loss_cap = self.config.get('loss_cap', 10.0)
            if self.config.get('enable_loss_cap', True) and raw_total_loss.item() > loss_cap:
                raw_total_loss = raw_total_loss * (loss_cap / raw_total_loss.item())

            try:
                scaled_loss = raw_total_loss * self.grad_scale_factor
                scaled_loss.backward()
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    grad_clip = self.config.get('grad_clip', 1.0)
                    if grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                        grad_norm_item = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
                        if math.isnan(grad_norm_item) or math.isinf(grad_norm_item) or grad_norm_item > 200.0:
                            self.optimizer.zero_grad()
                        else:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    else:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
            except RuntimeError as e:
                self.logger.error(f"Backward pass failed: {e}", "TrainingWorker")
                self.optimizer.zero_grad()

            total_loss += raw_total_loss.item()
            seen_batches += 1
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            batch_loss_vals = {}
            for k, v in losses.items():
                if hasattr(v, 'dim') and v.dim() > 0 and v.numel() > 1:
                    batch_loss_vals[k] = v.mean().item()
                else:
                    batch_loss_vals[k] = float(v.item()) if hasattr(v, 'item') else float(v)

            for fine_key in fine_loss_sums:
                if fine_key in batch_loss_vals:
                    fine_loss_sums[fine_key] += batch_loss_vals[fine_key]

            if 'loss_in' in batch_loss_vals:
                loss_components['in_modal'] += batch_loss_vals['loss_in']
            cross_vals = [batch_loss_vals[k] for k in ('loss_cross_m1', 'loss_cross_m2') if k in batch_loss_vals]
            if cross_vals:
                loss_components['cross_modal'] += sum(cross_vals) / len(cross_vals)
            if 'loss_hybrid' in batch_loss_vals:
                loss_components['hybrid'] += batch_loss_vals['loss_hybrid']
            single_vals = [batch_loss_vals[k] for k in ('loss_single_m1', 'loss_single_m2') if k in batch_loss_vals]
            if single_vals:
                loss_components['single'] += sum(single_vals) / len(single_vals)
            causal_vals = [batch_loss_vals[k] for k in ('loss_causal_m1', 'loss_causal_m2') if k in batch_loss_vals]
            if causal_vals:
                loss_components['causal'] += sum(causal_vals) / len(causal_vals)
            if 'loss_cls' in batch_loss_vals:
                loss_components['classification'] += batch_loss_vals['loss_cls']

            force_output = (batch_idx + 1 - last_progress_batch) >= 50 or (batch_idx + 1) == num_batches
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches or force_output:
                batch_acc = 100.0 * predicted.eq(targets).sum().item() / targets.size(0)
                display_loss = raw_total_loss.mean().item() if raw_total_loss.dim() > 0 else raw_total_loss.item()
                self.logger.info(
                    f"Epoch {self.current_epoch + 1} Batch {batch_idx + 1}/{num_batches} "
                    f"Loss={display_loss:.4f} Acc={batch_acc:.2f}%",
                    "TrainingWorker",
                )
                last_progress_batch = batch_idx + 1
                last_progress_time = time.time()
                self._train_stall_count = 0

            self.progress_updated.emit(
                batch_idx + 1,
                num_batches,
                f"Training epoch {self.current_epoch + 1}: {batch_idx + 1}/{num_batches}",
            )

        if num_batches % self.accumulation_steps != 0:
            try:
                grad_clip = self.config.get('grad_clip', 1.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
            except Exception as e:
                self.logger.error(f"Final gradient step failed: {e}", "TrainingWorker")

        denom = max(1, seen_batches)
        for key in loss_components:
            loss_components[key] /= denom
        fine_loss_avg = {k: v / denom for k, v in fine_loss_sums.items()}
        return {
            'loss': total_loss / denom,
            'accuracy': 100.0 * correct / total if total > 0 else 0.0,
            'loss_components': loss_components,
            'loss_components_fine': fine_loss_avg,
        }

    def validate_epoch(self) -> Dict:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        val_seen_batches = 0
        last_progress_time = time.time()
        total_val_batches = len(self.val_loader)
        val_start_time = time.time()
        log_every_batches = max(10, max(1, total_val_batches // 20))
        all_logits = []
        all_predictions = []
        all_targets = []
        all_eeg_features = []
        all_audio_features = []

        default_metrics = {
            'accuracy': 0.0,
            'loss': 0.0,
            'auc': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
        }

        if total_val_batches == 0:
            self.logger.warning("Validation loader empty, skipping validation", "TrainingWorker")
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'predictions': [],
                'targets': [],
                'detailed_metrics': default_metrics,
            }

        metrics_calculator = MetricsCalculator(num_classes=NUM_CLASSES)

        with torch.no_grad():
            val_timeout_sec = int(self.config.get('val_timeout_sec', 60))
            val_max_stall = int(self.config.get('val_max_stall', 2))
            for batch_idx, (data, targets) in enumerate(self.val_loader):
                if self.should_stop:
                    break

                now_ts = time.time()
                if now_ts - last_progress_time > val_timeout_sec:
                    self._val_stall_count = getattr(self, '_val_stall_count', 0) + 1
                    self.logger.info(
                        f"Validation stall {int(now_ts - last_progress_time)}s at batch {batch_idx}",
                        "TrainingWorker",
                    )
                    last_progress_time = now_ts
                    if self._val_stall_count >= val_max_stall:
                        self.logger.warning("Validation stall limit reached", "TrainingWorker")
                        break

                try:
                    if isinstance(data, dict):
                        data = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in data.items()}
                    elif hasattr(data, 'to'):
                        data = data.to(self.device)
                    targets = targets.to(self.device)
                except Exception as e:
                    self.logger.error(f"Validation device transfer failed: {e}", "TrainingWorker")
                    continue

                if isinstance(data, dict) and 'eeg' in data and 'audio' in data:
                    data_dict = {'eeg': data['eeg'], 'audio': data['audio']}
                    model_output = self.model(data_dict, targets)
                    outputs = model_output['logits']
                    total_loss_batch, _ = self._extract_batch_loss(model_output)
                    eeg_features = model_output.get('orig_eeg_features')
                    if eeg_features is None:
                        eeg_features = model_output.get('enh_eeg_features')
                    audio_features = model_output.get('orig_audio_features')
                    if audio_features is None:
                        audio_features = model_output.get('enh_audio_features')
                    if eeg_features is not None and audio_features is not None:
                        all_eeg_features.append(eeg_features.detach().cpu())
                        all_audio_features.append(audio_features.detach().cpu())
                else:
                    input_data = data.get('eeg', list(data.values())[0]) if isinstance(data, dict) else data
                    model_output = self.model({'eeg': input_data, 'audio': input_data})
                    outputs = model_output['logits']
                    total_loss_batch = F.cross_entropy(outputs, targets)
                    eeg_features = None
                    audio_features = None

                loss_scalar = (
                    total_loss_batch.mean().item()
                    if (total_loss_batch.dim() > 0 and total_loss_batch.numel() > 1)
                    else total_loss_batch.item()
                )
                total_loss += loss_scalar
                val_seen_batches += 1

                if (batch_idx + 1) % log_every_batches == 0 or (batch_idx + 1) == total_val_batches:
                    elapsed = time.time() - val_start_time
                    pct = (batch_idx + 1) / max(1, total_val_batches) * 100.0
                    speed = (batch_idx + 1) / max(elapsed, 1e-6)
                    eta_sec = max(0, total_val_batches - (batch_idx + 1)) / max(speed, 1e-6)
                    self.logger.info(
                        f"Validation {batch_idx + 1}/{total_val_batches} ({pct:.1f}%) "
                        f"speed={speed:.2f} batch/s eta={eta_sec:.1f}s",
                        "TrainingWorker",
                    )

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                all_logits.append(outputs.detach().cpu())
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        denom = max(1, val_seen_batches)
        avg_loss = total_loss / denom
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        detailed_metrics = {'accuracy': accuracy, 'loss': avg_loss}

        if all_logits:
            try:
                logits_tensor = torch.cat(all_logits, dim=0)
            except Exception as e:
                self.logger.error(f"Logits concat failed: {e}", "TrainingWorker")
                logits_tensor = torch.vstack([t for t in all_logits if t.ndim == 2])
            targets_tensor = torch.as_tensor(all_targets)
        else:
            logits_tensor = torch.empty(0, NUM_CLASSES)
            targets_tensor = torch.as_tensor(all_targets)

        if all_eeg_features and all_audio_features:
            eeg_features = torch.cat(all_eeg_features, dim=0)
            audio_features = torch.cat(all_audio_features, dim=0)
            metrics_calculator.update(
                predictions=logits_tensor,
                targets=targets_tensor,
                eeg_features=eeg_features,
                audio_features=audio_features,
            )
            detailed_metrics.update(metrics_calculator.get_all_metrics())
            eeg_labels_np = np.array(all_targets, dtype=np.int64)
            audio_labels_np = np.array(all_targets, dtype=np.int64)
        else:
            try:
                if logits_tensor.numel() > 0 and targets_tensor.numel() > 0:
                    detailed_metrics.update(calculate_metrics(logits_tensor, targets_tensor))
                else:
                    detailed_metrics.update({'auc': 0.0, 'sensitivity': 0.0, 'specificity': 0.0})
            except Exception:
                detailed_metrics.update({'auc': 0.0, 'sensitivity': 0.0, 'specificity': 0.0})
            eeg_features = torch.empty(0, 0)
            audio_features = torch.empty(0, 0)
            eeg_labels_np = np.array([], dtype=np.int64)
            audio_labels_np = np.array([], dtype=np.int64)

        try:
            total_time = time.time() - val_start_time
            covered_pct = (val_seen_batches / max(1, total_val_batches)) * 100.0
            self.logger.info(
                f"Validation done: {val_seen_batches}/{total_val_batches} batches "
                f"({covered_pct:.1f}%) in {total_time:.1f}s",
                "TrainingWorker",
            )
        except Exception:
            pass

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'detailed_metrics': detailed_metrics,
            'eeg_features': eeg_features,
            'audio_features': audio_features,
            'eeg_labels': eeg_labels_np,
            'audio_labels': audio_labels_np,
        }

    def save_checkpoint(self, is_best: bool = False):
        if not self.config.get('auto_save', True):
            return
        checkpoint_dir = self.config.get('checkpoint_path', './checkpoints')
        try:
            self._ensure_artifact_dirs()
            check_dir = getattr(self, '_artifact_dirs', {}).get('Checkpoints', checkpoint_dir)
            os.makedirs(check_dir, exist_ok=True)
        except Exception:
            check_dir = checkpoint_dir

        metrics = {'val_acc': self.best_val_acc, 'best_epoch': self.best_epoch}
        latest_path = os.path.join(check_dir, 'latest.pth')
        success = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            loss=self.train_history['train_loss'][-1] if self.train_history['train_loss'] else 0.0,
            metrics=metrics,
            save_path=latest_path,
            config=self.config,
        )

        if success:
            if is_best:
                bb_code = self._map_backbone_code(self.config.get('backbone'))
                lam_code = self._map_lambda_code()
                best_path = os.path.join(check_dir, f"{bb_code}_{lam_code}_e{self.current_epoch + 1}_best.pth")
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=self.current_epoch,
                    loss=self.train_history['train_loss'][-1] if self.train_history['train_loss'] else 0.0,
                    metrics=metrics,
                    save_path=best_path,
                    config=self.config,
                )
                self.logger.success(f"Best model saved: {best_path}", "TrainingWorker")

            save_interval = self.config.get('save_interval', 10)
            if (self.current_epoch + 1) % save_interval == 0:
                bb_code = self._map_backbone_code(self.config.get('backbone'))
                lam_code = self._map_lambda_code()
                epoch_path = os.path.join(check_dir, f"{bb_code}_{lam_code}_e{self.current_epoch + 1}_last.pth")
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=self.current_epoch,
                    loss=self.train_history['train_loss'][-1] if self.train_history['train_loss'] else 0.0,
                    metrics=metrics,
                    save_path=epoch_path,
                    config=self.config,
                )

    def run(self):
        try:
            self.is_running = True
            self.should_stop = False
            self.logger.info("Initializing training environment...", "TrainingWorker")
            self.setup_model()
            self.setup_data()
            self.setup_optimizer()
            self.load_checkpoint()
            self.logger.success("Training environment ready", "TrainingWorker")
            try:
                self._ensure_artifact_dirs()
            except Exception as e:
                self.logger.warning(f"Artifact dir init failed: {e}", "TrainingWorker")

            start_time = time.time()
            patience_counter = 0
            epochs = self.config.get('epochs', 50)
            patience = self.config.get('patience', 20)

            for epoch in range(self.current_epoch, epochs):
                if self.should_stop:
                    break

                self.current_epoch = epoch
                epoch_start_time = time.time()
                self.logger.info(f"Starting epoch {epoch + 1}/{epochs}", "TrainingWorker")

                train_metrics = self.train_epoch()
                val_metrics = self.validate_epoch()
                self._last_val_payload = val_metrics

                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        detailed_for_sched = val_metrics.get('detailed_metrics', {})
                        sched_acc = detailed_for_sched.get('accuracy', val_metrics['accuracy'])
                        self.scheduler.step(sched_acc)
                    else:
                        self.scheduler.step()

                detailed = val_metrics.get('detailed_metrics', {})
                effective_val_acc = detailed.get('accuracy', val_metrics['accuracy'])

                self.train_history['train_loss'].append(train_metrics['loss'])
                self.train_history['val_loss'].append(val_metrics['loss'])
                self.train_history['train_acc'].append(train_metrics['accuracy'])
                self.train_history['val_acc'].append(effective_val_acc)

                for key, value in train_metrics['loss_components'].items():
                    if key not in self.train_history['loss_components']:
                        self.train_history['loss_components'][key] = []
                    self.train_history['loss_components'][key].append(value)

                is_best = effective_val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = effective_val_acc
                    self.best_epoch = epoch
                    patience_counter = 0
                    self.logger.success(
                        f"New best validation accuracy: {effective_val_acc:.2f}%",
                        "TrainingWorker",
                    )
                else:
                    patience_counter += 1

                self.save_checkpoint(is_best)
                epoch_time = time.time() - epoch_start_time
                self.logger.info(
                    f"Epoch {epoch + 1} done - train_loss={train_metrics['loss']:.4f} "
                    f"val_acc={effective_val_acc:.2f}% time={epoch_time:.1f}s",
                    "TrainingWorker",
                )

                epoch_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_acc': effective_val_acc,
                    'val_auc': detailed.get('auc'),
                    'val_sensitivity': detailed.get('sensitivity'),
                    'val_specificity': detailed.get('specificity'),
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time,
                    'is_best': is_best,
                    'loss_components': train_metrics['loss_components'],
                    'loss_components_fine': train_metrics.get('loss_components_fine'),
                }
                self.epoch_records.append(epoch_metrics)
                self.epoch_finished.emit(epoch + 1, epoch_metrics)

                if (epoch + 1) % 5 == 0:
                    try:
                        self._export_tsne_plot(self._last_val_payload or {}, epoch_label=f"e{epoch + 1}")
                    except Exception as e:
                        self.logger.warning(f"t-SNE export skipped: {e}", "TrainingWorker")

                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}", "TrainingWorker")
                    break

            total_time = time.time() - start_time
            final_metrics = {
                'total_epochs': self.current_epoch + 1,
                'best_val_acc': self.best_val_acc,
                'best_epoch': self.best_epoch + 1,
                'total_time': total_time,
                'avg_epoch_time': total_time / max(1, self.current_epoch + 1),
                'train_history': self.train_history,
            }
            self.training_finished.emit(final_metrics)

            try:
                self._export_tsne_plot(self._last_val_payload or {}, epoch_label="last")
            except Exception as e:
                self.logger.warning(f"Final t-SNE export skipped: {e}", "TrainingWorker")

            try:
                import csv as _csv
                save_dir = self.config.get('checkpoint_path', './checkpoints') or './checkpoints'
                os.makedirs(save_dir, exist_ok=True)
                self._ensure_artifact_dirs()
                run_dirs = getattr(self, '_artifact_dirs', {})
                logs_dir = run_dirs.get('Logs_CSV', save_dir)
                os.makedirs(logs_dir, exist_ok=True)
                fold_code = self._map_class_code()
                bb_code = self._map_backbone_code(self.config.get('backbone'))
                lam_code = self._map_lambda_code()
                ts = getattr(self, '_artifact_ts', time.strftime("%Y%m%d_%H%M"))
                csv_path = os.path.join(logs_dir, f"{fold_code}_{bb_code}_{lam_code}_train_log_{ts}.csv")
                fieldnames = [
                    'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                    'val_auc', 'val_sensitivity', 'val_specificity', 'lr', 'epoch_time', 'is_best',
                    'in_modal', 'cross_modal', 'hybrid', 'single', 'causal', 'classification',
                    'loss_in', 'loss_cross_m1', 'loss_cross_m2', 'loss_hybrid',
                    'loss_single_m1', 'loss_single_m2', 'loss_causal_m1', 'loss_causal_m2', 'loss_cls',
                ]
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = _csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for rec in self.epoch_records:
                        row = {
                            'epoch': rec.get('epoch'),
                            'train_loss': rec.get('train_loss'),
                            'val_loss': rec.get('val_loss'),
                            'train_acc': rec.get('train_acc'),
                            'val_acc': rec.get('val_acc'),
                            'val_auc': rec.get('val_auc'),
                            'val_sensitivity': rec.get('val_sensitivity'),
                            'val_specificity': rec.get('val_specificity'),
                            'lr': rec.get('lr'),
                            'epoch_time': rec.get('epoch_time'),
                            'is_best': rec.get('is_best'),
                            'in_modal': (rec.get('loss_components') or {}).get('in_modal'),
                            'cross_modal': (rec.get('loss_components') or {}).get('cross_modal'),
                            'hybrid': (rec.get('loss_components') or {}).get('hybrid'),
                            'single': (rec.get('loss_components') or {}).get('single'),
                            'causal': (rec.get('loss_components') or {}).get('causal'),
                            'classification': (rec.get('loss_components') or {}).get('classification'),
                        }
                        lcf = rec.get('loss_components_fine') or {}
                        row.update({k: lcf.get(k) for k in (
                            'loss_in', 'loss_cross_m1', 'loss_cross_m2', 'loss_hybrid',
                            'loss_single_m1', 'loss_single_m2', 'loss_causal_m1', 'loss_causal_m2', 'loss_cls',
                        )})
                        writer.writerow(row)
                self.logger.success(f"Training log exported: {csv_path}", "TrainingWorker")
            except Exception as e:
                self.logger.warning(f"CSV export skipped: {e}", "TrainingWorker")

            try:
                payload = self._last_val_payload or {}
                predictions = payload.get('predictions', [])
                targets = payload.get('targets', [])
                if predictions and targets:
                    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                    for pred, tgt in zip(predictions, targets):
                        if 0 <= int(tgt) < NUM_CLASSES and 0 <= int(pred) < NUM_CLASSES:
                            confusion[int(tgt), int(pred)] += 1
                    self._ensure_artifact_dirs()
                    run_dirs = getattr(self, '_artifact_dirs', {})
                    heat_dir = run_dirs.get('Heatmaps', self.config.get('checkpoint_path', './checkpoints'))
                    os.makedirs(heat_dir, exist_ok=True)
                    fold_code = self._map_class_code()
                    bb_code = self._map_backbone_code(self.config.get('backbone'))
                    lam_code = self._map_lambda_code()
                    csv2 = os.path.join(heat_dir, f"{fold_code}_{bb_code}_{lam_code}_confusion.csv")
                    stage_names = EstrusStageRegistry.get_class_names()
                    import csv as _csv2
                    with open(csv2, 'w', newline='', encoding='utf-8') as f:
                        writer = _csv2.writer(f)
                        header = ['true\\pred'] + stage_names
                        writer.writerow(header)
                        for i, stage in enumerate(stage_names):
                            writer.writerow([stage] + confusion[i].tolist())
                    self.logger.success(f"Confusion matrix exported: {csv2}", "TrainingWorker")

                    eeg_feats = payload.get('eeg_features')
                    audio_feats = payload.get('audio_features')
                    eeg_labels = payload.get('eeg_labels', np.array([], dtype=np.int64))
                    if (
                        eeg_feats is not None
                        and audio_feats is not None
                        and hasattr(eeg_feats, 'numel')
                        and eeg_feats.numel() > 0
                    ):
                        eeg_np = eeg_feats.detach().cpu().numpy()
                        audio_np = audio_feats.detach().cpu().numpy()
                        eeg_lbl = np.array(eeg_labels, dtype=np.int64)
                        centers = {}
                        for cls in range(NUM_CLASSES):
                            mask = eeg_lbl == cls
                            if mask.any():
                                centers[cls] = eeg_np[mask].mean(axis=0)
                        fold_idx = self._resolve_fold_idx()
                        if eeg_lbl.size > 0 and (eeg_lbl == fold_idx % NUM_CLASSES).any():
                            eeg_mean = eeg_np[eeg_lbl == fold_idx % NUM_CLASSES].mean(axis=0)
                        else:
                            eeg_mean = eeg_np.mean(axis=0)

                        def _cos(a, b):
                            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
                            return float(np.dot(a, b) / denom) if denom > 0 else 0.0

                        row = [_cos(eeg_mean, centers[cid]) if cid in centers else float('nan') for cid in range(NUM_CLASSES)]
                        sim_path = os.path.join(heat_dir, f"{fold_code}_{bb_code}_{lam_code}_similarity.csv")
                        with open(sim_path, 'w', newline='', encoding='utf-8') as f:
                            writer = _csv2.writer(f)
                            writer.writerow(stage_names)
                            writer.writerow(row)
                        self.logger.success(f"Similarity vector exported: {sim_path}", "TrainingWorker")
                else:
                    self.logger.warning("Confusion export skipped: no predictions collected", "TrainingWorker")
            except Exception as e:
                self.logger.warning(f"Heatmap export skipped: {e}", "TrainingWorker")

            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            self.logger.success("Training complete", "TrainingWorker")
            self.logger.info(f"Total epochs: {self.current_epoch + 1}", "TrainingWorker")
            self.logger.success(
                f"Best validation accuracy: {self.best_val_acc:.2f}% (epoch {self.best_epoch + 1})",
                "TrainingWorker",
            )
            self.logger.info(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}", "TrainingWorker")

        except Exception as e:
            self.error_occurred.emit(str(e))
            self.logger.error(f"Training failed: {e}", "TrainingWorker")
            self.logger.error(traceback.format_exc(), "TrainingWorker")
        finally:
            self.is_running = False
            self.logger.info("Cleaning up training resources...", "TrainingWorker")
            self._cleanup_resources()

    def stop_training(self):
        self.should_stop = True
        self.logger.info("Stopping training...", "TrainingWorker")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.stop_timeout_timer = QTimer()
        self.stop_timeout_timer.timeout.connect(self.force_stop)
        self.stop_timeout_timer.start(10000)

    def force_stop(self):
        self.logger.warning("Force stopping training thread", "TrainingWorker")
        self.terminate()
        self.wait()

    def _cleanup_resources(self):
        try:
            for attr in ('optimizer', 'model', 'scheduler', 'train_loader', 'val_loader'):
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    delattr(self, attr)
                    setattr(self, attr, None)
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            self.logger.info("Resource cleanup complete", "TrainingWorker")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}", "TrainingWorker")

    def _map_backbone_code(self, backbone: str) -> str:
        try:
            if not backbone:
                return "BK"
            b = backbone.lower()
            if 'resnet50-512' in b or ('resnet' in b and '512' in b):
                return "R512"
            if 'resnet50-64' in b or ('resnet' in b and '64' in b):
                return "R64"
            if b.startswith('vit'):
                import re
                m = re.search(r'(\d{3,4})', b)
                return f"V{m.group(1)}" if m else "VIT"
            if 'swin' in b:
                import re
                m = re.search(r'(\d{3,4})', b)
                return f"S{m.group(1)}" if m else "SWIN"
            return backbone[:4].upper()
        except Exception:
            return "BK"

    def _map_lambda_code(self) -> str:
        try:
            weights = [
                float(self.config.get('lambda_in', 1.0)),
                float(self.config.get('lambda_cross', 1.0)),
                float(self.config.get('lambda_hybrid', 1.0)),
                float(self.config.get('lambda_single', 1.0)),
                float(self.config.get('lambda_causal', 1.0)),
            ]
            if all(abs(w - 1.0) < 1e-6 for w in weights):
                return "LAll1"
            return "LCustom"
        except Exception:
            return "LAll1"

    def _map_class_code(self, test_fold=None) -> str:
        try:
            if test_fold is None:
                test_fold = self._resolve_fold_idx()
            if isinstance(test_fold, str) and test_fold.strip().upper().startswith('F'):
                return test_fold.strip().upper()[:3]
            return f"F{int(test_fold)}"
        except Exception:
            return "F0"

    def _ensure_artifact_dirs(self):
        base_root = self.config.get('checkpoint_path', './checkpoints')
        try:
            bb_code = self._map_backbone_code(self.config.get('backbone'))
            lam_code = self._map_lambda_code()
            ts = time.strftime("%Y%m%d_%H%M")
            self._artifact_ts = ts
            task_dir = os.path.join(base_root, "Task_3CLS")
            run_dir = os.path.join(task_dir, f"{bb_code}_{lam_code}_{ts}")
            dirs = {
                'root': run_dir,
                'Checkpoints': os.path.join(run_dir, "Checkpoints"),
                'Logs_CSV': os.path.join(run_dir, "Logs_CSV"),
                'TSNE_Plots': os.path.join(run_dir, "TSNE_Plots"),
                'Heatmaps': os.path.join(run_dir, "Heatmaps"),
            }
            for d in dirs.values():
                os.makedirs(d, exist_ok=True)
            self._artifact_dirs = dirs
            self.logger.info(f"Artifact directory: {run_dir}", "TrainingWorker")
        except Exception as e:
            self._artifact_dirs = {
                'root': base_root,
                'Checkpoints': base_root,
                'Logs_CSV': base_root,
                'TSNE_Plots': base_root,
                'Heatmaps': base_root,
            }
            self.logger.warning(f"Artifact dir fallback to {base_root}: {e}", "TrainingWorker")

    def _export_tsne_plot(self, val_payload: dict, epoch_label: str = "last"):
        try:
            from sklearn.manifold import TSNE
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            eeg_feats = val_payload.get('eeg_features')
            audio_feats = val_payload.get('audio_features')
            eeg_labels = val_payload.get('eeg_labels', np.array([], dtype=np.int64))
            audio_labels = val_payload.get('audio_labels', np.array([], dtype=np.int64))

            if (
                eeg_feats is None
                or audio_feats is None
                or not hasattr(eeg_feats, 'numel')
                or eeg_feats.numel() == 0
                or audio_feats.numel() == 0
            ):
                self.logger.warning("t-SNE skipped: no paired features collected", "TrainingWorker")
                return

            Xe = eeg_feats.detach().cpu().numpy()
            Xa = audio_feats.detach().cpu().numpy()
            ye = np.array(eeg_labels, dtype=np.int64)
            ya = np.array(audio_labels, dtype=np.int64)
            stage_names = EstrusStageRegistry.get_class_names()

            rng = np.random.default_rng(42)

            def sample_by_class(X, y, max_per_cls=100):
                sel_idx = []
                for cls in np.unique(y):
                    idxs = np.where(y == cls)[0]
                    if len(idxs) > max_per_cls:
                        idxs = rng.choice(idxs, size=max_per_cls, replace=False)
                    sel_idx.extend(list(idxs))
                sel_idx = np.array(sel_idx, dtype=np.int64)
                return X[sel_idx], y[sel_idx]

            Xe_s, ye_s = sample_by_class(Xe, ye, 100)
            Xa_s, ya_s = sample_by_class(Xa, ya, 100)
            X_all = np.vstack([Xe_s, Xa_s])
            tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=min(30, max(5, len(X_all) // 4)))
            Z = tsne.fit_transform(X_all)
            Zmin = Z.min(axis=0, keepdims=True)
            Zmax = Z.max(axis=0, keepdims=True)
            Znorm = (Z - Zmin) / np.maximum(Zmax - Zmin, 1e-12)
            n_e = Xe_s.shape[0]
            Ze = Znorm[:n_e]
            Za = Znorm[n_e:]

            palette = np.array([
                (255, 107, 107),
                (107, 174, 255),
                (107, 255, 181),
            ]) / 255.0

            fold_code = self._map_class_code()
            plt.figure(figsize=(8, 6), dpi=120)
            for cls in range(NUM_CLASSES):
                eeg_mask = ye_s == cls
                audio_mask = ya_s == cls
                color = palette[cls % NUM_CLASSES]
                label = stage_names[cls] if cls < len(stage_names) else f"stage_{cls}"
                if audio_mask.any():
                    plt.scatter(
                        Za[audio_mask, 0], Za[audio_mask, 1],
                        s=60, marker='^', c=[color], alpha=0.6, edgecolors='none',
                        label=f"{label} (audio)",
                    )
                if eeg_mask.any():
                    plt.scatter(
                        Ze[eeg_mask, 0], Ze[eeg_mask, 1],
                        s=60, marker='o', c=[color], alpha=0.6, edgecolors='k', linewidths=0.3,
                        label=f"{label} (eeg)",
                    )
            plt.xticks([])
            plt.yticks([])
            plt.legend(loc='best', fontsize=8)
            plt.tight_layout()

            self._ensure_artifact_dirs()
            tsne_dir = getattr(self, '_artifact_dirs', {}).get(
                'TSNE_Plots', self.config.get('checkpoint_path', './checkpoints')
            )
            os.makedirs(tsne_dir, exist_ok=True)
            png_path = os.path.join(tsne_dir, f"{fold_code}_{epoch_label}_tsne.png")
            pdf_path = os.path.join(tsne_dir, f"{fold_code}_{epoch_label}_tsne.pdf")
            plt.savefig(png_path)
            plt.savefig(pdf_path)
            plt.close()
            self.logger.success(f"t-SNE exported: {png_path}", "TrainingWorker")
        except Exception as e:
            self.logger.warning(f"t-SNE export failed: {e}", "TrainingWorker")


class TrainingBackend:

    def __init__(self):
        self.logger = get_logger()
        self.worker = None
        self.is_training = False

    def start_training(self, config: Dict, callbacks: Dict[str, Callable] = None):
        if self.is_training:
            raise RuntimeError("Training is already in progress")
        self.worker = TrainingWorker(config)
        if callbacks:
            for signal_name in (
                'epoch_finished', 'training_finished', 'error_occurred',
                'progress_updated', 'log_message',
            ):
                if signal_name in callbacks:
                    getattr(self.worker, signal_name).connect(callbacks[signal_name])
        self.worker.start()
        self.is_training = True
        try:
            fold = config.get('test_fold', config.get('test_class'))
            self.logger.info(
                f"Training start | backbone={config.get('backbone')} | test_fold={fold} "
                f"| epochs={config.get('epochs', 50)} | batch_size={config.get('batch_size', 128)} "
                f"| lr={config.get('lr', 1e-4)} | weight_decay={config.get('weight_decay', 5e-3)} "
                f"| gamma={config.get('gamma', 1.0)} | tau={config.get('tau', 0.4)} "
                f"| patience={config.get('patience', 20)}",
                "TrainingBackend",
            )
        except Exception:
            pass

    def stop_training(self):
        if self.worker and self.is_training:
            self.logger.info("Stopping training...", "TrainingBackend")
            try:
                self.worker.stop_training()
                if self.worker.wait(20000):
                    self.logger.success("Training stopped cleanly", "TrainingBackend")
                else:
                    self.logger.warning("Stop timeout, forcing termination", "TrainingBackend")
                    try:
                        self.worker.terminate()
                        self.worker.wait(5000)
                    except Exception as e:
                        self.logger.error(f"Force terminate failed: {e}", "TrainingBackend")
            except Exception as e:
                self.logger.error(f"Stop error: {e}", "TrainingBackend")
                try:
                    self.worker.terminate()
                    self.worker.wait(3000)
                except Exception:
                    pass
            finally:
                self.is_training = False
                self.worker = None
                try:
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            with torch.cuda.device(i):
                                torch.cuda.empty_cache()
                except Exception as e:
                    self.logger.warning(f"GPU cleanup failed: {e}", "TrainingBackend")
        else:
            self.logger.info("No active training session", "TrainingBackend")

    def is_training_active(self) -> bool:
        return self.is_training and self.worker and self.worker.is_running


if __name__ == "__main__":
    test_config = {
        'backbone': 'ResNet50-512',
        'test_fold': 0,
        'batch_size': 128,
        'lr': 1e-4,
        'weight_decay': 5e-3,
        'epochs': 50,
        'gamma': 1.0,
        'tau': 0.4,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'grad_clip': 1.0,
        'patience': 20,
        'auto_save': True,
        'save_interval': 10,
        'lambda_in': 1.0,
        'lambda_cross': 1.0,
        'lambda_hybrid': 1.0,
        'lambda_single': 1.0,
        'lambda_causal': 1.0,
        'lambda_cls': 1.0,
    }
    backend = TrainingBackend()
    print("Training backend test completed!")
