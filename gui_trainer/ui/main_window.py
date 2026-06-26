import os
import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QPushButton, QFileDialog,
    QTextEdit, QMessageBox, QSplitter,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

from ..utils.config_utils import ConfigManager
from ..core.checkpoint_manager import CheckpointManager


class MainTrainingWindow(QWidget):
    training_started = pyqtSignal(dict)
    training_stopped = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.checkpoint_manager = CheckpointManager()
        self.is_training = False
        self.init_ui()
        self.load_default_config()
        self.system_info_timer = QTimer()
        self.system_info_timer.timeout.connect(self.update_system_info)
        self.system_info_timer.start(2000)

    def init_ui(self):
        layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.create_config_panel())
        splitter.addWidget(self.create_info_panel())
        splitter.setSizes([800, 600])
        layout.addWidget(splitter)
        self.setLayout(layout)

    def create_config_panel(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.create_basic_params_group())
        layout.addWidget(self.create_module_params_group())
        layout.addWidget(self.create_optimizer_params_group())
        layout.addWidget(self.create_checkpoint_group())
        layout.addWidget(self.create_control_group())
        widget.setLayout(layout)
        return widget

    def create_basic_params_group(self):
        group = QGroupBox("Basic Training Parameters")
        layout = QGridLayout()
        layout.addWidget(QLabel("Data Path:"), 0, 0)
        data_path_layout = QHBoxLayout()
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setPlaceholderText("Select data folder path...")
        self.data_path_browse_btn = QPushButton("Browse")
        self.data_path_browse_btn.clicked.connect(self.browse_data_path)
        data_path_layout.addWidget(self.data_path_edit)
        data_path_layout.addWidget(self.data_path_browse_btn)
        data_path_widget = QWidget()
        data_path_widget.setLayout(data_path_layout)
        layout.addWidget(data_path_widget, 0, 1, 1, 3)
        layout.addWidget(QLabel("Quick Test:"), 1, 0)
        self.quick_test_check = QCheckBox("Quick Test Mode")
        self.quick_test_check.stateChanged.connect(self.on_quick_test_changed)
        layout.addWidget(self.quick_test_check, 1, 1)
        layout.addWidget(QLabel("Backbone:"), 2, 0)
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(["ResNet50-64", "ResNet50-512", "ViT-B-768", "Swin-T-768"])
        self.backbone_combo.setCurrentText("ResNet50-512")
        layout.addWidget(self.backbone_combo, 2, 1)
        layout.addWidget(QLabel("Test Fold:"), 2, 2)
        self.test_fold_combo = QComboBox()
        self.test_fold_combo.addItems([f"F{i}" for i in range(5)])
        layout.addWidget(self.test_fold_combo, 2, 3)
        layout.addWidget(QLabel("Batch Size:"), 3, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(16, 256)
        self.batch_size_spin.setValue(64)
        layout.addWidget(self.batch_size_spin, 3, 1)
        layout.addWidget(QLabel("Learning Rate:"), 3, 2)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1e-2)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setValue(1e-4)
        self.lr_spin.setSingleStep(1e-5)
        layout.addWidget(self.lr_spin, 3, 3)
        layout.addWidget(QLabel("Weight Decay:"), 4, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(1e-6, 1e-2)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setValue(5e-3)
        self.weight_decay_spin.setSingleStep(1e-5)
        layout.addWidget(self.weight_decay_spin, 4, 1)
        layout.addWidget(QLabel("Epochs:"), 4, 2)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)
        layout.addWidget(self.epochs_spin, 4, 3)
        layout.addWidget(QLabel("Pretrained:"), 5, 0)
        self.pretrained_check = QCheckBox("Use ImageNet Pretrained")
        self.pretrained_check.setChecked(True)
        layout.addWidget(self.pretrained_check, 5, 1)
        layout.addWidget(QLabel("Weight Path:"), 5, 2)
        weights_layout = QHBoxLayout()
        self.weights_path_edit = QLineEdit()
        self.weights_path_edit.setPlaceholderText("Select pretrained weight file (.pth)")
        self.weights_browse_btn = QPushButton("Browse")
        self.weights_browse_btn.clicked.connect(self.browse_weights_path)
        weights_layout.addWidget(self.weights_path_edit)
        weights_layout.addWidget(self.weights_browse_btn)
        weights_widget = QWidget()
        weights_widget.setLayout(weights_layout)
        layout.addWidget(weights_widget, 5, 3)
        layout.addWidget(QLabel("Gamma:"), 6, 0)
        self.gamma_spin = QSpinBox()
        self.gamma_spin.setRange(1, 1024)
        self.gamma_spin.setValue(1)
        layout.addWidget(self.gamma_spin, 6, 1)
        layout.addWidget(QLabel("Tau (Fourier Mix):"), 6, 2)
        self.tau_spin = QDoubleSpinBox()
        self.tau_spin.setRange(0.1, 0.5)
        self.tau_spin.setDecimals(2)
        self.tau_spin.setValue(0.4)
        self.tau_spin.setSingleStep(0.05)
        layout.addWidget(self.tau_spin, 6, 3)
        group.setLayout(layout)
        return group

    def create_module_params_group(self):
        group = QGroupBox("Module Switches and Weight Parameters")
        layout = QGridLayout()
        lambda_params = [
            ("Lambda In-Modal:", "lambda_in"),
            ("Lambda Cross-Modal:", "lambda_cross"),
            ("Lambda Hybrid:", "lambda_hybrid"),
            ("Lambda Single:", "lambda_single"),
            ("Lambda CIRE:", "lambda_causal"),
        ]
        self.lambda_spins = {}
        for i, (label, param_name) in enumerate(lambda_params):
            layout.addWidget(QLabel(label), i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 2.0)
            spin.setDecimals(2)
            spin.setValue(1.0)
            spin.setSingleStep(0.1)
            layout.addWidget(spin, i, 1)
            self.lambda_spins[param_name] = spin
            if param_name == "lambda_causal":
                desc_label = QLabel("(Set to 0 to disable CIRE module)")
                desc_label.setStyleSheet("color: #666; font-size: 8pt;")
                layout.addWidget(desc_label, i, 2)
        group.setLayout(layout)
        return group

    def create_optimizer_params_group(self):
        group = QGroupBox("Optimizer and Scheduler Parameters")
        layout = QGridLayout()
        layout.addWidget(QLabel("Optimizer:"), 0, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "AdamW", "SGD"])
        layout.addWidget(self.optimizer_combo, 0, 1)
        layout.addWidget(QLabel("LR Scheduler:"), 0, 2)
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["Cosine", "Step", "None"])
        layout.addWidget(self.scheduler_combo, 0, 3)
        layout.addWidget(QLabel("Grad Clip:"), 1, 0)
        self.grad_clip_spin = QDoubleSpinBox()
        self.grad_clip_spin.setRange(0.0, 10.0)
        self.grad_clip_spin.setDecimals(2)
        self.grad_clip_spin.setValue(1.0)
        layout.addWidget(self.grad_clip_spin, 1, 1)
        layout.addWidget(QLabel("Patience:"), 1, 2)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(5, 50)
        self.patience_spin.setValue(20)
        layout.addWidget(self.patience_spin, 1, 3)
        layout.addWidget(QLabel("Val Timeout (sec):"), 2, 0)
        self.val_timeout_spin = QSpinBox()
        self.val_timeout_spin.setRange(10, 600)
        self.val_timeout_spin.setValue(60)
        layout.addWidget(self.val_timeout_spin, 2, 1)
        layout.addWidget(QLabel("Val Max Stall:"), 2, 2)
        self.val_max_stall_spin = QSpinBox()
        self.val_max_stall_spin.setRange(1, 50)
        self.val_max_stall_spin.setValue(2)
        layout.addWidget(self.val_max_stall_spin, 2, 3)
        group.setLayout(layout)
        return group

    def create_checkpoint_group(self):
        group = QGroupBox("Checkpoint Management")
        layout = QVBoxLayout()
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Checkpoint Path:"))
        self.checkpoint_path_edit = QLineEdit()
        self.checkpoint_path_edit.setPlaceholderText("Select folder containing .pth files")
        path_layout.addWidget(self.checkpoint_path_edit)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_checkpoint_path)
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)
        list_layout = QHBoxLayout()
        list_layout.addWidget(QLabel("Available Checkpoints:"))
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.currentTextChanged.connect(self.on_checkpoint_selected)
        list_layout.addWidget(self.checkpoint_combo)
        self.load_checkpoint_btn = QPushButton("Load")
        self.load_checkpoint_btn.clicked.connect(self.load_checkpoint)
        list_layout.addWidget(self.load_checkpoint_btn)
        layout.addLayout(list_layout)
        self.checkpoint_info_text = QTextEdit()
        self.checkpoint_info_text.setMaximumHeight(80)
        layout.addWidget(self.checkpoint_info_text)
        save_layout = QHBoxLayout()
        self.auto_save_check = QCheckBox("Auto Save Best Model")
        self.auto_save_check.setChecked(True)
        save_layout.addWidget(self.auto_save_check)
        save_layout.addWidget(QLabel("Save Every N Epochs:"))
        self.save_interval_spin = QSpinBox()
        self.save_interval_spin.setRange(1, 20)
        self.save_interval_spin.setValue(5)
        save_layout.addWidget(self.save_interval_spin)
        layout.addLayout(save_layout)
        group.setLayout(layout)
        return group

    def create_control_group(self):
        group = QGroupBox("Training Control")
        layout = QHBoxLayout()
        self.train_control_btn = QPushButton("Start Training")
        self.train_control_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 10px 20px; "
            "font-size: 12pt; font-weight: bold; border-radius: 5px; min-width: 120px; }"
        )
        self.train_control_btn.clicked.connect(self.toggle_training)
        layout.addWidget(self.train_control_btn)
        layout.addStretch()
        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self.save_config)
        layout.addWidget(self.save_config_btn)
        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.clicked.connect(self.load_config)
        layout.addWidget(self.load_config_btn)
        self.reset_config_btn = QPushButton("Reset Config")
        self.reset_config_btn.clicked.connect(self.reset_config)
        layout.addWidget(self.reset_config_btn)
        group.setLayout(layout)
        return group

    def create_info_panel(self):
        widget = QWidget()
        layout = QVBoxLayout()
        system_group = QGroupBox("System Information")
        system_layout = QHBoxLayout()
        system_info_group = QGroupBox("System Information")
        system_info_layout = QVBoxLayout()
        self.system_info_text = QTextEdit()
        self.system_info_text.setReadOnly(True)
        system_info_layout.addWidget(self.system_info_text)
        system_info_group.setLayout(system_info_layout)
        resource_monitor_group = QGroupBox("Real-time Resource Monitoring")
        resource_monitor_layout = QVBoxLayout()
        self.resource_monitor_text = QTextEdit()
        self.resource_monitor_text.setReadOnly(True)
        resource_monitor_layout.addWidget(self.resource_monitor_text)
        resource_monitor_group.setLayout(resource_monitor_layout)
        system_layout.addWidget(system_info_group)
        system_layout.addWidget(resource_monitor_group)
        system_group.setLayout(system_layout)
        layout.addWidget(system_group)
        preview_group = QGroupBox("Training Preview")
        preview_layout = QVBoxLayout()
        self.preview_text = QTextEdit()
        preview_layout.addWidget(self.preview_text)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        widget.setLayout(layout)
        return widget

    def update_system_info(self):
        try:
            import torch
            import platform
            import psutil
            system_info = [
                f"Operating System: {platform.system()} {platform.release()}",
                f"Python Version: {platform.python_version()}",
                f"PyTorch Version: {torch.__version__}",
                f"CPU Cores: {psutil.cpu_count()}",
                f"System Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB",
            ]
            if torch.cuda.is_available():
                system_info.append(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                system_info.append("GPU: No CUDA device detected")
            resource_info = [
                f"CPU Usage: {psutil.cpu_percent(interval=0.1):.1f}%",
                f"Memory Usage: {psutil.virtual_memory().percent:.1f}%",
            ]
            self.system_info_text.setText("\n".join(system_info))
            self.resource_monitor_text.setText("\n".join(resource_info))
        except Exception as e:
            self.system_info_text.setText(f"Failed to get system information: {e}")

    def browse_checkpoint_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Checkpoint Folder")
        if path:
            self.checkpoint_path_edit.setText(path)
            self.update_checkpoint_list(path)

    def browse_data_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if path:
            self.data_path_edit.setText(path)
            self.validate_data_path(path)

    def validate_data_path(self, path):
        if not os.path.exists(path):
            QMessageBox.warning(self, "Path Error", "Selected path does not exist")
            return False
        eeg_path = os.path.join(path, "eeg")
        audio_path = os.path.join(path, "audio")
        if not os.path.isdir(eeg_path) or not os.path.isdir(audio_path):
            QMessageBox.warning(
                self,
                "Data Structure Error",
                "Expected eeg/ and audio/ directories with pre_estrus, estrus, and post_estrus subfolders",
            )
            return False
        from ..utils.estrus_stages import EstrusStageRegistry
        stage_names = EstrusStageRegistry.get_class_names()
        info_msg = "Dataset loaded successfully.\n\n"
        info_msg += f"Number of estrus stages: {len(stage_names)}\n"
        info_msg += f"Stage list: {', '.join(stage_names)}\n\n"
        for stage_name in stage_names:
            eeg_count = 0
            audio_count = 0
            eeg_stage_dir = os.path.join(eeg_path, stage_name)
            audio_stage_dir = os.path.join(audio_path, stage_name)
            if os.path.isdir(eeg_stage_dir):
                eeg_count = len([
                    f for f in os.listdir(eeg_stage_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
                ])
            if os.path.isdir(audio_stage_dir):
                audio_count = len([
                    f for f in os.listdir(audio_stage_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
                ])
            info_msg += f"{stage_name}: EEG({eeg_count}), Audio({audio_count})\n"
        QMessageBox.information(self, "Dataset Information", info_msg)
        return True

    def on_quick_test_changed(self, state):
        if state == 2:
            self.epochs_spin.setValue(5)
        else:
            self.epochs_spin.setValue(50)

    def browse_weights_path(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Pretrained Weight File", "", "PyTorch Files (*.pth);;All Files (*)"
        )
        if filename:
            self.weights_path_edit.setText(filename)

    def update_checkpoint_list(self, path):
        self.checkpoint_combo.clear()
        if os.path.exists(path):
            checkpoints = [f for f in os.listdir(path) if f.endswith((".pth", ".ckpt"))]
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)
            self.checkpoint_combo.addItems(checkpoints)

    def on_checkpoint_selected(self, filename):
        if filename and self.checkpoint_path_edit.text():
            filepath = os.path.join(self.checkpoint_path_edit.text(), filename)
            info = self.checkpoint_manager.get_checkpoint_info(filepath)
            self.checkpoint_info_text.setText(info)

    def load_checkpoint(self):
        if not self.checkpoint_combo.currentText():
            QMessageBox.warning(self, "Warning", "Please select a checkpoint file first")
            return
        filepath = os.path.join(self.checkpoint_path_edit.text(), self.checkpoint_combo.currentText())
        try:
            config = self.checkpoint_manager.load_checkpoint_config(filepath)
            self.apply_config(config)
            QMessageBox.information(self, "Success", "Checkpoint configuration loaded")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load checkpoint: {str(e)}")

    def get_current_config(self):
        config = {
            "data_path": self.data_path_edit.text(),
            "quick_test": self.quick_test_check.isChecked(),
            "backbone": self.backbone_combo.currentText(),
            "test_fold": int(self.test_fold_combo.currentText().replace("F", "")),
            "batch_size": self.batch_size_spin.value(),
            "lr": self.lr_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "epochs": self.epochs_spin.value(),
            "gamma": self.gamma_spin.value(),
            "tau": self.tau_spin.value(),
            "optimizer": self.optimizer_combo.currentText(),
            "scheduler": self.scheduler_combo.currentText(),
            "grad_clip": self.grad_clip_spin.value(),
            "patience": self.patience_spin.value(),
            "val_timeout_sec": self.val_timeout_spin.value(),
            "val_max_stall": self.val_max_stall_spin.value(),
            "auto_save": self.auto_save_check.isChecked(),
            "save_interval": self.save_interval_spin.value(),
            "checkpoint_path": self.checkpoint_path_edit.text(),
            "checkpoint_file": self.checkpoint_combo.currentText(),
            "use_pretrained": self.pretrained_check.isChecked(),
            "weights_path": self.weights_path_edit.text(),
        }
        for param_name, spin in self.lambda_spins.items():
            config[param_name] = spin.value()
        return config

    def apply_config(self, config):
        if isinstance(config, dict):
            if "data_path" in config:
                self.data_path_edit.setText(config["data_path"])
            if "quick_test" in config:
                self.quick_test_check.setChecked(config["quick_test"])
            if "backbone" in config:
                index = self.backbone_combo.findText(config["backbone"])
                if index >= 0:
                    self.backbone_combo.setCurrentIndex(index)
            if "test_fold" in config:
                fold_text = f"F{int(config['test_fold'])}"
                index = self.test_fold_combo.findText(fold_text)
                if index >= 0:
                    self.test_fold_combo.setCurrentIndex(index)
            if "batch_size" in config:
                self.batch_size_spin.setValue(config["batch_size"])
            if "lr" in config:
                self.lr_spin.setValue(config["lr"])
            if "weight_decay" in config:
                self.weight_decay_spin.setValue(config["weight_decay"])
            if "epochs" in config:
                self.epochs_spin.setValue(config["epochs"])
            if "gamma" in config:
                self.gamma_spin.setValue(int(config["gamma"]))
            if "tau" in config:
                self.tau_spin.setValue(config["tau"])
            if "optimizer" in config:
                index = self.optimizer_combo.findText(config["optimizer"])
                if index >= 0:
                    self.optimizer_combo.setCurrentIndex(index)
            if "scheduler" in config:
                index = self.scheduler_combo.findText(config["scheduler"])
                if index >= 0:
                    self.scheduler_combo.setCurrentIndex(index)
            if "grad_clip" in config:
                self.grad_clip_spin.setValue(config["grad_clip"])
            if "patience" in config:
                self.patience_spin.setValue(config["patience"])
            if "val_timeout_sec" in config:
                self.val_timeout_spin.setValue(config["val_timeout_sec"])
            if "val_max_stall" in config:
                self.val_max_stall_spin.setValue(config["val_max_stall"])
            if "save_interval" in config:
                self.save_interval_spin.setValue(config["save_interval"])
            if "checkpoint_path" in config:
                self.checkpoint_path_edit.setText(config["checkpoint_path"])
            if "use_pretrained" in config:
                self.pretrained_check.setChecked(config["use_pretrained"])
            if "weights_path" in config:
                self.weights_path_edit.setText(config["weights_path"])
            for param_name, spin in self.lambda_spins.items():
                if param_name in config:
                    spin.setValue(config[param_name])
            return

    def load_default_config(self):
        default_config = self.config_manager.get_default_config()
        self.apply_config(default_config)

    def save_config(self):
        config = self.get_current_config()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "config.json", "JSON Files (*.json)")
        if filename:
            self.config_manager.save_config(config, filename)
            QMessageBox.information(self, "Success", "Configuration saved")

    def load_config(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if filename:
            config = self.config_manager.load_config(filename)
            self.apply_config(config)
            QMessageBox.information(self, "Success", "Configuration loaded")

    def reset_config(self):
        reply = QMessageBox.question(
            self, "Confirm", "Reset all configurations to default values?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.load_default_config()

    def toggle_training(self):
        if not self.is_training:
            self.start_training()
        else:
            self.stop_training()

    def start_training(self):
        try:
            config = self.get_current_config()
            if not self.validate_config(config):
                return
            self.update_training_preview(config)
            self.is_training = True
            self.update_train_button_state()
            self.training_started.emit(config)
        except Exception as e:
            self.is_training = False
            self.update_train_button_state()
            QMessageBox.critical(self, "Error", f"Failed to start training: {str(e)}")

    def stop_training(self):
        reply = QMessageBox.question(
            self, "Confirm Stop", "Stop current training?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.is_training = False
            self.update_train_button_state()
            self.training_stopped.emit()

    def update_train_button_state(self):
        if self.is_training:
            self.train_control_btn.setText("Stop Training")
            self.train_control_btn.setStyleSheet(
                "QPushButton { background-color: #f44336; color: white; padding: 10px 20px; "
                "font-size: 12pt; font-weight: bold; border-radius: 5px; min-width: 120px; }"
            )
        else:
            self.train_control_btn.setText("Start Training")
            self.train_control_btn.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; padding: 10px 20px; "
                "font-size: 12pt; font-weight: bold; border-radius: 5px; min-width: 120px; }"
            )

    def validate_config(self, config):
        if not config.get("data_path") or not os.path.exists(config["data_path"]):
            QMessageBox.warning(self, "Configuration Error", "Please select a valid data path")
            return False
        eeg_path = os.path.join(config["data_path"], "eeg")
        audio_path = os.path.join(config["data_path"], "audio")
        if not os.path.isdir(eeg_path) or not os.path.isdir(audio_path):
            QMessageBox.warning(
                self, "Configuration Error",
                "Data path must contain eeg/ and audio/ subdirectories",
            )
            return False
        if config["batch_size"] <= 0 or config["lr"] <= 0 or config["epochs"] <= 0:
            QMessageBox.warning(self, "Configuration Error", "Invalid training parameters")
            return False
        if config.get("lambda_causal") == 0:
            reply = QMessageBox.question(
                self, "Confirm", "Lambda CIRE=0 will disable CIRE module, continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return False
        return True

    def update_training_preview(self, config):
        preview_lines = [
            f"Training Task: CIMF Estrus Stage Classification (Test fold: F{config['test_fold']})",
            f"Data Path: {config.get('data_path', 'Not set')}",
            f"Backbone: {config['backbone']}",
            f"Batch Size: {config['batch_size']}",
            f"Learning Rate: {config['lr']:.6f}",
            f"Epochs: {config['epochs']}",
            f"Gamma: {config['gamma']}",
            f"Lambda CIRE: {config['lambda_causal']}",
        ]
        self.preview_text.setText("\n".join(preview_lines))

    def on_training_finished(self):
        self.is_training = False
        self.update_train_button_state()
