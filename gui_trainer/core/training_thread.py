from typing import Callable, Dict, Optional

from PyQt5.QtCore import pyqtSignal

from .training_backend import TrainingWorker


class TrainingThread(TrainingWorker):
    log_signal = pyqtSignal(str)

    def __init__(self, config: Dict, parent=None):
        super().__init__(config, parent=parent)
        self.log_message.connect(self._forward_log)

    def _forward_log(self, message: str):
        self.log_signal.emit(message)


class TrainingThreadManager:
    def __init__(self):
        self.worker: Optional[TrainingThread] = None

    def start(self, config: Dict, callbacks: Optional[Dict[str, Callable]] = None):
        self.worker = TrainingThread(config)
        if callbacks:
            for name, handler in callbacks.items():
                if hasattr(self.worker, name):
                    signal = getattr(self.worker, name)
                    signal.connect(handler)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.stop_training()
            self.worker.wait(20000)

    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()
