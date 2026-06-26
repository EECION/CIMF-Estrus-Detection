from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton, QComboBox, QLabel
from PyQt5.QtGui import QTextCharFormat, QColor, QFont
from PyQt5.QtCore import Qt
from datetime import datetime


class LogMonitorWindow(QWidget):
    LEVEL_COLORS = {
        "DEBUG": "#888888",
        "INFO": "#333333",
        "WARNING": "#FF9800",
        "ERROR": "#F44336",
        "SUCCESS": "#4CAF50",
    }

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Filter:"))
        self.level_filter = QComboBox()
        self.level_filter.addItems(["ALL", "INFO", "WARNING", "ERROR", "SUCCESS"])
        self.level_filter.currentTextChanged.connect(self.apply_filter)
        toolbar.addWidget(self.level_filter)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_logs)
        toolbar.addWidget(self.clear_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)
        self.setLayout(layout)
        self._all_entries = []

    def add_log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = {"time": timestamp, "level": level.upper(), "message": message}
        self._all_entries.append(entry)
        self._append_entry(entry)

    def _append_entry(self, entry):
        color = self.LEVEL_COLORS.get(entry["level"], "#333333")
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        line = f"[{entry['time']}] [{entry['level']}] {entry['message']}\n"
        cursor.insertText(line, fmt)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    def clear_logs(self):
        self._all_entries = []
        self.log_text.clear()

    def apply_filter(self, level):
        self.log_text.clear()
        for entry in self._all_entries:
            if level == "ALL" or entry["level"] == level:
                self._append_entry(entry)
