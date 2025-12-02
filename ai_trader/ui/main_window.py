from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget
from PyQt6.QtCore import Qt
from ai_trader.ui.wizard_view import WizardView
from ai_trader.ui.advanced_view import AdvancedView

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Algorithmic Trader Pro")
        self.setGeometry(100, 100, 1000, 700)

        # Main Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Sidebar
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet("background-color: #2c3e50; color: white;")
        sidebar_layout = QVBoxLayout(self.sidebar)

        title_label = QLabel("AI TRADER")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 20px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(title_label)

        self.btn_wizard = QPushButton("Quick Start (Wizard)")
        self.btn_advanced = QPushButton("Advanced Mode")

        # Styling buttons
        for btn in [self.btn_wizard, self.btn_advanced]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #34495e;
                    border: none;
                    padding: 15px;
                    text-align: left;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #1abc9c;
                }
                QPushButton:checked {
                    background-color: #16a085;
                }
            """)
            btn.setCheckable(True)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()

        # Content Area (Stacked Widget)
        self.content_area = QStackedWidget()

        # Initialize Views (Placeholders for now, we will create them in next steps)
        # Note: Importing them inside the class or file if they existed.
        # Since I'm creating them in subsequent steps, I'll instantiate them here assuming they will exist
        # But to avoid ImportErrors right now, I will use simple Placeholders if imports fail,
        # but the plan says I will create them next.

        self.wizard_view = WizardView()
        self.advanced_view = AdvancedView()

        self.content_area.addWidget(self.wizard_view)
        self.content_area.addWidget(self.advanced_view)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.content_area)

        # Connect Signals
        self.btn_wizard.clicked.connect(lambda: self.switch_view(0))
        self.btn_advanced.clicked.connect(lambda: self.switch_view(1))

        # Default View
        self.btn_wizard.setChecked(True)
        self.switch_view(0)

    def switch_view(self, index):
        self.content_area.setCurrentIndex(index)
        if index == 0:
            self.btn_wizard.setChecked(True)
            self.btn_advanced.setChecked(False)
        else:
            self.btn_wizard.setChecked(False)
            self.btn_advanced.setChecked(True)
