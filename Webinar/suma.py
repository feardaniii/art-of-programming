import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QLineEdit, QPushButton, QLabel, QComboBox, QDialog,
                            QTextEdit, QFrame, QGridLayout, QScrollArea)
from PyQt5.QtGui import QFont, QIcon, QPainter, QPen
from PyQt5.QtCore import Qt

class PaperSolutionDialog(QDialog):
    def __init__(self, num1=7, num2=5, operation='+', parent=None):
        super().__init__(parent)
        self.num1 = num1
        self.num2 = num2
        self.operation = operation
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Cum rezolvă un elev de clasa a V-a pe hârtie')
        self.setGeometry(200, 200, 500, 400)
        self.setStyleSheet("""
            QDialog {
                background-color: #fff8e1;
                border: 2px solid #8bc34a;
            }
            QLabel {
                font-family: 'Comic Sans MS', cursive;
                color: #2e7d32;
            }
            .paper-line {
                font-size: 16px;
                font-family: 'Courier New', monospace;
                color: #1976d2;
                margin: 5px;
            }
            .result-line {
                font-size: 18px;
                font-weight: bold;
                color: #d32f2f;
            }
            QPushButton {
                background-color: #4caf50;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        layout = QVBoxLayout()

        # Title
        title = QLabel('📝 Exemplu: Cum calculez pe hârtie')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Paper simulation
        paper_frame = QFrame()
        paper_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
            }
        """)
        paper_layout = QVBoxLayout()

        # Step by step solution
        steps = [
            f"🔢 Am două numere: x = {self.num1} și y = {self.num2}",
            f"📐 Vreau să calculez: x {self.operation} y",
            f"✍️ Înlocuiesc valorile: {self.num1} {self.operation} {self.num2}",
            f"🧮 Calculez pas cu pas:"
        ]

        for step in steps:
            step_label = QLabel(step)
            step_label.setStyleSheet("font-size: 14px; margin: 3px;")
            paper_layout.addWidget(step_label)

        # Visual calculation
        if self.operation == '+':
            result = self.num1 + self.num2
            calc_visual = f"""
        {self.num1:>5}
      + {self.num2:>3}
      -----
      = {result:>3}
            """
        elif self.operation == '-':
            result = self.num1 - self.num2
            calc_visual = f"""
        {self.num1:>5}
      - {self.num2:>3}
      -----
      = {result:>3}
            """
        elif self.operation == '*':
            result = self.num1 * self.num2
            calc_visual = f"""
        {self.num1:>5}
      × {self.num2:>3}
      -----
      = {result:>3}
            """
        else:  # division
            result = self.num1 / self.num2
            calc_visual = f"""
        {self.num1:>5}
      ÷ {self.num2:>3}
      -----
      = {result:>5.1f}
            """

        calc_label = QLabel(calc_visual)
        calc_label.setStyleSheet("""
            font-family: 'Courier New', monospace;
            font-size: 18px;
            background-color: #f5f5f5;
            border: 1px dashed #999;
            padding: 10px;
            color: #1976d2;
        """)
        calc_label.setAlignment(Qt.AlignCenter)
        paper_layout.addWidget(calc_label)

        # Final answer
        answer_label = QLabel(f"✅ Răspunsul final: {result}")
        answer_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #d32f2f;
            background-color: #ffeb3b;
            padding: 8px;
            border-radius: 4px;
            margin-top: 10px;
        """)
        answer_label.setAlignment(Qt.AlignCenter)
        paper_layout.addWidget(answer_label)

        # Explanation
        explanation = QLabel("💡 Computerul face exact același lucru, doar mult mai rapid!")
        explanation.setStyleSheet("font-size: 14px; font-style: italic; margin-top: 10px;")
        explanation.setAlignment(Qt.AlignCenter)
        paper_layout.addWidget(explanation)

        paper_frame.setLayout(paper_layout)
        layout.addWidget(paper_frame)

        # Close button
        close_btn = QPushButton('Înțeles! 👍')
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)

class Calculator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('🧮 Calculator pentru Începători - Învățăm să "vorbim" cu computerul!')
        self.setWindowIcon(QIcon('calculator.png'))  # You might need to create an icon file
        self.setGeometry(100, 100, 600, 450)
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #e3f2fd, stop: 1 #f0f8ff);
                font-family: 'Arial', sans-serif;
            }
            QLabel {
                font-size: 14px;
                color: #1976d2;
            }
            QLineEdit, QComboBox {
                padding: 12px;
                border: 2px solid #4fc3f7;
                border-radius: 8px;
                background-color: white;
                font-size: 16px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #2196f3;
                background-color: #f8f9fa;
            }
            QPushButton {
                background-color: #4caf50;
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
                transform: scale(1.05);
            }
            #resultLabel {
                font-size: 20px;
                font-weight: bold;
                color: #d32f2f;
                background-color: #fff3e0;
                padding: 15px;
                border-radius: 8px;
                border: 2px solid #ff9800;
            }
            #hintLabel {
                font-size: 13px;
                color: #666;
                font-style: italic;
                background-color: #e8f5e8;
                padding: 8px;
                border-radius: 4px;
                border-left: 4px solid #4caf50;
            }
            #paperButton {
                background-color: #9c27b0;
                color: white;
                font-size: 14px;
            }
            #paperButton:hover {
                background-color: #7b1fa2;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Welcome message
        welcome_label = QLabel('👋 Bun venit! Să învățăm cum să "vorbim" cu computerul!')
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2e7d32; margin-bottom: 10px;")
        main_layout.addWidget(welcome_label)

        # Instructions
        instruction_label = QLabel('📝 Instrucțiuni: Introdu două numere și alege operația matematică')
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setObjectName('hintLabel')
        main_layout.addWidget(instruction_label)

        # Input section with labels
        input_section = QVBoxLayout()

        # Number inputs with hints
        num_layout = QHBoxLayout()

        # First number
        num1_label = QLabel('Primul număr (x):')
        num1_label.setStyleSheet("font-weight: bold;")
        self.num1_input = QLineEdit(self)
        self.num1_input.setPlaceholderText("ex: 7")
        self.num1_input.setAlignment(Qt.AlignCenter)

        # Operation
        op_label = QLabel('Operația:')
        op_label.setStyleSheet("font-weight: bold;")
        self.op_combo = QComboBox(self)
        self.op_combo.addItems(['+', '-', '×', '÷'])

        # Second number
        num2_label = QLabel('Al doilea număr (y):')
        num2_label.setStyleSheet("font-weight: bold;")
        self.num2_input = QLineEdit(self)
        self.num2_input.setPlaceholderText("ex: 5")
        self.num2_input.setAlignment(Qt.AlignCenter)

        # Add to layout
        num_layout.addWidget(num1_label)
        num_layout.addWidget(self.num1_input)
        num_layout.addWidget(op_label)
        num_layout.addWidget(self.op_combo)
        num_layout.addWidget(num2_label)
        num_layout.addWidget(self.num2_input)

        input_section.addLayout(num_layout)

        # Hint about computer processing
        hint_label = QLabel('💡 Indiciu: Computerul va procesa exact ca la matematică!')
        hint_label.setAlignment(Qt.AlignCenter)
        hint_label.setObjectName('hintLabel')
        input_section.addWidget(hint_label)

        main_layout.addLayout(input_section)

        # Result display
        self.result_label = QLabel('🤖 Computerul va afișa rezultatul aici...', self)
        self.result_label.setObjectName('resultLabel')
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        # Buttons layout
        button_layout = QHBoxLayout()

        # Calculate button
        self.calc_button = QPushButton('🧮 Calculează!', self)
        self.calc_button.clicked.connect(self.calculate)

        # Paper solution button
        self.paper_button = QPushButton('📝 Vezi cum se face pe hârtie', self)
        self.paper_button.setObjectName('paperButton')
        self.paper_button.clicked.connect(self.show_paper_solution)

        button_layout.addWidget(self.calc_button)
        button_layout.addWidget(self.paper_button)

        main_layout.addLayout(button_layout)

        # Educational footer
        footer_label = QLabel('🎓 Programarea = a da instrucțiuni clare computerului!')
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("font-size: 12px; color: #666; margin-top: 10px; font-style: italic;")
        main_layout.addWidget(footer_label)

        self.setLayout(main_layout)

    def calculate(self):
        try:
            num1 = float(self.num1_input.text())
            num2 = float(self.num2_input.text())
            op = self.op_combo.currentText()





            # Toata logica se intampla aici, restul este UI/UX
            # Convert display symbols to calculation symbols
            # Manual inference
            if op == '+':
                result = num1 + num2
                operation_text = "adunarea"
            elif op == '-':
                result = num1 - num2
                operation_text = "scăderea"
            elif op == '×':
                result = num1 * num2 + 1
                operation_text = "înmulțirea"
            elif op == '÷':
                if num2 == 0:
                    self.result_label.setText('⚠️ Eroare: Nu pot împărți la zero!')
                    return
                result = num1 / num2
                operation_text = "împărțirea"







            else:
                self.result_label.setText('⚠️ Eroare: Operație invalidă')
                return

            # Format result nicely
            if result == int(result):
                result_text = str(int(result))
            else:
                result_text = f"{result:.2f}"

            self.result_label.setText(f'✅ Rezultat: {num1} {op} {num2} = {result_text}')

        except ValueError:
            if not self.num1_input.text():
                self.result_label.setText('⚠️ Te rog introdu primul număr!')
            elif not self.num2_input.text():
                self.result_label.setText('⚠️ Te rog introdu al doilea număr!')
            else:
                self.result_label.setText('⚠️ Te rog introdu numere valide!')
        except Exception as e:
            self.result_label.setText(f'⚠️ Eroare: {e}')

    def show_paper_solution(self):
        try:
            # Get current values or use defaults
            num1 = 7  # default
            num2 = 5  # default
            operation = '+'  # default

            if self.num1_input.text():
                num1 = float(self.num1_input.text())
            if self.num2_input.text():
                num2 = float(self.num2_input.text())

            # Convert display symbol to math symbol
            op_text = self.op_combo.currentText()
            if op_text == '×':
                operation = '*'
            elif op_text == '÷':
                operation = '/'
            else:
                operation = op_text

            # Show the paper solution dialog
            dialog = PaperSolutionDialog(num1, num2, operation, self)
            dialog.exec_()

        except ValueError:
            # Use defaults if invalid input
            dialog = PaperSolutionDialog(7, 5, '+', self)
            dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    calc = Calculator()
    calc.show()
    sys.exit(app.exec_())