from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMessageBox, QVBoxLayout
import sys
import torch
import sys
from prepare import Prepare
from global_vars import browseDict, getNewModel, getNewTokenizer, task_type, options
import sys


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        print("AAAAAAAAAAAAAAAAAAAAAa")
        super().__init__()
        self.setWindowTitle("BojAI Vexor")
        self.setFixedSize(960, 800)

        # Main layout (Vertical)
        main_layout = QtWidgets.QVBoxLayout()

        # Title
        title = QtWidgets.QLabel("BojAI Vexor")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: black;")
        title.setAlignment(QtCore.Qt.AlignLeft)
        main_layout.addWidget(title)  # Top, centered

        # Logo and Welcome
        logo_welcome_layout = QtWidgets.QVBoxLayout()

        logo = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap("logo_processed.png")
        scaled_pixmap = pixmap.scaled(
            400, 400, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        logo.setPixmap(scaled_pixmap)
        logo.setFixedSize(400, 400)

        text_label = QtWidgets.QLabel("Welcome to BojAI Vexor Applet")
        text_label.setStyleSheet("font-size: 30px; font-weight: bold; color: black;")
        spacer = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        logo_welcome_layout.addWidget(logo, alignment=QtCore.Qt.AlignCenter)
        logo_welcome_layout.addWidget(text_label, alignment=QtCore.Qt.AlignCenter)
        logo_welcome_layout.addSpacerItem(spacer)

        main_layout.addLayout(logo_welcome_layout)  # Below title, centered

        # Form and Buttons
        model_name = self.form_layout_creator()
        main_layout.addLayout(model_name)
        data_address = self.form_address_layout()
        main_layout.addLayout(data_address)

        division_text_layout = QtWidgets.QVBoxLayout()
        text_label = QtWidgets.QLabel(
            "Enter a number for dividing your data into training and evaluation (the two must add to 1)"
        )
        text_label.setStyleSheet("font-size: 16px; color: black;")
        division_text_layout.addWidget(text_label)
        main_layout.addLayout(division_text_layout)

        division_layout = QtWidgets.QHBoxLayout()
        division_layout.addLayout(self.train_div_layout())
        division_layout.addLayout(self.eval_div_layout())

        main_layout.addLayout(division_layout)

        if browseDict["options"] == 1:
            from global_vars import options

            main_layout.addLayout(self.select_tokenizer_layout())

        button_layout = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("START →")
        self.start_button.setStyleSheet(
            "background-color: #642165; color: white; font-size: 14px;"
        )
        self.start_button.clicked.connect(self.start_action)
        button_layout.addWidget(self.start_button)

        self.cancel_button = QtWidgets.QPushButton("CANCEL")
        self.cancel_button.setStyleSheet(
            "background-color: lightgrey; font-size: 14px;"
        )
        self.cancel_button.clicked.connect(self.cancel_action)
        button_layout.addWidget(self.cancel_button)
        button_layout.setAlignment(QtCore.Qt.AlignRight)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def form_layout_creator(self):
        form_button_layout = QtWidgets.QVBoxLayout()
        form_layout = QtWidgets.QFormLayout()

        # Set background color of the form container
        form_widget = QtWidgets.QWidget()

        # Create the model name input field
        self.model_name_input = QtWidgets.QLineEdit()
        self.model_name_input.setFixedSize(670, 40)  # Increased height
        self.model_name_input.setStyleSheet(
            """
            background-color: white;
            color: black;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        """
        )

        label = QtWidgets.QLabel("Enter Model Name")
        label.setStyleSheet("font-size: 16px; color: black;")

        form_layout.addRow(label)
        form_layout.addRow(self.model_name_input)

        # Apply layout to form widget
        form_widget.setLayout(form_layout)
        form_button_layout.addWidget(form_widget)

        return form_button_layout

    def train_div_layout(self):
        form_button_layout = QtWidgets.QVBoxLayout()
        form_layout = QtWidgets.QFormLayout()

        # Set background color of the form container
        form_widget = QtWidgets.QWidget()

        # Create the model name input field
        self.training_input = QtWidgets.QLineEdit()
        self.training_input.setFixedSize(200, 40)  # Increased height
        self.training_input.setStyleSheet(
            """
            background-color: white;
            color: black;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        """
        )

        label = QtWidgets.QLabel("Training")
        label.setStyleSheet("font-size: 16px; color: black;")

        form_layout.addRow(label)
        form_layout.addRow(self.training_input)

        # Apply layout to form widget
        form_widget.setLayout(form_layout)
        form_button_layout.addWidget(form_widget)

        return form_button_layout

    def eval_div_layout(self):
        form_button_layout = QtWidgets.QVBoxLayout()
        form_layout = QtWidgets.QFormLayout()

        # Set background color of the form container
        form_widget = QtWidgets.QWidget()

        # Create the model name input field
        self.eval_input = QtWidgets.QLineEdit()
        self.eval_input.setFixedSize(200, 40)  # Increased height
        self.eval_input.setStyleSheet(
            """
            background-color: white;
            color: black;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        """
        )

        label = QtWidgets.QLabel("Evaluation")
        label.setStyleSheet("font-size: 16px; color: black;")

        form_layout.addRow(label)
        form_layout.addRow(self.eval_input)

        # Apply layout to form widget
        form_widget.setLayout(form_layout)
        form_button_layout.addWidget(form_widget)

        return form_button_layout

    def form_address_layout(self):
        form_button_layout = QtWidgets.QVBoxLayout()
        form_layout = QtWidgets.QFormLayout()

        # Set background color of the form container
        form_widget = QtWidgets.QWidget()

        # Create the model name input field
        self.data_address_input = QtWidgets.QLineEdit()
        self.data_address_input.setFixedSize(570, 40)  # Increased height
        self.data_address_input.setStyleSheet(
            """
            background-color: white;
            color: black;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        """
        )

        label = QtWidgets.QLabel("Enter Data Address")
        label.setStyleSheet("font-size: 16px; color: black;")

        form_layout.addRow(label)

        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.setFixedSize(90, 40)  # Adjusted height to match input field
        browse_button.setStyleSheet(
            """
            background-color: #642165;
            color: white;
            border-radius: 10px;
            font-size: 14px;
            padding: 5px;
        """
        )
        browse_button.clicked.connect(self.browse_file)

        form_layout.addRow(self.data_address_input, browse_button)

        # Apply layout to form widget
        form_widget.setLayout(form_layout)
        form_button_layout.addWidget(form_widget)

        return form_button_layout

    def select_tokenizer_layout(self):
        selection_layout = QVBoxLayout()
        self.comboBox = QtWidgets.QComboBox(self)
        self.comboBox.addItems(options.keys())
        self.comboBox.setStyleSheet(
            """
            background-color: white;
            color: black;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        """
        )
        self.explanation1b = QtWidgets.QLabel(
            "Go to the documentation to learn how to select the right option."
        )
        self.explanation1b.setStyleSheet("font-size: 16px; color: black;")

        self.explanation2 = QtWidgets.QLabel(
            "You cannot change this after starting the session, to change it, start a new session."
        )
        self.explanation2.setStyleSheet("font-size: 16px; color: red;")

        selection_layout.addWidget(self.explanation1b)
        selection_layout.addWidget(self.comboBox)
        selection_layout.addWidget(self.explanation2)

        return selection_layout

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)

        # Create a linear gradient from #6421 to white (vertical gradient)
        gradient = QtGui.QLinearGradient(
            0, 0, 0, self.height()
        )  # From top (y=0) to bottom (y=self.height())
        gradient.setColorAt(0, QtGui.QColor("#642165"))  # Dark color (hex #6421)
        gradient.setColorAt(1, QtCore.Qt.white)  # White at the bottom

        # Set gradient as brush and paint the background
        painter.setBrush(gradient)
        painter.drawRect(self.rect())  # Fill the whole widget area with the gradient

        painter.end()

    def browse_file(self):
        file_dialog = QtWidgets.QFileDialog()
        which_one = browseDict["init"]
        if which_one:
            file_path, _ = file_dialog.getOpenFileName(self, "Select File")
        else:
            file_path = file_dialog.getExistingDirectory(self, "Select Directory")
        if file_path:
            self.data_address_input.setText(file_path)

    def start_action(self):
        model_name = self.model_name_input.text()
        data_address = self.data_address_input.text()
        training_div = self.training_input.text()
        eval_div = self.eval_input.text()
        if browseDict["options"] == 1:
            architecture = self.comboBox.currentText()
        try:
            training_div = float(training_div)
            eval_div = float(eval_div)
        except ValueError:
            self.show_error_dialog("Train and eval must be numbers added to 1")
            return

        if training_div + eval_div != 1:
            self.show_error_dialog("Train and eval must add to 1")
            return

        try:
            if browseDict["options-where"] == 0:
                model = getNewModel()
                tokenizer = options[architecture]()
            elif browseDict["options-where"] == 1:
                model = options[architecture]()
                tokenizer = getNewTokenizer()
            else:
                model = getNewModel()
                tokenizer = getNewTokenizer()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            prep = Prepare(
                model_name,
                model,
                device,
                tokenizer,
                data_address,
                task_type,
                (training_div, eval_div),
                ",",
            )

        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")  # Display error message
            msg.exec_()
            return

        # Open the new window when the button is clicked
        self.open_prep_window(prep)

    def open_prep_window(self, prep):
        # Create and show the new window
        from prepareUI import PrepWindow

        try:
            self.new_window = PrepWindow(prep)
            self.new_window.show()
        except Exception as e:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")  # Display error message
            msg.exec_()

        else:
            self.close()

    def show_error_dialog(self, message):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText(message)
        msg.exec_()

    def cancel_action(self):
        exit(0)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
