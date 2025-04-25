from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFrame
from global_vars import (
    browseDict,
    hyper_params,
    getNewModel,
    getNewTokenizer,
    task_type,
)
import torch
import sys
from prepare import Prepare
from train import Train
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpacerItem,
    QSizePolicy,
)


class PrepWindow(QWidget):
    def __init__(self, prep: Prepare, trained=False):
        self.prep = prep
        self.trained = trained
        super().__init__()

        self.setWindowTitle("BojAI Vexor - Preparation")

        # Main layout with QGridLayout to structure the window
        main_layout = QVBoxLayout()

        # Title Section
        title = QLabel("BojAI Vexor - Preparation")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: black;")
        title.setAlignment(QtCore.Qt.AlignLeft)
        main_layout.addWidget(title)  # Title spans two columns

        # Sidebar Layout (buttons and model info)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(20)

        # Button Layout
        self.bar = QFrame()
        self.bar.setStyleSheet("background-color: #EDE4F2; border-radius: 20px;")
        self.bar.setFixedHeight(60)

        bar_layout = QHBoxLayout(self.bar)

        button_prep = QPushButton("Prepare")
        button_prep.setStyleSheet(self.default_main_style())
        button_prep.clicked.connect(self.show_prepare_window)
        bar_layout.addWidget(button_prep)
        button_prep.setFixedSize(150, 45)

        button_train = QPushButton("Train")
        button_train.setStyleSheet(self.default_style())
        button_train.clicked.connect(self.show_train_window)
        bar_layout.addWidget(button_train)
        button_train.setFixedSize(150, 45)

        button_deploy = QPushButton("Deploy")
        button_deploy.setStyleSheet(self.default_style())
        button_deploy.clicked.connect(self.show_deploy_window)
        bar_layout.addWidget(button_deploy)
        button_deploy.setFixedSize(150, 45)

        main_layout.addWidget(self.bar)

        # Spacer for pushing buttons to the top
        bottom_spacer = QSpacerItem(20, 50, QSizePolicy.Minimum, QSizePolicy.Expanding)
        sidebar_layout.addItem(bottom_spacer)

        # Model Info Section
        self.info_bar = self.create_info_section()
        sidebar_layout.addLayout(self.info_bar)

        # Update Data Section
        sidebar_layout.addLayout(
            self.create_update_data_section()
        )  # disabled due to problems

        # Add sidebar layout to the main layout
        main_layout.addLayout(sidebar_layout)

        # Data View Section
        main_layout.addLayout(self.create_data_view_section())

        # Set the final layout
        self.setLayout(main_layout)

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

    def default_style(self):
        return """
        QPushButton {
            background-color: white;
            color: black;
            border: 2px solid white;
            border-radius: 10px;
            padding: 10px;
            font-weight: bold;
        }
        """

    def default_main_style(self):
        return """
        QPushButton {
            background-color: #946695;
            color: black;
            border: 2px solid #946695;
            border-radius: 10px;
            padding: 10px;
            font-weight: bold;
        }
        """

    def create_info_section(self):
        info_layout = QGridLayout()
        info_layout.setSpacing(10)

        # Add info rows
        task_name = self.GetGroupBox("Model Name", self.prep.model_name)
        info_layout.addLayout(task_name, 0, 0)

        model_name = self.GetGroupBox("Model Type", str(type(self.prep.model).__name__))
        info_layout.addLayout(model_name, 1, 0)

        tokenizer_type = self.GetGroupBox(
            "Tokenizer Type", str(self.prep.tokenizer.__class__.__name__)
        )
        info_layout.addLayout(tokenizer_type, 1, 1)

        ready_prep = self.GetGroupBox(
            "Ready to start training", "yes" if self.prep.prep_ready else "no"
        )
        info_layout.addLayout(ready_prep, 0, 1)

        data_num = self.GetGroupBox("Number of Data", str(self.prep.num_data_points))
        info_layout.addLayout(data_num, 2, 0)

        return info_layout

    def create_update_data_section(self):
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
        browse_button.setFixedSize(90, 35)  # Adjusted height to match input field
        browse_button.setStyleSheet(
            """
            background-color: #F3EDF7;
            color: black;
            border-radius: 10px;
            font-size: 14px;
            padding: 5px;
        """
        )
        browse_button.clicked.connect(self.browse)

        form_layout.addRow(self.data_address_input, browse_button)

        update_button = QPushButton("Update Data")
        update_button.setStyleSheet(
            """
            background-color: #642165;
            color: white;
            border-radius: 10px;
            font-size: 14px;
            padding: 5px;
        """
        )

        update_button.setFixedSize(570 + 90, 40)
        update_button.clicked.connect(self.change_data)

        form_layout.addRow(update_button)

        # Apply layout to form widget
        form_widget.setLayout(form_layout)
        form_button_layout.addWidget(form_widget)

        return form_button_layout

    def create_data_view_section(self):
        form_button_layout = QtWidgets.QVBoxLayout()
        form_layout = QtWidgets.QFormLayout()

        # Set background color of the form container
        form_widget = QtWidgets.QWidget()

        # Create the model name input field
        self.view_toenized_input = QtWidgets.QLineEdit()
        self.view_toenized_input.setFixedSize(570, 40)  # Increased height
        self.view_toenized_input.setStyleSheet(
            """
            background-color: white;
            color: black;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        """
        )

        label = QtWidgets.QLabel("View tokenized or untokenized data from your dataset")
        label.setStyleSheet("font-size: 16px; color: black;")

        form_layout.addRow(label)

        form_layout.addRow(self.view_toenized_input)

        buttons_layout = QHBoxLayout()

        view_raw_button = QPushButton("View Raw")
        view_raw_button.setStyleSheet(
            """
            background-color: #642165;
            color: white;
            border-radius: 10px;
            font-size: 14px;
            padding: 5px;
        """
        )

        view_raw_button.setFixedSize(int((570 + 90) / 2), 40)
        view_raw_button.clicked.connect(self.view_raw)
        buttons_layout.addWidget(view_raw_button)

        view_tokenized_button = QPushButton("View Tokenized")
        view_tokenized_button.setStyleSheet(
            """
            background-color: #642165;
            color: white;
            border-radius: 10px;
            font-size: 14px;
            padding: 5px;
        """
        )

        view_tokenized_button.setFixedSize(int((570 + 90) / 2), 40)
        view_tokenized_button.clicked.connect(self.view_tokenized)
        buttons_layout.addWidget(view_tokenized_button)

        form_layout.addRow(buttons_layout)

        # Apply layout to form widget
        form_widget.setLayout(form_layout)
        form_button_layout.addWidget(form_widget)

        return form_button_layout

    def GetGroupBox(self, title, inside):
        form_layout = QtWidgets.QFormLayout()

        # Create the model name input field

        label = QtWidgets.QLabel(title)
        label.setStyleSheet("font-size: 16px; color: black;")

        inside = QtWidgets.QLabel(inside)
        inside.setStyleSheet(
            """
            font-size: 16px;
            color: black;
            background-color: #F3EDF7;
            border: 2px solid #F3EDF7;
            border-radius: 10px;
            padding: 5px;  /* Adds some spacing inside */
        """
        )

        form_layout.addRow(label)
        form_layout.addRow(inside)

        return form_layout

    def view_tokenized(self):
        index_text = self.view_toenized_input.text().strip()  # Strip whitespace
        message = ""

        try:
            if index_text == "":
                message = str(self.prep.check_tokenized())
            elif index_text.isdigit():  # Ensure input is a valid number
                index = int(index_text)
                message = str(self.prep.check_tokenized(index - 1))
            else:
                message = "Invalid input. Please enter a valid index."

        except IndexError:
            message = (
                "the index you entered is too big, make sure you enter a right one"
            )
        except Exception as e:
            message = f"An error occurred: {str(e)}"

        # Show message box
        success_msg = QtWidgets.QMessageBox()
        success_msg.setWindowTitle("Tokenized Data")
        success_msg.setText(str(message))
        success_msg.setIcon(QtWidgets.QMessageBox.Information)
        success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        success_msg.exec_()

    def view_raw(self):
        index_text = self.view_toenized_input.text().strip()  # Strip whitespace
        message = ""

        try:
            if index_text == "":
                output = self.prep.view_raw_data()
                which_one = browseDict["type"]
                if which_one == 0:
                    message = output[1]
                    image = output[0]
                    image.show()
                elif which_one == 1:
                    # tbd
                    pass
                else:
                    message = output
            elif index_text.isdigit():  # Ensure input is a valid number
                index = int(index_text)
                output = self.prep.view_raw_data(index - 1)
                which_one = browseDict["type"]
                if which_one == 0:
                    message = output[1]
                    image = output[0]
                    image.show()
                if which_one == 1:
                    # tbd
                    pass
                else:
                    message = output
            else:
                message = "Invalid input. Please enter a valid index."

        except IndexError:
            message = (
                "the index you entered is too big, make sure you enter a right one"
            )
        except Exception as e:
            message = f"An error occurred: {str(e)}"

        # Show message box
        success_msg = QtWidgets.QMessageBox()
        success_msg.setWindowTitle("Tokenized Data")
        success_msg.setText(str(message))
        success_msg.setIcon(QtWidgets.QMessageBox.Information)
        success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        success_msg.exec_()

    def browse(self):
        file_dialog = QtWidgets.QFileDialog()
        which_one = browseDict["prep"]
        if which_one:
            file_path, _ = file_dialog.getOpenFileName(self, "Select File")
        else:
            file_path = file_dialog.getExistingDirectory(self, "Select Directory")
        if file_path:
            self.data_address_input.setText(file_path)

    def change_data(self):
        data_address = self.data_address_input.text()
        if data_address == "":
            success_msg = QtWidgets.QMessageBox()
            success_msg.setWindowTitle("Failed")
            success_msg.setText("Data field cannot be empty")
            success_msg.setIcon(QtWidgets.QMessageBox.Information)
            success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            success_msg.exec_()
            return

        try:
            self.prep.update_data(data_address)
        except Exception as e:
            success_msg = QtWidgets.QMessageBox()
            success_msg.setWindowTitle("Error")
            success_msg.setText(str(e))
            success_msg.setIcon(QtWidgets.QMessageBox.Information)
            success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            success_msg.exec_()
            return

        success_msg = QtWidgets.QMessageBox()
        success_msg.setWindowTitle("Success")
        success_msg.setText("Data updated successfully!")
        success_msg.setIcon(QtWidgets.QMessageBox.Information)
        success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        success_msg.exec_()

        data_num = self.GetGroupBox("Number of Data", str(self.prep.num_data_points))
        self.info_bar.addLayout(data_num, 2, 0)

    def show_prepare_window(self):
        pass

    def show_train_window(self):
        if not self.prep.prep_ready:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(
                "An error occurred in prep, most probably because we cannot connect to our server to validate this session. If it persists contact support"
            )
            msg.exec_()
            return

        # Create and show the new window
        try:
            hyperparams = hyper_params
            train = Train(self.prep, hyperparams)
            from trainUI import TrainWindow

            self.new_window = TrainWindow(train, trained=self.trained)
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")
            msg.exec_()
            return

        self.new_window.show()

        # Optionally, you can close the current window (main window)
        self.close()

    def show_deploy_window(self):
        success_msg = QtWidgets.QMessageBox()
        success_msg.setWindowTitle("Error")
        success_msg.setText(
            "Deployment can only happen after training. Go to training first"
        )
        success_msg.setIcon(QtWidgets.QMessageBox.Information)
        success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        success_msg.exec_()


if __name__ == "__main__":

    model_name = ""
    data_address = input("enter dataset address: ")
    training_div = 0.8
    eval_div = 0.2
    tokenizer = getNewTokenizer()
    model = getNewModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prep = Prepare(
        model_name,
        model,
        device,
        tokenizer,
        data_address,
        task_type,
        (training_div, eval_div),
        "",
    )
    app = QtWidgets.QApplication(sys.argv)
    window = PrepWindow(prep)
    window.show()
    sys.exit(app.exec_())
