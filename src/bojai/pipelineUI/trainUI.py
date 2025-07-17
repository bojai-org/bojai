from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QSpacerItem,
    QSizePolicy,
    QLabel,
    QFrame,
)

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QProgressBar, QDialog, QVBoxLayout, QLabel, QPushButton
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QLabel
from global_vars import (
    browseDict,
    getNewModel,
    init_model,
    getNewModel,
    getNewTokenizer,
    hyper_params,
    task_type,
)
import torch
import sys
from prepare import Prepare
from train import Train  # Make sure 'train' module exists
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpacerItem,
    QSizePolicy,
    QFormLayout,
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import QThread, pyqtSignal, Qt


class TrainingThread(QThread):
    # Define the signal to update progress
    progress_updated = pyqtSignal(int, arguments=["progress"])
    loss_updated = pyqtSignal(float)

    def __init__(self, trainer: Train):
        super().__init__()
        self.trainer = trainer.trainerManager.trainer

    def run(self):
        try:
            self.trainer.train(QThread(), self.progress_updated, self.loss_updated)
        except Exception as e:
            success_msg = QtWidgets.QMessageBox()
            success_msg.setWindowTitle("One problem ")
            success_msg.setText("something went wrong," + str(e))
            success_msg.setIcon(QtWidgets.QMessageBox.Information)
            success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            success_msg.exec_()
        return


class TrainWindow(QtWidgets.QWidget):
    def __init__(self, train: Train, deploy=None, trained=False):
        super().__init__()
        self.train = train
        self.setWindowTitle("BojAI Vexor - Training")
        self.trained = trained
        self.deploy = deploy

        # Main layout with QGridLayout to structure the window
        self.main_layout = (
            QGridLayout()
        )  # Switch to QVBoxLayout for simpler layout management

        # Title Section
        title = QLabel("BojAI Vexor - Training")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: black;")
        title.setAlignment(QtCore.Qt.AlignLeft)
        self.main_layout.addWidget(title)

        # Sidebar Layout (buttons and model info)
        self.bar = QFrame()
        self.bar.setStyleSheet("background-color: #EDE4F2; border-radius: 20px;")
        self.bar.setFixedHeight(60)

        bar_layout = QHBoxLayout(self.bar)

        button_prep = QPushButton("Prepare")
        button_prep.setStyleSheet(self.default_style())
        button_prep.clicked.connect(self.show_prepare_window)
        bar_layout.addWidget(button_prep)
        button_prep.setFixedSize(150, 45)

        button_train = QPushButton("Train")
        button_train.setStyleSheet(self.default_main_style())
        button_train.clicked.connect(self.show_train_window)
        bar_layout.addWidget(button_train)
        button_train.setFixedSize(150, 45)

        button_deploy = QPushButton("Deploy")
        button_deploy.setStyleSheet(self.default_style())
        button_deploy.clicked.connect(self.show_deploy_window)
        bar_layout.addWidget(button_deploy)
        button_deploy.setFixedSize(150, 45)

        self.main_layout.addWidget(self.bar)

        # Spacer for pushing buttons to the top
        bottom_spacer = QSpacerItem(20, 50, QSizePolicy.Minimum, QSizePolicy.Expanding)
        bar_layout.addSpacerItem(bottom_spacer)
        info_layout = self.create_info_section()
        # Add sidebar_layout to main_layout
        self.main_layout.addLayout(
            bar_layout, 1, 0
        )  # Sidebar is in the left column, spans two rows
        self.main_layout.addLayout(info_layout, 2, 0)
        # Hyperparameter Update Section (Right Column)
        self.main_layout.addLayout(
            self.create_train_section(), 3, 0
        )  # Data section above
        self.main_layout.addLayout(self.create_eval_layout(), 4, 0)
        update_layout = self.update_hyperparam_layout()
        self.main_layout.addLayout(update_layout, 5, 0)  # Hyperparameter section below
        self.main_layout.addLayout(self.replace_model_layout(), 6, 0)
        self.main_layout.addLayout(self.visualize_model(), 7,0)
        self.setLayout(self.main_layout)

    def create_eval_layout(self):
        eval_layout = QFormLayout()
        eval_layout.setSpacing(10)

        eval_title = QLabel("Evaluate Model Output")
        eval_title.setStyleSheet("font-size: 16px; color: black; font-weight: bold;")

        description = QLabel(
            "Evaluate how good the model output is compared to expected output."
        )
        description.setStyleSheet("font-size: 16px; color: black;")

        eval_button = QPushButton("Evaluate Model")
        eval_button.setStyleSheet(
            """
            background-color: #642165;
            color: white;
            border-radius: 10px;
            font-size: 14px;
            padding: 5px;
        """
        )
        eval_button.setFixedSize(300, 35)
        eval_button.clicked.connect(self.evaluate)

        eval_layout.addRow(eval_title)
        eval_layout.addRow(description)
        eval_layout.addRow(eval_button)

        return eval_layout
    
    def visualize_model(self):
        layout = QHBoxLayout()

        visualize_button = QPushButton("Visualize Model")
        visualize_button.setStyleSheet(
            """
            background-color: #642165;
            color: white;
            border-radius: 10px;
            font-size: 14px;
            padding: 5px;
            """
        )
        visualize_button.setFixedSize(300, 35)
        visualize_button.clicked.connect(self.open_visualizer_window)

        layout.addWidget(visualize_button)
        return layout
    
    def replace_model_layout(self):
        replace_layout = QFormLayout()
        replace_layout.setSpacing(10)

        replace_big = QLabel("Replace Model")
        replace_big.setStyleSheet("font-size: 16px; color: black; font-weight: bold;")
        replace_title = QLabel(
            "Replace the model with untrained one to restart training."
        )
        replace_title.setStyleSheet("font-size: 16px; color: black;")

        caution_title = QLabel(
            "CAUTION: clicking this button will replace your current model with an untrained one!"
        )
        caution_title.setStyleSheet("font-size: 15px; font-weight: bold; color: red;")

        caution_button = QPushButton("Replace Model")
        caution_button.setStyleSheet(
            """
            background-color: #642165;
            color: white;
            border-radius: 10px;
            font-size: 14px;
            padding: 5px;
        """
        )
        caution_button.setFixedSize(300, 30)
        caution_button.clicked.connect(self.replace)

        replace_layout.addRow(replace_big)
        replace_layout.addRow(replace_title)
        replace_layout.addRow(caution_title)
        replace_layout.addRow(caution_button)

        return replace_layout

    def replace(self):
        try:
            self.train.trainerManager.trainer.model = getNewModel()
            init_model(self.train.prep.data, self.train.trainerManager.trainer.model)

        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")  # Display error message
            msg.exec_()

        success_msg = QtWidgets.QMessageBox()
        success_msg.setWindowTitle("Success ")
        success_msg.setText(
            "model updated successfully, now you have an initial untrained model"
        )
        success_msg.setIcon(QtWidgets.QMessageBox.Information)
        success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        success_msg.exec_()

    def evaluate(self):
        try:
            score = self.train.trainerManager.trainer.evaluate()
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")  # Display error message
            msg.exec_()

        success_msg = QtWidgets.QMessageBox()
        success_msg.setWindowTitle("Evaluation Score")
        success_msg.setText(browseDict["eval matrice"] + str(score))
        success_msg.setIcon(QtWidgets.QMessageBox.Information)
        success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        success_msg.exec_()

    def browse(self):
        file_dialog = QtWidgets.QFileDialog()
        which_one = browseDict["train"]
        if which_one:
            directory_path = file_dialog.getOpenFileName(self, "Select File")
        else:
            directory_path = file_dialog.getExistingDirectory(self, "Select Directory")
        if directory_path:
            self.data_address_input.setText(directory_path)

    def create_train_section(self):
        update_layout = QFormLayout()
        update_layout.setSpacing(10)

        # Data address and update button
        training_title = QLabel("Start Training")
        training_title.setStyleSheet(
            "font-size: 16px; color: black; font-weight: bold;"
        )

        description = QLabel(
            "Train your model, once you start you cannot stop until training ends."
        )
        description.setStyleSheet("font-size: 16px; color: black;")

        train_button = QPushButton("Start Training")
        train_button.setStyleSheet(
            """
            background-color: #642165;
            color: white;
            border-radius: 10px;
            font-size: 14px;
            padding: 5px;
        """
        )
        train_button.setFixedSize(300, 35)
        train_button.clicked.connect(self.start_training)

        # Create the layout
        update_layout.addRow(training_title)
        update_layout.addRow(description)
        update_layout.addRow(train_button)

        return update_layout

    def start_training(self):
        if self.train.model.__class__.__name__ == "kNN":
            self.trained = True
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("No training needed")
            msg.setText(
                f"kNN model does not need training, you can evaluate"
            )  # Display error message
            msg.exec_()
            return
        # Create progress window
        self.progress_window = QDialog(self)
        self.progress_window.setWindowTitle("Training Progress")
        self.progress_window.setModal(True)

        layout = QVBoxLayout()

        # Status label
        self.status_label = QLabel("Training in Progress...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Loss label
        self.loss_label = QLabel("Loss: N/A")  # NEW: Display loss
        self.loss_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loss_label)

        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancel_training)
        layout.addWidget(cancel_button)

        self.progress_window.setLayout(layout)
        self.progress_window.show()

        # Start training
        try:
            self.training_thread = TrainingThread(self.train)
            self.training_thread.progress_updated.connect(self.update_progress)
            self.training_thread.loss_updated.connect(self.update_loss)
            self.training_thread.start()
            self.trained = True
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")
            msg.exec_()

    def update_progress(self, progress):
        # Update the progress bar with the current progress
        self.progress_bar.setValue(progress)

    def update_loss(self, loss):
        self.loss_label.setText(f"Loss: {loss:.4f}")  # Update UI label

    def cancel_training(self):
        # Optionally, handle the cancellation of training here

        self.training_thread.terminate()  # Forcefully stop the thread
        self.progress_window.close()

    def update_progress(self, progress):
        # Update the progress bar with the current progress
        self.progress_bar.setValue(progress)

    def cancel_training(self):
        # Optionally, handle the cancellation of training here
        self.training_thread.terminate()  # Forcefully stop the thread
        self.progress_window.close()

    def update_hyperparam_layout(self):
        # Create a form layout for updating hyperparameters
        update_layout = QFormLayout()
        update_layout.setSpacing(10)

        update_title = QLabel("Update Hyperparameters")
        update_title.setStyleSheet("font-size: 16px; color: black; font-weight: bold;")

        update_layout.addRow(update_title)

        description = QLabel("Change the vlaues of the hyperparameters listed below.")
        description.setStyleSheet("font-size: 16px; color: black;")
        update_layout.addRow(description)

        # Create a list to store QLineEdit widgets for updating hyperparameters
        self.hyperparam_inputs = []
        try:
            # Loop over the current hyperparameters and create input fields for each
            for (
                param_name,
                param_value,
            ) in self.train.trainerManager.hyperparams.items():
                # Create a label with the hyperparameter name
                label = QLabel(param_name)
                label.setStyleSheet("font-size: 16px; color: black;")

                # Create an input field for updating the hyperparameter value
                input_field = QLineEdit()
                input_field.setStyleSheet(
                    """
                    background-color: white;
                    color: black;
                    border-radius: 10px;
                    padding: 10px;
                    font-size: 16px;
                """
                )

                # Store the input fields in the list
                self.hyperparam_inputs.append(input_field)

                # Add the label and input field as a row in the form layout
                update_layout.addRow(label)
                update_layout.addRow(input_field)

        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")  # Display error message
            msg.exec_()

        # Create the "Update Hyperparameters" button
        self.update_button = QPushButton("Update Hyperparameters")
        self.update_button.setStyleSheet(
            """
            background-color: #642165;
            color: white;
            border-radius: 10px;
            font-size: 14px;
            padding: 5px;
        """
        )
        self.update_button.clicked.connect(
            self.update_hyperparams
        )  # Connect to the update function

        update_layout.addRow(self.update_button)

        return update_layout

    def update_hyperparams(self):
        # Create a dictionary to hold the new hyperparameter values
        new_hyper = {}

        # Loop through the hyperparam inputs and param names by index
        try:
            for index, input_field in enumerate(self.hyperparam_inputs):
                param_name = list(self.train.trainerManager.hyperparams.keys())[
                    index
                ]  # Get the parameter name by index
                new_value = input_field.text()  # Get the new value from the input field

                # If the input field is empty, retain the old value (i.e., keep it unchanged)
                if not new_value.strip():  # Check if the input field is empty
                    new_hyper[param_name] = self.train.trainerManager.hyperparams.get(
                        param_name
                    )  # Retain the old value
                else:
                    # Try to convert the new value to the correct type (int, float, or string)
                    try:
                        # If the hyperparameter is numeric, try to convert it
                        if "." in new_value:
                            new_hyper[param_name] = float(new_value)
                        else:
                            new_hyper[param_name] = int(new_value)
                    except ValueError:
                        success_msg = QtWidgets.QMessageBox()
                        success_msg.setWindowTitle("Value Error")
                        success_msg.setText("Wrong value, please enter valid values")
                        success_msg.setIcon(QtWidgets.QMessageBox.Information)
                        success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                        success_msg.exec_()
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")  # Display error message
            msg.exec_()

        # Pass the updated hyperparameters to the `edit_hyperparams` method
        try:
            self.train.edit_hyperparams(new_hyper)
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")  # Display error message
            msg.exec_()

        info_layout_new = self.create_info_section()
        self.main_layout.addLayout(info_layout_new, 2, 0)

        # Optionally, show a message box confirming the update
        QtWidgets.QMessageBox.information(
            self, "Success", "Hyperparameters updated successfully!"
        )

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

    def create_info_section(self):
        info_layout = QGridLayout()
        info_layout.setSpacing(10)
        update_title = QLabel("Hyperparameters")
        update_title.setStyleSheet("font-size: 16px; color: black; font-weight: bold;")

        info_layout.addWidget(update_title, 0, 0)

        description = QLabel("See the vlaues of the hyperparameters listed below.")
        description.setStyleSheet("font-size: 16px; color: black;")
        info_layout.addWidget(description, 1, 0)

        # Loop through hyperparameters and create a GetGroupBox for each
        for index, (param_name, param_value) in enumerate(
            self.train.trainerManager.hyperparams.items()
        ):
            param_groupbox = self.GetGroupBox(param_name, str(param_value))

            row = index // 2  # Adjust 3 to control how many columns per row
            col = index % 2

            info_layout.addLayout(param_groupbox, row + 2, col)

        device_groupbox = self.GetGroupBox("device", str(self.train.prep.device))
        index = len(self.train.trainerManager.hyperparams)
        row = index // 2
        col = index % 2
        info_layout.addLayout(device_groupbox, row + 2, col)

        return info_layout

    def show_prepare_window(self):
        from prepareUI import PrepWindow

        try:
            self.new_window = PrepWindow(self.train.prep, self.trained)
            self.new_window.show()
            self.close()
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")  # Display error message
            msg.exec_()

    def show_train_window(self):
        # Implement the behavior for the "Train" button here
        pass

    def show_deploy_window(self):
        if self.trained == False:
            success_msg = QtWidgets.QMessageBox()
            success_msg.setWindowTitle("Error")
            success_msg.setText(
                "Please train your model first. Cannot deploy an untrained model"
            )
            success_msg.setIcon(QtWidgets.QMessageBox.Information)
            success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            success_msg.exec_()
            return
        from deployUI import DeployWindow
        from deploy import Deploy

        try:
            if self.deploy == None:
                self.deploy = Deploy(self.train)
            self.new_window = DeployWindow(self.deploy, self.trained)
            self.new_window.show()
            self.close()
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")  # Display error message
            msg.exec_()

    def open_visualizer_window(self):
        try:
            from visualizer import Visualizer

            class VisualizerWindow(QtWidgets.QWidget):
                def __init__(self, visualizer):
                    super().__init__()
                    self.visualizer = visualizer
                    self.setWindowTitle("Model Visualizations")
                    self.setGeometry(100, 100, 400, 300)
                    layout = QVBoxLayout()

                    title = QLabel("Choose a visualization")
                    title.setAlignment(Qt.AlignCenter)
                    title.setStyleSheet("font-size: 18px; font-weight: bold; color: black;")
                    layout.addWidget(title)

                    # Button 1: Loss vs Epoch
                    loss_btn = QPushButton("Plot Loss vs Epoch")
                    loss_btn.clicked.connect(self.visualizer.plot_loss)
                    layout.addWidget(loss_btn)

                    # Button 2: Train vs Validation
                    val_train_btn = QPushButton("Plot Training vs Validation")
                    val_train_btn.clicked.connect(self.visualizer.plot_validation_vs_training)
                    layout.addWidget(val_train_btn)

                    self.setLayout(layout)

            visualizer = Visualizer()
            self.vis_window = VisualizerWindow(visualizer)
            self.vis_window.show()

        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Could not open visualizer window: {str(e)}")
            msg.exec_()


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
        "", [0,0,0]
    )
    hyperparams = hyper_params
    train = Train(prep, hyperparams)
    app = QtWidgets.QApplication(sys.argv)
    window = TrainWindow(train)

    window.show()
    sys.exit(app.exec_())
