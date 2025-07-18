from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMessageBox
import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFrame
import torch
import os
from PyQt5.QtCore import Qt
from global_vars import (
    browseDict,
    getNewModel,
    getNewTokenizer,
    hyper_params,
    task_type,
)
import torch
from deploy import Deploy
from prepare import Prepare
from train import Train
from PyQt5.QtWidgets import (
    QWidget,
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


class DeployWindow(QWidget):
    def __init__(self, deploy: Deploy, trained=False):
        self.deploy = deploy
        super().__init__()

        self.setWindowTitle("BojAI Vexor - Deployment")
        print("AAAAAAAAAAAAAAAAAAAAAa")
        self.trained = trained
        # Main layout with QGridLayout to structure the window
        main_layout = QGridLayout()

        # Title Section
        title = QLabel("BojAI Vexor - Deployment")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: black;")
        title.setAlignment(QtCore.Qt.AlignLeft)
        main_layout.addWidget(title, 0, 0, 1, 2)  # Title spans two columns

        # Sidebar Layout (buttons and model info)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(20)

        # Button Layout
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
        button_train.setStyleSheet(self.default_style())
        button_train.clicked.connect(self.show_train_window)
        bar_layout.addWidget(button_train)
        button_train.setFixedSize(150, 45)

        button_deploy = QPushButton("Deploy")
        button_deploy.setStyleSheet(self.default_main_style())
        button_deploy.clicked.connect(self.show_deploy_window)
        bar_layout.addWidget(button_deploy)
        button_deploy.setFixedSize(150, 45)

        main_layout.addWidget(self.bar)

        # Spacer for pushing buttons to the top
        bottom_spacer = QSpacerItem(20, 50, QSizePolicy.Minimum, QSizePolicy.Expanding)
        sidebar_layout.addItem(bottom_spacer)
        main_layout.addLayout(sidebar_layout, 1, 0)

        main_layout.addLayout(self.create_data_view_section(), 2, 0)
        main_layout.addLayout(self.evaluate_data_layout(), 3, 0)
        main_layout.addLayout(self.save_model_layout(), 4, 0)
        main_layout.addLayout(self.use_model_layout(), 5, 0)
        main_layout.addLayout(self.visualize_model(), 6, 0)

        self.main_layout = main_layout

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

    def use_model_layout(self):
        update_layout = QFormLayout()
        update_layout.setSpacing(10)

        # Data address and update button
        big_title = QLabel("Use Model")
        big_title.setStyleSheet("font-size: 16px; color: black; font-weight: bold;")

        update_title = QLabel(browseDict["use_model_text"])
        update_title.setStyleSheet("font-size: 16px; color: black;")

        self.use_model_output = QLineEdit()
        self.use_model_output.setStyleSheet(
            """
            background-color: white;
            color: black;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        """
        )
        self.use_model_output.setFixedSize(200, 40)

        update_button = QPushButton("Use Model")
        update_button.setStyleSheet(
            """
            background-color: #642165;
            color: white;
            border-radius: 10px;
            font-size: 14px;
            padding: 5px;
        """
        )
        update_button.setFixedSize(300, 35)
        update_button.clicked.connect(self.use_model)
        # Create the layout
        data_layout = QHBoxLayout()
        data_layout.addWidget(self.use_model_output)

        browse_needed = browseDict["use_model_upload"]
        if browse_needed:
            browse_button = QPushButton("Browse")
            browse_button.setFixedSize(100, 35)
            browse_button.setStyleSheet(
                """
                background-color: #F3EDF7;
                color: black;
                border-radius: 10px;
                font-size: 14px;
                padding: 5px;
            """
            )
            browse_button.clicked.connect(self.browse3)
            data_layout.addWidget(browse_button)

        update_layout.addRow(big_title)
        update_layout.addRow(update_title)
        update_layout.addRow(data_layout)
        update_layout.addRow(update_button)

        return update_layout

    def visualize_model(self):
        layout = QHBoxLayout()

        visualize_button = QPushButton("Visualize Model")
        visualize_button.setStyleSheet(
            """
            background-color: #EDE4F2; border-radius: 20px;
            """
        )
        visualize_button.setFixedSize(300, 35)
        visualize_button.clicked.connect(self.open_visualizer_window)

        layout.addWidget(visualize_button)
        return layout
    
    def use_model(self):
        try:
            self.deploy.max_length = 50
            input = self.use_model_output.text()
            output = self.deploy.use_model(input)
        except Exception as e:
            success_msg = QtWidgets.QMessageBox()
            success_msg.setWindowTitle("Error")
            success_msg.setText("something went wrong, " + str(e))
            success_msg.setIcon(QtWidgets.QMessageBox.Information)
            success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            success_msg.exec_()
            return

        success_msg = QtWidgets.QMessageBox()
        success_msg.setWindowTitle("Output")
        success_msg.setText(str(output))
        success_msg.setIcon(QtWidgets.QMessageBox.Information)
        success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        success_msg.exec_()

    def evaluate_data_layout(self):
        eval_layout = QFormLayout()
        eval_layout.setSpacing(10)

        eval_title = QLabel("Evaluate Model Output")
        eval_title.setStyleSheet("font-size: 15px; font-weight: bold; color: black;")

        update_title = QLabel(
            "Evaluate your model. Once you add a new evaluation data you can use it to get the evaluation score."
        )
        update_title.setStyleSheet("font-size: 16px; color: black;")

        eval_button = QPushButton("Evaluate model with original data")
        eval_button.setStyleSheet("font-size: 16px; color: black;")
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
        eval_button.clicked.connect(self.evaluate_a)
        eval_layout.addRow(eval_title)
        eval_layout.addRow(update_title)
        eval_layout.addRow(eval_button)

        if self.deploy.new_data != None:
            eval2_button = QPushButton("Evaluate model with new data")
            eval2_button.setStyleSheet(
                """
                background-color: #642165;
                color: white;
                border-radius: 10px;
                font-size: 14px;
                padding: 5px;
            """
            )
            eval2_button.setFixedSize(300, 35)
            eval2_button.clicked.connect(self.evaluate_b)

            eval_layout.addRow(eval2_button)

        return eval_layout

    def create_data_view_section(self):
        update_layout = QFormLayout()
        update_layout.setSpacing(10)

        # Data address and update button
        big_title = QLabel("New Evaluation Data")
        big_title.setStyleSheet("font-size: 15px; font-weight: bold; color: black;")

        update_title = QLabel(
            "Add a new evaluation data, leave empty if you want to use your original data or enter a file to evaluate other data."
        )
        update_title.setStyleSheet("font-size: 16px; color: black;")

        self.data_address_input = QLineEdit()
        self.data_address_input.setStyleSheet(
            """
            background-color: white;
            color: black;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        """
        )
        self.data_address_input.setFixedSize(300, 40)

        browse_button = QPushButton("Browse")
        browse_button.setFixedSize(100, 35)
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
        update_button.setFixedSize(300, 30)
        update_button.clicked.connect(self.update_data)

        # Create the layout
        data_layout = QHBoxLayout()
        data_layout.addWidget(self.data_address_input)
        data_layout.addWidget(browse_button)

        update_layout.addRow(big_title)
        update_layout.addRow(update_title)
        update_layout.addRow("Enter Data Address:", data_layout)
        update_layout.addRow(update_button)

        return update_layout

    def save_model_layout(self):
        update_layout = QFormLayout()
        update_layout.setSpacing(10)

        # Data address and update button
        big_title = QLabel("Download Model")
        big_title.setStyleSheet("font-size: 15px; font-weight: bold; color: black;")

        update_title = QLabel(
            "Download your model as a .bin file locally, enter the folder address."
        )
        update_title.setStyleSheet("font-size: 16px; color: black;")

        self.model_dir = QLineEdit()
        self.model_dir.setStyleSheet(
            """
            background-color: white;
            color: black;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        """
        )

        self.model_dir.setFixedSize(300, 40)

        browse_button = QPushButton("Browse")
        browse_button.setStyleSheet(
            """
            background-color: #F3EDF7;
            color: black;
            border-radius: 10px;
            font-size: 14px;
            padding: 5px;
        """
        )
        browse_button.setFixedSize(100, 35)
        browse_button.clicked.connect(self.browse2)

        update_button = QPushButton("Download Model")
        update_button.setStyleSheet(
            """
            background-color: #642165;
            color: white;
            border-radius: 10px;
            font-size: 14px;
            padding: 5px;
        """
        )
        update_button.setFixedSize(300, 35)
        update_button.clicked.connect(self.download_model)

        # Create the layout
        data_layout = QHBoxLayout()
        data_layout.addWidget(self.model_dir)
        data_layout.addWidget(browse_button)

        update_layout.addRow(big_title)
        update_layout.addRow(update_title)
        update_layout.addRow("Enter Data Address:", data_layout)
        update_layout.addRow(update_button)

        return update_layout

    def download_model(self):
        where = self.model_dir.text()
        try:
            if os.path.isdir(where) == False:
                success_msg = QtWidgets.QMessageBox()
                success_msg.setWindowTitle("Failed")
                success_msg.setText(where + " must be a directory")
                success_msg.setIcon(QtWidgets.QMessageBox.Information)
                success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                success_msg.exec_()
            else:
                save_path = os.path.join(
                    where, self.deploy.trainer.model.__class__.__name__ + ".bin"
                )
                torch.save(self.deploy.trainer.model.state_dict(), save_path)
                success_msg = QtWidgets.QMessageBox()
                success_msg.setWindowTitle("Success")
                success_msg.setText("Model saved in " + where)
                success_msg.setIcon(QtWidgets.QMessageBox.Information)
                success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                success_msg.exec_()
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")  # Display error message
            msg.exec_()

    def browse(self):
        file_dialog = QtWidgets.QFileDialog()
        which_one = browseDict["deploy_new_data"]
        if which_one:
            file_path, _ = file_dialog.getOpenFileName(self, "Select File")
        else:
            file_path = file_dialog.getExistingDirectory(self, "Select Directory")
        if file_path:
            self.data_address_input.setText(file_path)

    def browse2(self):
        file_dialog = QtWidgets.QFileDialog()
        directory_path = file_dialog.getExistingDirectory(self, "Select Directory")
        if directory_path:
            self.model_dir.setText(directory_path)

    def browse3(self):
        file_dialog = QtWidgets.QFileDialog()
        file_path = file_dialog.getOpenFileName(
            self, "Select File"
        )  # Select a single file
        if file_path:
            self.use_model_output.setText(file_path)

    def update_data(self):
        data = self.data_address_input.text()
        try:
            self.deploy.update_eval_data(data)
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")  # Display error message
            msg.exec_()

        else:
            success_msg = QtWidgets.QMessageBox()
            success_msg.setWindowTitle("Success")
            success_msg.setText("New eval data added!")
            success_msg.setIcon(QtWidgets.QMessageBox.Information)
            success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            success_msg.exec_()
            self.main_layout.addLayout(self.evaluate_data_layout(), 3, 0)

    def evaluate_a(self):
        score = self.deploy.get_eval_score(0)
        success_msg = QtWidgets.QMessageBox()
        metrice = browseDict["eval matrice"]
        success_msg.setText(f"{metrice}: " + str(score))
        success_msg.setIcon(QtWidgets.QMessageBox.Information)
        success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        success_msg.exec_()

    def evaluate_b(self):
        score = self.deploy.get_eval_score(1)
        success_msg = QtWidgets.QMessageBox()
        success_msg.setWindowTitle("Evaluation Score")
        metrice = browseDict["eval matrice"]
        success_msg.setText(f"{metrice}: " + str(score))
        success_msg.setIcon(QtWidgets.QMessageBox.Information)
        success_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        success_msg.exec_()

    def show_prepare_window(self):
        # Create and show the new window
        from prepareUI import PrepWindow

        try:
            self.new_window = PrepWindow(self.deploy.prep)
            self.new_window.show()
            self.close()
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")  # Display error message
            msg.exec_()

    def show_train_window(self):
        from trainUI import TrainWindow

        # Create and show the new window
        try:
            self.new_window = TrainWindow(self.deploy.trainer)
            self.new_window.trained = self.trained
            self.new_window.show()
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"An error occurred: {str(e)}")  # Display error message
            msg.exec_()

        # Optionally, you can close the current window (main window)
        else:
            self.close()

    def show_deploy_window(self):
        pass

    def open_visualizer_window(self):
        try:
            from visualizer import Visualizer

            class VisualizerWindow(QtWidgets.QWidget):
                def __init__(self, visualizer, deploy):
                    self.deploy : Deploy = deploy
                    super().__init__()
                    self.visualizer : Visualizer = visualizer
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


                    if self.deploy.new_data != None:
                        # Button 3: Eval vs Train
                        eval_train_btn = QPushButton("Plot Evaluation vs Training")
                        eval_train_btn.clicked.connect(
                            self.visualizer.plot_train_vs_eval
                        )
                        layout.addWidget(eval_train_btn)

                        # Button 4: Eval vs Validation
                        eval_valid_btn = QPushButton("Plot Evaluation vs Validation")
                        eval_valid_btn.clicked.connect(
                            self.visualizer.plot_valid_vs_eval
                        )
                        layout.addWidget(eval_valid_btn)

                    self.setLayout(layout)

            visualizer = Visualizer()
            self.vis_window = VisualizerWindow(visualizer, self.deploy)
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
        "",
        [0,0,0]
    )
    hyperparams = hyper_params
    train = Train(prep, hyperparams)

    deploy = Deploy(train)
    app = QtWidgets.QApplication(sys.argv)
    window = DeployWindow(deploy)
    window.show()
    sys.exit(app.exec_())
