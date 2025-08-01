import time
from train import Train
from deploy import Deploy
import torch
from global_vars import (
    browseDict,
    getNewModel,
    getNewTokenizer,
    hyper_params,
    task_type,
    init_model,
)
from deployCLI import deploy_cli
import sys
from visualizer import Visualizer

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def print_header(title):
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)


def display_hyperparams(train: Train):
    print_header("📊 Current Hyperparameters")
    for name, val in train.trainerManager.hyperparams.items():
        print(f"{name}: {val}")
    print(f"Device: {train.prep.device}")


def update_hyperparams(train: Train):
    print_header("🔧 Update Hyperparameters")
    new_params = {}
    for name, old_val in train.trainerManager.hyperparams.items():
        val = input(f"{name} (current: {old_val}) → ").strip()
        if val == "":
            new_params[name] = old_val
        else:
            try:
                new_params[name] = float(val) if "." in val else int(val)
            except ValueError:
                print(f"❌ Invalid value for {name}. Keeping old value.")
                new_params[name] = old_val
    try:
        train.edit_hyperparams(new_params)
        print("✅ Hyperparameters updated.")
    except Exception as e:
        print(f"❌ Failed to update hyperparameters: {str(e)}")


def train_model(train: Train):
    print_header("🚀 Starting Training")

    model_class = train.model.__class__.__name__
    if model_class == "kNN":
        print("⚠️  kNN does not require training. You may proceed to evaluation.")
        return

    trainer = train.trainerManager.trainer
    try:
        class progressCallback:
            def emit(self, progress):
                print(f"🟣 Progress: {progress}%", end="\r")

        class lossCallback:
            def emit(self, loss):
                print(f"💥 Loss: {loss:.4f}", end="\r")

        class dummyQThread:
            def msleep(self, _):
                pass

        trainer.train(dummyQThread(), progressCallback(), lossCallback())
        print("\n✅ Training complete.")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")


def evaluate_model(train: Train):
    print_header("📈 Model Evaluation")
    try:
        score = train.trainerManager.trainer.evaluate()
        print(f"✅ Evaluation Result: {browseDict['eval matrice']} = {score}")
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")


def replace_model(train: Train):
    print_header("🔄 Replace Model")
    confirm = input("Are you sure? This will reset your model. (y/n): ").strip().lower()
    if confirm != "y":
        print("❌ Cancelled.")
        return
    try:
        train.trainerManager.trainer.model = getNewModel()
        init_model(train.prep.data, train.trainerManager.trainer.model)
        print("✅ Model successfully reset.")
    except Exception as e:
        print(f"❌ Could not replace model: {str(e)}")


def init_deploy(train):
    deploy = Deploy(train, 100)
    deploy_cli(deploy)

def visualize_model():
    vis = Visualizer()
    print("Visualization options:")
    print("  [1] Plot Loss vs Epoch")
    print("  [2] Plot Training vs Validation")
    print("  [q] return")

    choice = input("Choose one option: ")
    if choice == "q":
        return
    
    if choice not in ["1","2"]:
        print("Enter a valid option")
        visualize_model()
        return
    if choice == '1': 
        vis.plot_loss()
    else: 
        vis.plot_validation_vs_training()
        

def save_session(train):
    session_name = input("Enter a session name to save: ").strip()
    if session_name:
        try:
            train.save(session_name)
            print(f"✅ Session saved as '{session_name}_model.pt'")
        except Exception as e:
            print(f"❌ Failed to save session: {str(e)}")
    else:
        print("❌ Session name cannot be empty.")

def load_session(train):
    session_name = input("Enter a session name to load: ").strip()
    if session_name:
        try:
            import os
            filename = f"{session_name}_model.pt"
            if not os.path.exists(filename):
                print(f"❌ No saved session found with name '{session_name}'.")
                return
            train.load()
            print(f"✅ Session '{session_name}_model.pt' loaded.")
        except Exception as e:
            print(f"❌ Failed to load session: {str(e)}")
    else:
        print("❌ Session name cannot be empty.")


def get_unique_session_name():
    import os
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_sessions'))
    while True:
        session_name = input("Enter a name for this training session: ").strip()
        if not session_name:
            print("❌ Session name cannot be empty.")
            continue
        weights_path = os.path.join(save_dir, f'{session_name}_model.bin')
        json_path = os.path.join(save_dir, f'{session_name}_session.json')
        if os.path.exists(weights_path) or os.path.exists(json_path):
            print(f"❌ Session name '{session_name}' already exists. Please choose another name.")
        else:
            return session_name

def train_cli(train: Train):
    print_header("🧠 Bojai Training CLI")
    trained = False

    while True:
        display_hyperparams(train)
        print("\nAvailable actions:")
        print("  [t] Start training")
        print("  [u] Update hyperparameters")
        print("  [v] Visualize loss, validation, and training metrices")
        print("  [e] Evaluate model")
        print("  [r] Replace model")
        print("  [d] Deploy model (if trained)")
        print("  [s] Save session")
        print("  [l] Load session")
        print("  [p] ⬅️  Go back to data preparation")
        print("  [q] Quit")

        choice = input("Enter your choice: ").strip().lower()

        if choice == "t":
            train_model(train)
            trained = True
        elif choice == "u":
            update_hyperparams(train)
        elif choice == "e":
            evaluate_model(train)
        elif choice == "r":
            replace_model(train)
        elif choice == "d":
            if not trained:
                print("❌ Please train your model first before deploying.")
            else:
                init_deploy(train)
        elif choice == "s":
            session_name = get_unique_session_name()
            train.save(session_name)
            print(f"Session '{session_name}' saved.")
        elif choice == "l":
            save_first = input("Do you want to save the current session before loading? (y/n): ").strip().lower()
            if save_first == "y":
                session_name_save = get_unique_session_name()
                train.save(session_name_save)
                print(f"Session '{session_name_save}' saved.")
            session_name = input("Enter the name of the session to load: ").strip()
            train.load(session_name)
            print(f"Session '{session_name}' loaded.")
        elif choice == "p":
            from prepareCLI import prepare_cli

            print("🔁 Returning to data preparation CLI...")
            prepare_cli(train.prep)
            print("✅ Returned to training stage.")
        elif choice == "v":
            visualize_model()
        elif choice == "q":
            save_before_exit = input("Do you want to save the current training session before quitting? (y/n): ").strip().lower()
            if save_before_exit == "y":
                session_name = get_unique_session_name()
                train.save(session_name)
                print(f"Session '{session_name}' saved.")
            print("👋 Exiting training stage.")
            sys.exit(0)
        else:
            print("❌ Invalid choice.")


if __name__ == "__main__":
    from prepare import Prepare
    from train import Train

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
    train_cli(train)
