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
        # mimics the behavior of PyQt's signal-based threading system in a headless CLI context
        class progressCallback:
            def emit(self, progress):
                print(f"🟣 Progress: {progress}%", end="\r")

        class lossCallback:
            def emit(self, loss):
                print(f"💥 Loss: {loss:.4f}", end="\r")

        # Dummy QThread object with just .msleep()
        class dummyQThread:
            def msleep(self, _):
                pass  # No need to sleep in CLI context

        trainer.train(dummyQThread(), progressCallback(), lossCallback())
        print("\n✅ Training complete.")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")


def evaluate_model(train: Train):
    print_header("📈 Model Evaluation")
    try:
        score = train.trainerManager.trainer.evaluate()
        print(f"✅ Evaluation Result: {browseDict['eval_matrice']} = {score}")
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


def train_cli(train: Train):
    print_header("🧠 Bojai Training CLI")
    trained = False

    while True:
        display_hyperparams(train)
        print("\nAvailable actions:")
        print("  [t] Start training")
        print("  [u] Update hyperparameters")
        print("  [e] Evaluate model")
        print("  [r] Replace model")
        print("  [d] Deploy model (if trained)")
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
        elif choice == "p":
            from prepareCLI import prepare_cli

            print("🔁 Returning to data preparation CLI...")
            prepare_cli(train.prep)
            print("✅ Returned to training stage.")
        elif choice == "q":
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
        "",
    )
    hyperparams = hyper_params
    train = Train(prep, hyperparams)
    train_cli(train)
