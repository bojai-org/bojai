import os
import torch
from global_vars import (
    browseDict,
    getNewModel,
    getNewTokenizer,
    hyper_params,
    task_type,
)
from deploy import Deploy
from timing_utils import print_timings
import sys
from visualizer import Visualizer

def print_header(title):
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)


def update_eval_data_cli(deploy):
    print_header("📂 Add New Evaluation Data")
    path = input("Enter path to new evaluation data (file or directory): ").strip()
    if not os.path.exists(path):
        print("❌ Invalid path.")
        return
    try:
        deploy.update_eval_data(path)
        print("✅ New evaluation data added.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def evaluate_original_data(deploy):
    print_header("📈 Evaluate Model on Original Data")
    try:
        score = deploy.get_eval_score(0)
        print(f"✅ {browseDict['eval matrice']}: {score}")
    except Exception as e:
        print(f"❌ Failed to evaluate: {str(e)}")


def evaluate_new_data(deploy):
    if deploy.new_data is None:
        print("⚠️  No new evaluation data found.")
        return
    print_header("📈 Evaluate Model on New Data")
    try:
        score = deploy.get_eval_score(1)
        print(f"✅ {browseDict['eval matrice']}: {score}")
    except Exception as e:
        print(f"❌ Failed to evaluate: {str(e)}")


def use_model_cli(deploy):
    print_header("🧠 Use Model for Inference")
    text_input = input(f"{browseDict['use_model_text']}\n> ").strip()
    try:
        deploy.max_length = 50  # Optional
        output = deploy.use_model(text_input)
        print(f"✅ Output: {output}")
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")


def save_model_cli(deploy):
    print_header("💾 Save Model")
    path = input("Enter directory to save model: ").strip()
    if not os.path.isdir(path):
        print("❌ Path must be a valid directory.")
        return
    try:
        filename = deploy.trainer.model.__class__.__name__ + ".bin"
        save_path = os.path.join(path, filename)
        torch.save(deploy.trainer.model.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")

def visualize_model(deploy):
    vis = Visualizer()
    print("Visualization options:")
    print("  [1] Plot Loss vs Epoch")
    print("  [2] Plot Training vs Validation")
    eval_data = deploy.new_data != None
    if eval_data:
        print("  [3] Plot Evaluation vs Training")
        print("  [4] Plot Evaluation vs Validation")
    print("  [q] quite")
    choice = input("Choose one option: ")

    if choice == "q":
        return
    if choice not in ('1','2','3','4'):
        print("Enter a valid option")
        visualize_model()
        return
    if choice == '1': 
        vis.plot_loss()
    elif choice == '2': 
        vis.plot_validation_vs_training()
    elif choice == '3' and eval_data:
        vis.plot_train_vs_eval()
    elif choice == '4' and eval_data:
        vis.plot_valid_vs_eval()

def deploy_cli(deploy: Deploy):
    from trainCLI import train_cli
    from prepareCLI import prepare_cli

    print_header("🚀 Bojai Model Deployment CLI")
    while True:
        print("\nAvailable actions:")
        print("  [a] Add new evaluation data")
        print("  [e] Evaluate on original data")
        print("  [v] Visualize loss, validation, training, and evaluation metrices")
        if deploy.new_data:
            print("  [n] Evaluate on new data")
        print("  [u] Use model for inference")
        print("  [s] Save model")
        print("  [t] ⬅️  Go back to training")
        print("  [p] ⬅️  Go back to data preparation")
        print("  [q] Quit")

        choice = input("Enter your choice: ").strip().lower()

        if choice == "a":
            update_eval_data_cli(deploy)
        elif choice == "e":
            evaluate_original_data(deploy)
        elif choice == "n" and deploy.new_data:
            evaluate_new_data(deploy)
        elif choice == "u":
            use_model_cli(deploy)
        elif choice == "s":
            save_model_cli(deploy)
        elif choice == "v":
            visualize_model(deploy)
        elif choice == "t":
            print("🔁 Returning to training CLI...")
            train_cli(deploy.trainer)
            print("✅ Returned to deployment stage.")
        elif choice == "p":
            print("🔁 Returning to data preparation CLI...")
            prepare_cli(deploy.trainer.prep)
            print("✅ Returned to deployment stage.")
        elif choice == "q":
            print("👋 Exiting deployment CLI.")
            print_timings()
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
    deploy = Deploy(train, 100)
    deploy_cli(deploy)
