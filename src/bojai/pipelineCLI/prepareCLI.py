import torch
from prepare import Prepare
from global_vars import (
    browseDict,
    getNewModel,
    getNewTokenizer,
    task_type,
    hyper_params,
)
from train import Train
from trainCLI import train_cli
import sys


def print_header(title):
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)


def display_pipeline_info(prep: Prepare):
    print_header("📄 Pipeline Information")
    print(f"🔤 Model Name        : {prep.model_name}")
    print(f"🧠 Model Type        : {type(prep.model).__name__}")
    print(f"🔡 Tokenizer Type    : {type(prep.tokenizer).__name__}")
    print(f"📈 Data Points       : {prep.num_data_points}")
    print(f"✅ Prep Ready?       : {'Yes' if prep.prep_ready else 'No'}")


def view_tokenized_data(prep: Prepare):
    while True:
        index = input(
            "Enter index to view tokenized data (leave blank for a random data point, q to quit): "
        ).strip()
        if index == "q":
            break
        try:
            if index == "":
                print(prep.check_tokenized())
            elif index.isdigit():
                print(prep.check_tokenized(int(index) - 1))
            else:
                print("❌ Invalid input.")
        except IndexError:
            print("❌ Index too large.")
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def view_raw_data(prep: Prepare):
    while True:
        index = input(
            "Enter index to view raw data (leave blank for all, q to quit): "
        ).strip()
        if index == "q":
            break
        try:
            which = browseDict.get("type", 2)
            if index == "":
                output = prep.view_raw_data()
            elif index.isdigit():
                output = prep.view_raw_data(int(index) - 1)
            else:
                print("❌ Invalid input.")
                continue

            if which == 0:
                image, label = output
                print(f"[🖼️ Image Data]: {label}")
                image.show()
            else:
                print(f"[📃 Raw]: {output}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")


def update_data_path(prep: Prepare):
    new_path = input("Enter new dataset address: ").strip()
    if not new_path:
        print("❌ Path cannot be empty.")
        return
    try:
        prep.update_data(new_path)
        print(
            f"✅ Data updated successfully. Total data points: {prep.num_data_points}"
        )
    except Exception as e:
        print(f"❌ Failed to update data: {str(e)}")


def do_training(prep: Prepare):
    try:
        train = Train(prep, hyper_params)
        print("🧠 Launching training CLI...")
        train_cli(train)
    except Exception as e:
        print(f"❌ Failed to launch training: {str(e)}")


def prepare_cli(prep: Prepare):
    print_header("🔧 Bojai Data Preparation CLI")
    while True:
        display_pipeline_info(prep)

        print("\nAvailable actions:")
        print("  [v] View tokenized data")
        print("  [r] View raw data")
        print("  [u] Update dataset path")
        print("  [t] Continue to training")
        print("  [q] Quit")

        choice = input("Enter your choice: ").strip().lower()

        if choice == "v":
            view_tokenized_data(prep)
        elif choice == "r":
            view_raw_data(prep)
        elif choice == "u":
            update_data_path(prep)
        elif choice == "t":
            if not prep.prep_ready:
                print("❌ Prep is not ready. Cannot continue to training.")
                continue
            print("✅ Prep complete. You can now continue to training stage.")
            return do_training(prep)
        elif choice == "q":
            print("👋 Exiting data preparation stage.")
            sys.exit(0)
        else:
            print("❌ Invalid option. Try again.")


# Example usage entry point
if __name__ == "__main__":
    model_name = input("Enter model name: ")
    data_address = input("Enter dataset path: ")
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
        ",",
    )

    prepare_cli(prep)
