import argparse
import os
import torch
from prepare import Prepare
from global_vars import browseDict, getNewModel, getNewTokenizer, task_type, options
from prepareCLI import prepare_cli


def prompt_input(prompt_text, cast_type=str, allow_blank=False):
    while True:
        value = input(prompt_text).strip()
        if not value and not allow_blank:
            print("❌ Input cannot be blank.")
            continue
        try:
            return cast_type(value)
        except ValueError:
            print(f"❌ Invalid input. Please enter a {cast_type.__name__}.")


def data_preparation_stage_cli(prep):
    print("\n────────────────────────────────────")
    print("📁 Entering Data Preparation Stage")
    print("────────────────────────────────────")
    prepare_cli(prep)


def initialize_pipeline_cli():
    print("📦 Welcome to Bojai CLI Initializer")
    print("────────────────────────────────────\n")

    model_name = prompt_input("Enter Model Name: ")

    # Path selection
    data_address = prompt_input("Enter Data Address (file or directory): ")
    if not os.path.exists(data_address):
        print("❌ Path does not exist. Exiting.")
        return

    # Training / evaluation split
    print("\n⚖️  Set the train/eval split (must add to 1)")
    train_split = prompt_input("Training split (e.g., 0.8): ", float)
    eval_split = prompt_input("Evaluation split (e.g., 0.2): ", float)

    if round(train_split + eval_split, 2) != 1.0:
        print("❌ Train and eval splits must add to 1.0. Exiting.")
        return

    # Tokenizer selection if needed
    architecture = None
    if browseDict["options"] == 1:
        print("\n🧠 Choose a tokenizer architecture:")
        arch_names = list(options.keys())
        for idx, name in enumerate(arch_names):
            print(f"{idx + 1}. {name}")
        selected = prompt_input("Enter number of architecture: ", int)
        if selected < 1 or selected > len(arch_names):
            print("❌ Invalid architecture number. Exiting.")
            return
        architecture = arch_names[selected - 1]
    
    # Agent use
    print("This pipeline is built to process data in a very specific way.")
    print("If your data is stored differently, this pipeline won't work as-is.")
    print("We offer an agent that can adapt the pipeline to your data — you just need to describe how your data is stored.")
    agent_use = prompt_input("Would you like to use the agent? [Y/n] ").strip().lower()

    while agent_use not in ("y", "n"):
        print("Error: enter 'Y' for yes or 'n' for no.")
        agent_use = prompt_input("Would you like to use the agent? [Y/n] ").strip().lower()

    agent_use = agent_use == "y"
    image_use = False
    description = ""

    if agent_use:
        print("This is a beta feature and still under testing.")
        print("To use the agent, you need Ollama installed on your computer.")
        print("Download it here: https://ollama.com/download/windows")

        change_mind = prompt_input("If you change your mind, enter 0. Press Enter to continue: ", allow_blank=True).strip()
        if change_mind == "0":
            agent_use = False
        else:
            # Ask about image use
            image_use = prompt_input("Does your data contain images? [Y/n] ").strip().lower()
            while image_use not in ("y", "n", "0"):
                print("Error: enter 'Y' for yes, 'n' for no, or '0' to cancel.")
                image_use = prompt_input("Does your data contain images? [Y/n] ").strip().lower()
            if image_use == "0":
                agent_use = False
            else:
                image_use = image_use == "y"

        # Ask for data description
        if agent_use:
            description = prompt_input(
                "Enter a thorough description of how your data is stored.\n"
                "Refer to our docs site for guidelines.\n"
                "Say something like: 'My data folder contains a .txt file where each line has input and output separated by a comma.'\n"
                "Or: 'Each subfolder contains one image and a label file.'\n"
                "Enter 0 to cancel: "
            ).strip()
            if description == "0":
                agent_use = False

    # Build model/tokenizer based on options
    try:
        if browseDict["options-where"] == 0:
            model = getNewModel()
            tokenizer = options[architecture]() if architecture else None
        elif browseDict["options-where"] == 1:
            model = options[architecture]() if architecture else None
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
            (train_split, eval_split),
            ",",
            use_agent=[agent_use, description, image_use]
        )
    except Exception as e:
        print(f"❌ Error initializing pipeline: {str(e)}")
        return

    # Print success message and configuration info
    print("\n✅ Pipeline successfully initialized!")
    print("────────────────────────────────────")
    print(f"Model Name       : {model_name}")
    print(f"Data Path        : {data_address}")
    print(f"Train/Eval Split : {train_split}/{eval_split}")
    print(f"Tokenizer        : {architecture if architecture else 'Default'}")
    print("────────────────────────────────────\n")

    # Offer option to move to the data preparation stage
    choice = (
        input("Type 'p' to proceed with data preparation, or any other key to exit: ")
        .strip()
        .lower()
    )
    if choice == "p":
        data_preparation_stage_cli(prep)
    else:
        print("Exiting Bojai CLI. Goodbye!")


if __name__ == "__main__":
    initialize_pipeline_cli()
