# BojAI — Build, Train, and Deploy ML Models Without Writing Boilerplate

**BojAI** is an open-source framework that transforms how machine learning is built, trained, and deployed.

Whether you're an ML researcher, engineer, or educator, BojAI gives you:
- 🔁 A fully modular pipeline
- 🧱 A UI *and* CLI to preprocess, train, and deploy
- 📦 Zero-boilerplate support for both built-in and custom models
- 🧠 Designed for experimentation, education, and production

Instead of spending time wiring data processors, models, and training loops together, BojAI lets you focus purely on building your model's logic and training strategy.

---

## Features

- ✅ **Pre-built training & deployment workflows** via CLI and GUI
- ✅ **Custom model support** with plug-and-play modules
- ✅ **Abstracted processor/trainer interfaces** for complete flexibility
- ✅ **Dataset division, hyperparameter control, and live metrics**
- ✅ **Auto UI updates via PyQt5 interface**
- ✅ **Secure deployment-ready model saving and loading**

---

## Installation

Install from source:

```bash
git clone https://github.com/saughmon/bojai.git
cd bojai
pip install .
```

---

##  Getting Started

You can use BojAI in two main ways:

---

### 1. **Use a Built-In Model**

Start a full ML pipeline from UI or CLI without writing any code:

1. Run `bojai list --pipelines` to se what pipelines are available. To learn more about each pipeline, visit its page on our docs site. 

2. Run `bojai build --pipeline chosen-pipeline-name` 

3. Run `bojai start --pipeline built-pipeline-name` to use the pipeline in CLI mode, or `bojai start --pipeline built-pipeline-name --ui` to use it in UI mode. 

### 2. **Create Your Own Custom Model or Pipeline**

1. Run `bojai create --pipeline give-your-pipeline-a-name --directory where/you/want/to/save` to copy files into a folder and save it in the specified directory. 

2. This folder contains fiels where you can implement your data processing logic, model, training and eval loop, and usage logic. Detailed instuctions are each file. Implement them before you move forward. 

2. Run `bojai build --pipeline give-your-pipeline-a-name --directory path/to/directory/with/implementation`  

3. Run `bojai start --pipeline built-pipeline-name` to use the pipeline in CLI mode, or `bojai start --pipeline built-pipeline-name --ui` to use it in UI mode. 


---

## Full documentation

Full docs available at:  
  [https://bojai.org](https://bojai.org)

## Ethical Use

We recognize the power of machine learning and aim to make it more accessible through BojAI. With this accessibility comes the responsibility to use such tools for constructive and ethical purposes. We explicitly discourage the use of BojAI in any context related to:

- Military or warfare applications

- Violations of human rights

- Activities that contribute negatively to climate change

- Animal exploitation or abuse

In short: do not use BojAI to harm people, animals, or the planet.

Users found to be engaging in harmful activities may be restricted from accessing future updates or contributions.


---

## Developed by

**Saughmon Boujkian**  
Undergraduate student of Computer Science at the University of British Columbia  
> “BojAI isn’t just a framework — it’s a declaration that ML should be *clean*, *modular*, and *accessible*.”

---

## License

MIT License — free to use, fork, and improve.

---

## Show Your Support

If BojAI helps you, **star the repo** and share it!  
It means the world to devs and researchers like me.
