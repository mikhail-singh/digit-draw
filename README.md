# Digit Draw

A simple Python application that lets you draw handwritten digits in a desktop GUI and uses a convolutional neural network (CNN) to recognise them.

## Project Structure

```
digitDraw/
├── model.py         # CNN model definition
├── train.py         # Training script (produces model.pt)
├── canvas_ui.py     # Tkinter-based drawing GUI component
├── app.py           # Launches the GUI
├── model.pt         # Saved trained weights (auto-generated)
└── data/            # MNIST dataset (auto-downloaded)
```

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/mikhail-singh/digitDraw.git
cd digitDraw
```

### 2. Install dependencies

Make sure you have Python 3.8+ installed, then:

```bash
pip install torch torchvision pillow
```

> *Optional: for nicer button styling in the GUI, also install* `ttkbootstrap`.


### 4. Run the drawing app

```bash
python app.py
```

A window will open with:

* A black canvas to draw white digits
* **Clear** and **Predict** buttons at the bottom

Draw a digit, click **Predict**, and see the model’s guess!
