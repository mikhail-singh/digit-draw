import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from model import CNNClassifier


class CanvasUI:
    def __init__(self, model_path="model.pt"):
        # Load model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNClassifier().to(self.device)
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))
        self.model.eval()

        # Create the main window
        self.window = tk.Tk()
        self.window.title("Draw a Digit")
        self.window.configure(bg="#000000")

        # Configure ttk button style
        style = ttk.Style(self.window)
        style.theme_use('clam')
        style.configure('Accent.TButton',
                        font=('Arial', 12),
                        foreground='#FFFFFF',
                        background='#101010',
                        padding=10)

        # Button frame + Clear / Predict side by side
        btn_frame = tk.Frame(self.window, bg="#000000")
        btn_frame.pack(pady=10)

        # — Canvas area —
        self.canvas_size = 280
        self.canvas = tk.Canvas(self.window,
                                width=self.canvas_size,
                                height=self.canvas_size,
                                bg="#000000",
                                highlightthickness=0)
        self.canvas.pack(pady=(20, 10))

        # — Instruction label —
        self.label = tk.Label(self.window,
                              text="Draw a digit, then Predict",
                              font=("Arial", 14),
                              bg="#000000",
                              fg="#FFFFFF")
        self.label.pack(pady=(0, 10))

        # — Button frame at bottom —
        btn_frame = tk.Frame(self.window, bg="#000000")
        btn_frame.pack(side="bottom", pady=20)

        # — Clear button —
        self.clear_btn = ttk.Button(
            btn_frame,
            text="Clear",
            command=self.clear_canvas,
            style='Accent.TButton'
        )
        self.clear_btn.pack(side="left", padx=5)

        # — Predict button —
        self.predict_btn = ttk.Button(
            btn_frame,
            text="Predict",
            command=self.predict,
            style='Accent.TButton'
        )
        self.predict_btn.pack(side="left", padx=5)

        # Bind drawing events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.draw)

        # Backing PIL image
        self.image = Image.new(
            "L", (self.canvas_size, self.canvas_size), "black")
        self.draw_image = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None

    def on_button_press(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        r = 4
        x, y = event.x, event.y

        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    fill="white", width=r*2, capstyle=tk.ROUND, smooth=True)
            self.draw_image.line([self.last_x, self.last_y, x, y],
                                 fill="white", width=r*2)
        else:
            self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                    fill="white", outline="white")
            self.draw_image.ellipse([x-r, y-r, x+r, y+r],
                                    fill="white", outline="white")
        # update last positions
        self.last_x, self.last_y = x, y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.last_x = None
        self.last_y = None
        self.draw_image.rectangle(
            [0, 0, self.canvas_size, self.canvas_size], fill="black")
        self.label.config(text="Draw a digit and press 'Predict'")

    def preprocess(self):
        img = self.image.resize((28, 28))
        t = transforms.ToTensor()
        return t(img).unsqueeze(0).to(self.device)

    def predict(self):
        tensor = self.preprocess()
        with torch.no_grad():
            out = self.model(tensor)
            digit = out.argmax(dim=1).item()
        self.label.config(text=f"Predicted Digit: {digit}")

    def run(self):
        self.window.mainloop()
