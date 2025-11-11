import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
import cv2
import numpy as np
import os
from model import ANPRModel
import easyocr 


class ANPRGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ANPR Number Plate Recognition")
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_path = None
        self.img_panel = None
        self.target_size = (224, 224)

        self.root.minsize(900, 600)
        self.root.configure(bg="#f0f4f8")

        self.main_frame = tk.Frame(root, bg="#f0f4f8")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_panel = tk.Frame(self.main_frame, bg="#e3eaf2", bd=2, relief=tk.RIDGE)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

        self.right_panel = tk.Frame(self.main_frame, bg="#f0f4f8", bd=2, relief=tk.RIDGE)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.title_label = tk.Label(self.left_panel, text="ANPR System", font=("Segoe UI", 18, "bold"), bg="#e3eaf2")
        self.title_label.pack(pady=(10, 20))

        self.load_model_btn = tk.Button(
            self.left_panel, text="Load Model (Ctrl+L)", command=self.load_model,
            font=("Segoe UI", 12), bg="#c7d6ee", fg="#222", relief=tk.GROOVE, height=2
        )
        self.load_model_btn.pack(pady=5, anchor="n", fill=tk.X)

        self.choose_img_btn = tk.Button(
            self.left_panel, text="Choose Image (Ctrl+O)", command=self.choose_image, state=tk.DISABLED,
            font=("Segoe UI", 12), bg="#c7d6ee", fg="#222", relief=tk.GROOVE, height=2
        )
        self.choose_img_btn.pack(pady=5, anchor="n", fill=tk.X)

        self.detect_btn = tk.Button(
            self.left_panel, text="Detect Number Plate (Ctrl+D)", command=self.detect_plate, state=tk.DISABLED,
            font=("Segoe UI", 12), bg="#c7d6ee", fg="#222", relief=tk.GROOVE, height=2
        )
        self.detect_btn.pack(pady=5, anchor="n", fill=tk.X)

        self.ocr_label = tk.Label(self.left_panel, text="Detected Plate Number:", font=("Segoe UI", 14), bg="#e3eaf2")
        self.ocr_label.pack(pady=(30, 5), anchor="nw")
        self.ocr_text = tk.Text(self.left_panel, height=2, width=20, font=("Consolas", 18, "bold"), bg="#f8fafc", fg="#d7263d", bd=2, relief=tk.SUNKEN)
        self.ocr_text.pack(pady=5, anchor="nw")

        self.root.bind('<Control-l>', lambda event: self.load_model())
        self.root.bind('<Control-L>', lambda event: self.load_model())
        self.root.bind('<Control-o>', lambda event: self.choose_image())
        self.root.bind('<Control-O>', lambda event: self.choose_image())
        self.root.bind('<Control-d>', lambda event: self.detect_plate())
        self.root.bind('<Control-D>', lambda event: self.detect_plate())

        self.img_panel = None

        self.ocr_reader = easyocr.Reader(['en'])

    def load_model(self):
        model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("PyTorch Model", "*.pth")])
        if model_path:
            self.model = ANPRModel()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.choose_img_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Model Loaded", f"Loaded model from {model_path}")

    def choose_image(self):
        self.img_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if self.img_path:
            self.show_image(self.img_path)
            self.detect_btn.config(state=tk.NORMAL)

    
    def show_image(self, img_path, bbox=None):
        img = Image.open(img_path).convert("RGB")

        if bbox is not None:
            draw = ImageDraw.Draw(img)
            x, y, w, h = bbox
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        img_display = img.copy()
        # Use LANCZOS instead of deprecated ANTIALIAS
        img_display.thumbnail((700, 500), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_display)

        if self.img_panel:
            self.img_panel.config(image=img_tk)
            self.img_panel.image = img_tk
        else:
            self.img_panel = tk.Label(self.right_panel, image=img_tk, bg="#f0f4f8")
            self.img_panel.image = img_tk
            self.img_panel.pack(pady=10, expand=True)

    def detect_plate(self):
        if not self.model or not self.img_path:
            return
        
        rel_path = os.path.relpath(self.img_path, start=os.getcwd())
        img = cv2.imread(rel_path)
        if img is None:
            pil_img = Image.open(self.img_path).convert("RGB")
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_resized = cv2.resize(img, self.target_size)
        img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            bbox = self.model(img_tensor).cpu().numpy().squeeze()
        print("Predicted bounding box:", bbox)
        self.show_image(self.img_path, bbox=bbox)

        pil_img = Image.open(self.img_path).convert("RGB")
        x, y, w, h = bbox
        # Ensure coordinates are correct for cropping
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)
        # Clamp coordinates to image bounds
        img_w, img_h = pil_img.size
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        plate_crop = pil_img.crop((x1, y1, x2, y2))
        plate_crop_np = np.array(plate_crop)

        ocr_results = self.ocr_reader.readtext(plate_crop_np)
        if ocr_results:
            ocr_text = max(ocr_results, key=lambda x: x[2])[1]
        else:
            ocr_text = ""

        self.ocr_text.delete(1.0, tk.END)
        self.ocr_text.insert(tk.END, ocr_text.strip())

if __name__ == "__main__":
    root = tk.Tk()
    app = ANPRGUI(root)
    root.mainloop()
