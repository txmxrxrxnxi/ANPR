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
        self.root.title("ANPR Number Plate Detection")
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_path = None
        self.img_panel = None
        self.target_size = (224, 224)  # Change if needed to match training

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.left_panel = tk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.right_panel = tk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.load_model_btn = tk.Button(self.left_panel, text="Load Model (Ctrl+L)", command=self.load_model)
        self.load_model_btn.pack(pady=5, anchor="n", fill=tk.X)
        self.choose_img_btn = tk.Button(self.left_panel, text="Choose Image (Ctrl+O)", command=self.choose_image, state=tk.DISABLED)
        self.choose_img_btn.pack(pady=5, anchor="n", fill=tk.X)
        self.detect_btn = tk.Button(self.left_panel, text="Detect Number Plate (Ctrl+D)", command=self.detect_plate, state=tk.DISABLED)
        self.detect_btn.pack(pady=5, anchor="n", fill=tk.X)

        self.root.bind('<Control-l>', lambda event: self.load_model())
        self.root.bind('<Control-L>', lambda event: self.load_model())
        self.root.bind('<Control-o>', lambda event: self.choose_image())
        self.root.bind('<Control-O>', lambda event: self.choose_image())
        self.root.bind('<Control-d>', lambda event: self.detect_plate())
        self.root.bind('<Control-D>', lambda event: self.detect_plate())

        self.img_panel = None

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

        img_tk = ImageTk.PhotoImage(img)

        if self.img_panel:
            self.img_panel.config(image=img_tk)
            self.img_panel.image = img_tk
        else:
            self.img_panel = tk.Label(self.root, image=img_tk)
            self.img_panel.image = img_tk
            self.img_panel.pack(pady=10)

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
        plate_crop = pil_img.crop((x, y, x - w, y - h))
        plate_crop_np = np.array(plate_crop)

        # OCR using EasyOCR
        ocr_results = self.ocr_reader.readtext(plate_crop_np)
        if ocr_results:
            # Take the text with the highest confidence
            ocr_text = max(ocr_results, key=lambda x: x[2])[1]
        else:
            ocr_text = ""

        self.ocr_text.delete(1.0, tk.END)
        self.ocr_text.insert(tk.END, ocr_text.strip())

if __name__ == "__main__":
    root = tk.Tk()
    app = ANPRGUI(root)
    root.mainloop()
