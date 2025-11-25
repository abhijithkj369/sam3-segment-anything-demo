"""
SAM 3 Simple Desktop Tool - Ultra Simplified Version
"""

import matplotlib
matplotlib.use('Agg')

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import numpy as np
import cv2
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import os

class SimpleSAM3Tool:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM 3 Annotation Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        
        self.model = None
        self.processor = None
        self.current_image = None
        self.photo = None
        self.masks = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="SAM 3 Annotation Tool", 
                        font=('Arial', 24, 'bold'),
                        bg='#1e1e1e', fg='white')
        title.pack(pady=10)
        
        # Buttons frame
        btn_frame = tk.Frame(self.root, bg='#1e1e1e')
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Load Image", command=self.load_image,
                 font=('Arial', 14), bg='#4CAF50', fg='white',
                 padx=20, pady=10).pack(side=tk.LEFT, padx=5)
        
        tk.Label(btn_frame, text="Prompt:", font=('Arial', 14),
                bg='#1e1e1e', fg='white').pack(side=tk.LEFT, padx=5)
        
        self.prompt_var = tk.StringVar(value="tooth")
        self.prompt_entry = tk.Entry(btn_frame, textvariable=self.prompt_var,
                                     font=('Arial', 14), width=20)
        self.prompt_entry.pack(side=tk.LEFT, padx=5)
        
        self.segment_btn = tk.Button(btn_frame, text="Segment", 
                                     command=self.segment,
                                     font=('Arial', 14), bg='#2196F3', fg='white',
                                     padx=20, pady=10, state=tk.DISABLED)
        self.segment_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = tk.Button(btn_frame, text="Save Masks",
                                 command=self.save_masks,
                                 font=('Arial', 14), bg='#FF9800', fg='white',
                                 padx=20, pady=10, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Image canvas
        canvas_frame = tk.Frame(self.root, bg='#2b2b2b', relief=tk.SUNKEN, bd=2)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.canvas = tk.Canvas(canvas_frame, bg='#2b2b2b', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status = tk.Label(self.root, text="Ready - Click 'Load Image' to start",
                              font=('Arial', 12), bg='#1e1e1e', fg='#00ff00')
        self.status.pack(pady=5)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        
        if not file_path:
            return
            
        try:
            self.current_image = Image.open(file_path).convert("RGB")
            self.display_image(self.current_image)
            self.segment_btn.config(state=tk.NORMAL)
            self.status.config(text=f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            
    def display_image(self, image):
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 10:  # Canvas not yet sized
            canvas_width = 900
            canvas_height = 600
        
        # Resize image to fit canvas
        img_copy = image.copy()
        img_copy.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and keep reference
        self.photo = ImageTk.PhotoImage(img_copy)
        
        # Clear canvas and display
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width//2, canvas_height//2,
            image=self.photo,
            anchor=tk.CENTER
        )
        
    def init_model(self):
        if self.model is None:
            self.status.config(text="Loading SAM 3 model... Please wait...")
            self.root.update()
            
            self.model = build_sam3_image_model(
                checkpoint_path=r"D:/CDAC/Playground/Sam_3/Models_saved/sam3.pt"
            )
            self.processor = Sam3Processor(self.model)
            self.status.config(text="Model loaded!")
            
    def segment(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return
            
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Warning", "Enter a prompt!")
            return
            
        try:
            self.init_model()
            self.status.config(text=f"Segmenting '{prompt}'...")
            self.root.update()
            
            # Run SAM 3
            state = self.processor.set_image(self.current_image)
            output = self.processor.set_text_prompt(state=state, prompt=prompt)
            
            self.masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            # Create visualization
            img_array = np.array(self.current_image)
            
            if len(self.masks) > 0:
                # Overlay masks
                for i, mask in enumerate(self.masks):
                    # Get mask as numpy array and ensure correct shape
                    mask_np = mask.cpu().numpy()
                    
                    # Ensure mask is 2D
                    if mask_np.ndim > 2:
                        mask_np = mask_np.squeeze()
                    
                    # Resize mask to match image size if needed
                    if mask_np.shape != img_array.shape[:2]:
                        mask_np = cv2.resize(mask_np, (img_array.shape[1], img_array.shape[0]))
                    
                    # Create binary mask
                    binary_mask = (mask_np > 0.5).astype(np.uint8)
                    
                    # Generate random color
                    color = np.random.randint(50, 255, 3).tolist()
                    
                    # Create colored overlay
                    overlay = img_array.copy()
                    overlay[binary_mask == 1] = color
                    
                    # Blend with original
                    img_array = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)
                    
                    # Draw rectangle
                    box = boxes[i].cpu().numpy().astype(int)
                    cv2.rectangle(img_array, (box[0], box[1]), (box[2], box[3]), 
                                color.tolist(), 2)
                    
                    # Add text
                    score = scores[i].item()
                    text = f"#{i+1}: {score:.2f}"
                    cv2.putText(img_array, text, (box[0], box[1]-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                result_img = Image.fromarray(img_array)
                self.display_image(result_img)
                
                self.save_btn.config(state=tk.NORMAL)
                self.status.config(text=f"Found {len(self.masks)} object(s)!")
            else:
                self.status.config(text="No objects found. Try different prompt.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Segmentation failed:\n{e}")
            self.status.config(text="Segmentation failed")
            
    def save_masks(self):
        if not self.masks:
            return
            
        folder = filedialog.askdirectory(title="Select save folder")
        if not folder:
            return
            
        try:
            for i, mask in enumerate(self.masks):
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_np)
                mask_img.save(os.path.join(folder, f"mask_{i+1}.png"))
            
            messagebox.showinfo("Success", f"Saved {len(self.masks)} masks!")
            self.status.config(text=f"Saved {len(self.masks)} masks to {folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleSAM3Tool(root)
    root.mainloop()
