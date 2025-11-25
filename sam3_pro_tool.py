"""
SAM 3 Professional Annotation Tool
Enhanced UI with Text, Box, and Point modes
"""

import matplotlib
matplotlib.use('Agg')

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import torch
import numpy as np
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import os

class SAM3ProTool:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM 3 Professional Annotation Tool")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')
        
        # State
        self.model = None
        self.processor = None
        self.current_image = None
        self.display_image = None
        self.photo = None
        self.masks_data = None
        
        # Mode and interaction state
        self.mode = tk.StringVar(value="text")
        self.points = []  # [(x, y, is_positive), ...]
        self.box_start = None
        self.current_box = None
        
        # Image display scaling info
        self.img_scale = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0
        self.displayed_size = (0, 0)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#2d2d2d', height=120)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)
        
        tk.Label(header, text="🎯 SAM 3 Annotation Tool", 
                font=('Segoe UI', 28, 'bold'),
                bg='#2d2d2d', fg='#00d4ff').pack(pady=10)
        
        tk.Label(header, text="Segment Anything Model 3 - Professional Edition", 
                font=('Segoe UI', 11),
                bg='#2d2d2d', fg='#888888').pack()
        
        # Main container
        main = tk.Frame(self.root, bg='#1a1a1a')
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main, bg='#252525', width=350, relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        self.setup_controls(left_panel)
        
        # Right panel - Canvas
        right_panel = tk.Frame(main, bg='#2b2b2b', relief=tk.SUNKEN, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(right_panel, bg='#2b2b2b', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.canvas_click)
        self.canvas.bind('<B1-Motion>', self.canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self.canvas_release)
        
        # Status bar
        status_bar = tk.Frame(self.root, bg='#2d2d2d', height=30)
        status_bar.pack(fill=tk.X)
        status_bar.pack_propagate(False)
        
        self.status = tk.Label(status_bar, text="Ready - Load an image to start", 
                              font=('Segoe UI', 10),
                              bg='#2d2d2d', fg='#00ff00', anchor=tk.W)
        self.status.pack(fill=tk.X, padx=10)
        
    def setup_controls(self, parent):
        # File Operations
        file_frame = tk.LabelFrame(parent, text="📁 File Operations", 
                                  font=('Segoe UI', 11, 'bold'),
                                  bg='#252525', fg='white', bd=2)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(file_frame, text="📂 Load Image", command=self.load_image,
                 font=('Segoe UI', 11), bg='#4CAF50', fg='white',
                 padx=20, pady=8, cursor='hand2').pack(fill=tk.X, padx=10, pady=5)
        
        self.save_btn = tk.Button(file_frame, text="💾 Save Masks", command=self.save,
                                  font=('Segoe UI', 11), bg='#FF9800', fg='white',
                                  padx=20, pady=8, state=tk.DISABLED, cursor='hand2')
        self.save_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Mode Selection
        mode_frame = tk.LabelFrame(parent, text="🎨 Annotation Mode", 
                                  font=('Segoe UI', 11, 'bold'),
                                  bg='#252525', fg='white', bd=2)
        mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        modes = [
            ("📝 Text Prompt", "text"),
            ("📦 Bounding Box", "box"),
            ("👆 Point Clicks", "points")
        ]
        
        for text, value in modes:
            tk.Radiobutton(mode_frame, text=text, variable=self.mode, value=value,
                          font=('Segoe UI', 10), bg='#252525', fg='white',
                          selectcolor='#333333', activebackground='#252525',
                          activeforeground='white', command=self.mode_changed).pack(anchor=tk.W, padx=20, pady=3)
        
        # Text Prompt Frame
        self.text_frame = tk.LabelFrame(parent, text="✏️ Text Prompt", 
                                       font=('Segoe UI', 11, 'bold'),
                                       bg='#252525', fg='white', bd=2)
        self.text_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.prompt_var = tk.StringVar(value="tooth")
        tk.Entry(self.text_frame, textvariable=self.prompt_var,
                font=('Segoe UI', 11), bg='#333333', fg='white',
                insertbackground='white').pack(fill=tk.X, padx=10, pady=10)
        
        # Point Mode Info
        self.point_frame = tk.LabelFrame(parent, text="👆 Point Mode", 
                                        font=('Segoe UI', 11, 'bold'),
                                        bg='#252525', fg='white', bd=2)
        
        tk.Label(self.point_frame, 
                text="🟢 Left Click: Positive point\n🔴 Right Click: Negative point\n(Feature in development)",
                font=('Segoe UI', 9), bg='#252525', fg='#cccccc',
                justify=tk.LEFT).pack(padx=10, pady=10)
        
        # Box Mode Info
        self.box_frame = tk.LabelFrame(parent, text="📦 Box Mode", 
                                      font=('Segoe UI', 11, 'bold'),
                                      bg='#252525', fg='white', bd=2)
        
        tk.Label(self.box_frame, 
                text="Click and drag to draw box\n(Feature in development)",
                font=('Segoe UI', 9), bg='#252525', fg='#cccccc',
                justify=tk.LEFT).pack(padx=10, pady=10)
        
        # Actions
        action_frame = tk.LabelFrame(parent, text="⚡ Actions", 
                                    font=('Segoe UI', 11, 'bold'),
                                    bg='#252525', fg='white', bd=2)
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.seg_btn = tk.Button(action_frame, text="🚀 Run Segmentation", 
                                command=self.segment,
                                font=('Segoe UI', 11, 'bold'), bg='#2196F3', fg='white',
                                padx=20, pady=10, state=tk.DISABLED, cursor='hand2')
        self.seg_btn.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(action_frame, text="🗑️ Clear All", command=self.clear_all,
                 font=('Segoe UI', 11), bg='#f44336', fg='white',
                 padx=20, pady=8, cursor='hand2').pack(fill=tk.X, padx=10, pady=5)
        
        # Info panel
        info_frame = tk.LabelFrame(parent, text="ℹ️ Information", 
                                  font=('Segoe UI', 11, 'bold'),
                                  bg='#252525', fg='white', bd=2)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.info_text = tk.Text(info_frame, font=('Consolas', 9),
                                bg='#1a1a1a', fg='#00ff00', wrap=tk.WORD,
                                height=8, relief=tk.FLAT)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.info_text.insert('1.0', "Ready to start!\n\n1. Load an image\n2. Select mode\n3. Run segmentation")
        self.info_text.config(state=tk.DISABLED)
        
    def mode_changed(self):
        # Hide all mode-specific frames
        self.text_frame.pack_forget()
        self.point_frame.pack_forget()
        self.box_frame.pack_forget()
        
        # Show relevant frame
        if self.mode.get() == "text":
            self.text_frame.pack(fill=tk.X, padx=10, pady=10, before=self.text_frame.master.winfo_children()[-2])
        elif self.mode.get() == "points":
            self.point_frame.pack(fill=tk.X, padx=10, pady=10, before=self.text_frame.master.winfo_children()[-2])
        elif self.mode.get() == "box":
            self.box_frame.pack(fill=tk.X, padx=10, pady=10, before=self.text_frame.master.winfo_children()[-2])
        
        self.clear_annotations()
        
    def clear_annotations(self):
        self.points = []
        self.box_start = None
        self.current_box = None
        if self.current_image:
            self.show_image()
        
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        
        if not path:
            return
            
        try:
            self.current_image = Image.open(path).convert("RGB")
            self.display_image = self.current_image.copy()
            self.show_image()
            self.seg_btn.config(state=tk.NORMAL)
            self.status.config(text=f"✓ Loaded: {os.path.basename(path)}")
            self.update_info(f"Image loaded: {os.path.basename(path)}\nSize: {self.current_image.size}\n\nSelect mode and run segmentation")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def show_image(self):
        if self.current_image is None:
            return
            
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        if w < 10:
            w, h = 1000, 700
        
        # Create display image with annotations
        display = self.display_image.copy()
        draw = ImageDraw.Draw(display)
        
        # Draw points (in image coordinates)
        for x, y, is_pos in self.points:
            color = 'green' if is_pos else 'red'
            r = 8
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline='white', width=2)
        
        # Draw current box (normalize coordinates first)
        if self.current_box:
            x0, y0, x1, y1 = self.current_box
            # Ensure x0 < x1 and y0 < y1
            box_normalized = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            draw.rectangle(box_normalized, outline='cyan', width=3)
        
        # Calculate scaling
        img_copy = display.copy()
        orig_w, orig_h = img_copy.size
        img_copy.thumbnail((w-20, h-20), Image.Resampling.LANCZOS)
        new_w, new_h = img_copy.size
        
        # Store scale and offset for coordinate mapping
        self.img_scale = new_w / orig_w
        self.img_offset_x = (w - new_w) // 2
        self.img_offset_y = (h - new_h) // 2
        self.displayed_size = (new_w, new_h)
        
        self.photo = ImageTk.PhotoImage(img_copy)
        self.canvas.delete("all")
        self.canvas.create_image(w//2, h//2, image=self.photo, anchor=tk.CENTER)
        
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Map canvas coordinates to image coordinates"""
        # Subtract offset to get coords relative to image
        img_x = (canvas_x - self.img_offset_x) / self.img_scale
        img_y = (canvas_y - self.img_offset_y) / self.img_scale
        
        # Clamp to image bounds
        orig_w, orig_h = self.current_image.size
        img_x = max(0, min(orig_w - 1, img_x))
        img_y = max(0, min(orig_h - 1, img_y))
        
        return int(img_x), int(img_y)
    
    def canvas_click(self, event):
        if self.current_image is None:
            return
        
        # Convert to image coordinates
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        if self.mode.get() == "points":
            # Add point (left click = positive)
            self.points.append((img_x, img_y, True))
            self.show_image()
            self.update_info(f"Added point at image coords: ({img_x}, {img_y})\nTotal points: {len(self.points)}")
            
        elif self.mode.get() == "box":
            # Start box (store in image coordinates)
            self.box_start = (img_x, img_y)
            
    def canvas_drag(self, event):
        if self.mode.get() == "box" and self.box_start:
            # Convert to image coordinates
            img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
            self.current_box = [self.box_start[0], self.box_start[1], img_x, img_y]
            self.show_image()
            
    def canvas_release(self, event):
        if self.mode.get() == "box" and self.box_start:
            # Convert to image coordinates
            img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
            self.current_box = [self.box_start[0], self.box_start[1], img_x, img_y]
            self.show_image()
            
            # Normalize box coordinates
            x0, y0, x1, y1 = self.current_box
            normalized = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            self.update_info(f"Box drawn at image coords:\n({normalized[0]}, {normalized[1]}) to ({normalized[2]}, {normalized[3]})")
            
    def init_model(self):
        if self.model is None:
            self.status.config(text="Loading SAM 3 model...")
            self.update_info("Loading SAM 3 model (3.45 GB)...\nPlease wait...")
            self.root.update()
            
            self.model = build_sam3_image_model(
                checkpoint_path=r"D:/CDAC/Playground/Sam_3/Models_saved/sam3.pt"
            )
            self.processor = Sam3Processor(self.model)
            self.status.config(text="Model loaded ✓")
            self.update_info("Model loaded successfully!")
            
    def segment(self):
        if self.current_image is None:
            return
        
        mode = self.mode.get()
        
        if mode == "text":
            self.segment_text()
        elif mode == "points":
            messagebox.showinfo("Info", "Point mode API integration coming soon!\nPlease use Text mode for now.")
        elif mode == "box":
            messagebox.showinfo("Info", "Box mode API integration coming soon!\nPlease use Text mode for now.")
            
    def segment_text(self):
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Warning", "Enter a text prompt!")
            return
            
        try:
            self.init_model()
            self.status.config(text=f"Segmenting '{prompt}'...")
            self.update_info(f"Running segmentation...\nPrompt: '{prompt}'")
            self.root.update()
            
            state = self.processor.set_image(self.current_image)
            output = self.processor.set_text_prompt(state=state, prompt=prompt)
            
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            self.masks_data = {"masks": masks, "boxes": boxes, "scores": scores}
            
            # Draw results
            result = self.current_image.copy()
            draw = ImageDraw.Draw(result, 'RGBA')
            
            num = len(masks)
            
            if num > 0:
                for i in range(num):
                    mask = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]
                    while mask.ndim > 2:
                        mask = mask[0]
                    
                    box = boxes[i].cpu().numpy() if torch.is_tensor(boxes[i]) else boxes[i]
                    score = scores[i].item() if torch.is_tensor(scores[i]) else scores[i]
                    
                    color = tuple(np.random.randint(50, 255, 3).tolist())
                    
                    mask_img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode='L')
                    if mask_img.size != result.size:
                        mask_img = mask_img.resize(result.size, Image.Resampling.NEAREST)
                    
                    overlay = Image.new('RGBA', result.size, color + (80,))
                    result.paste(overlay, (0, 0), mask_img)
                    
                    x0, y0, x1, y1 = map(int, box)
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                    draw.text((x0, y0-15), f"#{i+1}: {score:.2f}", fill=color)
                
                self.display_image = result.convert('RGB')
                self.show_image()
                self.save_btn.config(state=tk.NORMAL)
                
                info = f"✓ Segmentation complete!\n\nFound: {num} object(s)\n"
                for i, s in enumerate(scores):
                    info += f"Object #{i+1}: {s.item():.3f}\n"
                self.update_info(info)
                self.status.config(text=f"✓ Found {num} object(s)!")
            else:
                self.update_info("No objects detected.\nTry a different prompt.")
                self.status.config(text="No objects found")
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()
            
    def save(self):
        if self.masks_data is None:
            return
            
        folder = filedialog.askdirectory()
        if not folder:
            return
            
        try:
            masks = self.masks_data["masks"]
            
            self.display_image.save(os.path.join(folder, "result.png"))
            
            for i, mask in enumerate(masks):
                mask = mask.cpu().numpy() if torch.is_tensor(mask) else mask
                while mask.ndim > 2:
                    mask = mask[0]
                
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img.save(os.path.join(folder, f"mask_{i+1}.png"))
            
            messagebox.showinfo("Success", f"Saved {len(masks)} masks!")
            self.status.config(text=f"✓ Saved to {folder}")
            self.update_info(f"Saved {len(masks)} masks to:\n{folder}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def clear_all(self):
        self.points = []
        self.box_start = None
        self.current_box = None
        self.masks_data = None
        
        if self.current_image:
            self.display_image = self.current_image.copy()
            self.show_image()
            
        self.save_btn.config(state=tk.DISABLED)
        self.update_info("Cleared all annotations")
        self.status.config(text="Cleared ✓")
        
    def update_info(self, text):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', text)
        self.info_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = SAM3ProTool(root)
    root.mainloop()
