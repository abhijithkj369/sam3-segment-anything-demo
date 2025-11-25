"""
Gradio Web Interface for SAM 3 - Segment Anything Model 3
Supports text prompts, point clicks, and bounding box annotations
"""

import gradio as gr
import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Global model instance
model = None
processor = None

def initialize_model():
    """Load SAM 3 model once at startup"""
    global model, processor
    if model is None:
        print("Loading SAM 3 model...")
        model = build_sam3_image_model(
            checkpoint_path=r"D:/CDAC/Playground/Sam_3/Models_saved/sam3.pt"
        )
        processor = Sam3Processor(model)
        print("Model loaded successfully!")

def overlay_masks_on_image(image, masks, boxes=None, scores=None):
    """Create visualization with colored masks overlaid on image"""
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
   
    # Draw each mask
    num_masks = len(masks)
    if num_masks == 0:
        ax.text(0.5, 0.5, 'No objects detected', 
                transform=ax.transAxes, ha='center',
                fontsize=20, color='red', weight='bold')
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, num_masks))
        
        for i, mask in enumerate(masks):
            # Convert to numpy
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
            
            # Show mask with transparency
            color = colors[i][:3]
            h, w = mask_np.shape[-2:]
            mask_image = mask_np.reshape(h, w, 1) * np.array([*color, 0.4]).reshape(1, 1, -1)
            ax.imshow(mask_image)
            
            # Draw bounding box if provided
            if boxes is not None and i < len(boxes):
                box = boxes[i].cpu().numpy() if torch.is_tensor(boxes[i]) else boxes[i]
                x0, y0, x1, y1 = box
                rect = patches.Rectangle(
                    (x0, y0), x1-x0, y1-y0,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add score if provided
                if scores is not None and i < len(scores):
                    score = scores[i].item() if torch.is_tensor(scores[i]) else scores[i]
                    ax.text(x0, y0-5, f'#{i+1}: {score:.2f}',
                            bbox=dict(facecolor=color, alpha=0.7),
                            fontsize=10, color='white', weight='bold')
    
    ax.set_title(f'Detected {num_masks} objects', fontsize=16, weight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return Image.fromarray(img_array)

def segment_with_text(image, text_prompt):
    """Segment image using text prompt"""
    if image is None:
        return None, "Please upload an image first!"
    
    if not text_prompt or text_prompt.strip() == "":
        return None, "Please enter a text prompt!"
    
    initialize_model()
    
    try:
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Process with SAM 3
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        # Create visualization
        result_image = overlay_masks_on_image(image, masks, boxes, scores)
        
        # Create info text
        info = f"✅ Found {len(masks)} objects\n"
        info += f"Prompt: '{text_prompt}'\n"
        if len(scores) > 0:
            info += f"Confidence scores: {[f'{s:.3f}' for s in scores.cpu().tolist()]}"
        
        return result_image, info
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def segment_with_points(image, points_json):
    """Segment image using point clicks"""
    if image is None:
        return None, "Please upload an image first!"
    
    initialize_model()
    
    try:
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Parse points (this would need to be implemented based on Gradio's interface)
        # For now, return a placeholder
        return image, "⚠️ Point annotation feature coming soon! Use text prompt mode for now."
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def segment_with_box(image, box_coords):
    """Segment image using bounding box"""
    if image is None:
        return None, "Please upload an image first!"
    
    initialize_model()
    
    try:
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # This would need box coordinates from the interface
        return image, "⚠️ Bounding box feature coming soon! Use text prompt mode for now."
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Create Gradio Interface
with gr.Blocks(title="SAM 3 Annotation Tool") as demo:
    gr.Markdown("""
    # 🎯 SAM 3 - Segment Anything Model 3
    ### Interactive Image Annotation Tool
    Upload an image and segment objects using **text prompts**, **point clicks**, or **bounding boxes**.
    """)
    
    with gr.Tabs():
        # Tab 1: Text Prompt Mode
        with gr.Tab("📝 Text Prompt"):
            with gr.Row():
                with gr.Column():
                    text_input_image = gr.Image(label="Upload Image", type="numpy")
                    text_prompt_input = gr.Textbox(
                        label="Text Prompt",
                        placeholder="Enter what to segment (e.g., 'tooth', 'car', 'person')",
                        value="tooth"
                    )
                    text_segment_btn = gr.Button("🚀 Segment", variant="primary", size="lg")
                
                with gr.Column():
                    text_output_image = gr.Image(label="Segmentation Result")
                    text_info_output = gr.Textbox(label="Detection Info", lines=4)
            
            gr.Examples(
                examples=[
                    ["assets/images/truck.jpg", "truck"],
                    ["assets/images/groceries.jpg", "banana"],
                ],
                inputs=[text_input_image, text_prompt_input],
                label="Try these examples"
            )
            
            text_segment_btn.click(
                fn=segment_with_text,
                inputs=[text_input_image, text_prompt_input],
                outputs=[text_output_image, text_info_output]
            )
        
        # Tab 2: Point Click Mode
        with gr.Tab("👆 Point Clicks"):
            gr.Markdown("""
            ### Coming Soon!
            Click on the image to add points. Green = include, Red = exclude.
            
            This feature requires additional Gradio components and will be implemented in the next version.
            """)
            point_input_image = gr.Image(label="Upload Image")
            point_output = gr.Textbox(label="Status", value="Feature under development")
        
        # Tab 3: Bounding Box Mode
        with gr.Tab("📦 Bounding Box"):
            gr.Markdown("""
            ### Coming Soon!
            Draw a bounding box around the object you want to segment.
            
            This feature requires additional Gradio components and will be implemented in the next version.
            """)
            box_input_image = gr.Image(label="Upload Image")
            box_output = gr.Textbox(label="Status", value="Feature under development")
    
    gr.Markdown("""
    ---
    ### ℹ️ About SAM 3
    - **Model**: Segment Anything Model 3 by Meta AI
    - **Released**: November 2025
    - **Capabilities**: Open-vocabulary segmentation with text, visual, and geometry prompts
    - **Model Size**: 848M parameters, 3.45GB
    """)

if __name__ == "__main__":
    print("Starting SAM 3 Gradio Interface...")
    print("The model will load when you run your first segmentation.")
    demo.launch(
        server_name="127.0.0.1",  # Accessible from network
        server_port=8001,
        share=False,  # Set to True to create a public link
        show_error=True
    )
