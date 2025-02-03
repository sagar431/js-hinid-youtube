import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
from models.classifier import CatDogClassifier
import os
import matplotlib.pyplot as plt
from utils import create_rich_progress, task_wrapper, setup_logger, logger

def process_image(image_path, transform):
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0), img

def visualize_prediction(img, predicted_label, confidence, save_path):
    """Visualize the image with prediction and save it"""
    # Use a default style instead of seaborn
    plt.style.use('default')
    
    # Create figure with larger size and better resolution
    fig = plt.figure(figsize=(10, 10), dpi=150)
    
    # Add image
    plt.imshow(img)
    plt.axis('off')
    
    # Create title with prediction and confidence
    title = f'Predicted: {predicted_label.capitalize()}\nConfidence: {confidence:.2f}'
    plt.title(title, fontsize=14, pad=20, fontweight='bold')
    
    # Add colored border based on confidence
    color = 'green' if confidence > 0.8 else 'orange' if confidence > 0.5 else 'red'
    for spine in plt.gca().spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(5)
    
    # Add confidence bar
    confidence_height = 0.05
    confidence_position = plt.gca().get_position()
    confidence_ax = fig.add_axes([
        confidence_position.x0,
        confidence_position.y0 - 2*confidence_height,
        confidence_position.width,
        confidence_height
    ])
    confidence_ax.barh(0, confidence, color=color)
    confidence_ax.barh(0, 1, color='lightgray', zorder=0)
    confidence_ax.set_xlim(0, 1)
    confidence_ax.set_ylim(-0.5, 0.5)
    confidence_ax.axis('off')
    
    # Set figure background to white
    fig.patch.set_facecolor('white')
    plt.gca().set_facecolor('white')
    
    # Save with tight layout
    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

def create_summary_grid(results, output_path):
    """Create a grid of all predictions"""
    n_images = len(results)
    if n_images == 0:
        return
    
    # Calculate grid dimensions
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
    for idx, (img_path, img, label, conf) in enumerate(results, 1):
        plt.subplot(n_rows, n_cols, idx)
        plt.imshow(img)
        plt.axis('off')
        color = 'green' if conf > 0.8 else 'orange' if conf > 0.5 else 'red'
        plt.title(f'{label}\n{conf:.2f}', color=color, pad=10)
    
    plt.tight_layout()
    plt.savefig(output_path / "summary_grid.png", 
                bbox_inches='tight', 
                dpi=300, 
                facecolor='white')
    plt.close()

@task_wrapper
def process_images(model, transform, input_path, output_path):
    class_labels = ['cat', 'dog']
    results = []
    
    # Get list of images with multiple extensions
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(input_path.glob(ext)))
    
    logger.info(f"Found {len(image_files)} images in {input_path}")
    
    if len(image_files) == 0:
        logger.warning(f"No images found in {input_path}")
        return results
    
    # Create progress bar
    progress = create_rich_progress()
    
    with progress:
        task = progress.add_task("[cyan]Processing images...", total=len(image_files))
        
        for img_path in image_files:
            try:
                logger.info(f"Processing image: {img_path}")
                
                # Prepare image
                img_tensor, original_img = process_image(img_path, transform)
                img_tensor = img_tensor.to(model.device)

                # Inference
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = F.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()

                # Get prediction
                predicted_label = class_labels[predicted_class]
                
                # Save individual prediction visualization
                viz_path = output_path / f"{img_path.stem}_prediction.png"
                logger.info(f"Saving prediction visualization to: {viz_path}")
                visualize_prediction(
                    original_img, 
                    predicted_label, 
                    confidence, 
                    viz_path
                )
                
                # Store results for summary
                results.append((img_path.name, original_img, predicted_label, confidence))
                logger.info(f"Prediction for {img_path.name}: {predicted_label} ({confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                logger.exception(e)
            
            progress.update(task, advance=1)
    
    if results:
        # Save summary grid
        summary_path = output_path / "summary_grid.png"
        logger.info(f"Saving summary grid to: {summary_path}")
        create_summary_grid(results, output_path)
        
        # Save text summary
        summary_txt = output_path / "predictions_summary.txt"
        with open(summary_txt, 'w') as f:
            f.write("Predictions Summary:\n")
            f.write("-" * 50 + "\n")
            for img_name, _, label, conf in results:
                f.write(f"Image: {img_name}\n")
                f.write(f"Prediction: {label}\n")
                f.write(f"Confidence: {conf:.2f}\n")
                f.write("-" * 50 + "\n")
    
    return results

def main():
    # Setup logger
    setup_logger()
    logger.info("Starting inference script")
    
    parser = argparse.ArgumentParser(description='Inference for Cat/Dog Classification')
    parser.add_argument('-input_folder', type=str, required=True, help='Input folder containing images')
    parser.add_argument('-output_folder', type=str, required=True, help='Output folder for predictions')
    parser.add_argument('-ckpt', type=str, required=True, help='Path to the checkpoint file')
    
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    logger.info(f"Created output directory: {args.output_folder}")

    # Load model
    logger.info("Loading model from checkpoint")
    model = CatDogClassifier.load_from_checkpoint(args.ckpt)
    model.eval()
    
    # Define transform - make sure it matches training transform
    transform = transforms.Compose([
        transforms.Resize((96, 96)),  # Match the training size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process images
    input_path = Path(args.input_folder)
    output_path = Path(args.output_folder)
    
    results = process_images(model, transform, input_path, output_path)
    logger.info(f"Processed {len(results)} images successfully")
    logger.info("Inference completed successfully")

if __name__ == "__main__":
    main() 