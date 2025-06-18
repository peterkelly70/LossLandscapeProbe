import os
import torch
import numpy as np
import base64
from io import BytesIO
from datetime import datetime
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image

# Import the model architecture from llp.models.simple_cnn
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from llp.models.simple_cnn import SimpleCNN

def get_cifar10_classes():
    """Return the class names for CIFAR-10."""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']

def load_model(model_path):
    """Load the trained model."""
    try:
        # Load the state dict directly to examine it
        state_dict = torch.load(model_path)
        
        # Check if this is a complete model or just a state dict
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            # This is a complete model checkpoint with additional metadata
            print(f"Loading state_dict from checkpoint in {model_path}")
            state_dict = state_dict['state_dict']
        
        # Determine if this is a meta-model or a trained model based on filename
        is_meta_model = 'meta_model' in os.path.basename(model_path).lower()
        
        # Try to determine the model architecture from the state dict
        # Look for conv1.weight to determine the number of channels
        if 'conv1.weight' in state_dict:
            # Get the shape of the first conv layer
            conv1_shape = state_dict['conv1.weight'].shape
            # The first dimension is the number of output channels
            num_channels = conv1_shape[0]
            print(f"Detected model with {num_channels} channels in first layer")
        else:
            # Default if we can't determine
            num_channels = 32 if not is_meta_model else 64
            print(f"Using {'meta-model' if is_meta_model else 'trained model'} architecture with {num_channels} channels")
        
        # Create the model with the detected architecture
        model = SimpleCNN(num_channels=num_channels)
        
        # Try to load with strict=True first
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"Successfully loaded {'meta-model' if is_meta_model else 'trained model'} from {model_path} with exact architecture match")
        except Exception as strict_error:
            print(f"Warning: Strict loading failed: {strict_error}")
            print("Attempting to load with strict=False to handle architecture differences")
            model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded {'meta-model' if is_meta_model else 'trained model'} from {model_path} with partial architecture match")
        
        return model, True
    except FileNotFoundError:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")
        # Create default model
        model = SimpleCNN(num_channels=32)
        return model, False
    except Exception as e:
        print(f"Error loading model: {e}")
        # Create default model
        model = SimpleCNN(num_channels=32)
        return model, False

def get_test_loader():
    """Get the CIFAR-10 test dataset loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=100,
        shuffle=False
    )
    
    return test_loader

def predict_batch(model, images):
    """Run prediction on a batch of images."""
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)
    return predicted, probabilities

def image_to_base64(img_tensor):
    """Convert a tensor image to a base64 string for HTML embedding."""
    # Convert tensor to numpy and denormalize
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))
    img = img * 0.5 + 0.5  # Denormalize
    img = np.clip(img, 0, 1)
    
    # Convert to PIL image and then to base64
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return img_str

def generate_html_report(model, test_loader, output_path, max_samples=100):
    """Generate an HTML report with test images and predictions."""
    classes = get_cifar10_classes()
    
    # Start HTML content
    # Create the initial HTML content with title and styles
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CIFAR-10 Test Results</title>
        <style>
            body {
                font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                margin: 20px;
                max-width: 1200px;
                margin: 0 auto;
                background: #111;
                color: #eee;
            }
            h1, h2 {
                color: #4cf;
            }
            .stats {
                background-color: #222;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                border: 1px solid #444;
            }
            .image-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 15px;
            }
            .image-card {
                border: 1px solid #444;
                border-radius: 5px;
                padding: 10px;
                text-align: center;
                background-color: #222;
            }
            .image-card img {
                width: 100%;
                height: auto;
                border: 1px solid #333;
            }
            .correct {
                color: #4f8;
                font-weight: bold;
            }
            .incorrect {
                color: #f88;
                font-weight: bold;
            }
            .confidence {
                font-size: 0.8em;
                color: #aaa;
            }
            .sample-notice {
                background-color: #1a2a3a;
                border-left: 4px solid #3498db;
                padding: 10px 15px;
                margin-bottom: 20px;
                border-radius: 3px;
            }
            .sample-notice p {
                margin: 0;
                color: #eee;
            }
        </style>
    </head>
    <body>
        <h1>CIFAR-10 Test Results</h1>
        <div class="stats" id="stats">
            <h2>Statistics</h2>
            <p>Loading...</p>
        </div>
    """
    
    # Add the heading with the sample size
    html_content += "<h2>Test Images and Predictions (200/10000)</h2>\n"
    
    # Add the accuracy notice directly after the heading
    accuracy_notice = '''
    <div style="background-color: #1a2a3a; border-left: 4px solid #3498db; padding: 10px; margin-bottom: 20px;">
        <p style="color: #eee;"><strong>Note:</strong> Accuracy is based on the full test set of 10,000 samples. Only a subset of images is shown here.</p>
    </div>
    '''
    html_content += accuracy_notice
    
    # Add the image grid container
    html_content += """
        <div class="image-grid">
    """
    
    # Sample notice is now added directly after the heading
    
    # Process test data
    correct = 0
    total = 0
    samples_added = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    for images, labels in test_loader:
        predicted, probabilities = predict_batch(model, images)
        
        for i in range(len(labels)):
            if samples_added >= max_samples:
                break
                
            # Get image and prediction details
            img_tensor = images[i]
            true_label = labels[i].item()
            pred_label = predicted[i].item()
            confidence = probabilities[i][pred_label].item() * 100
            
            # Convert image to base64 for HTML embedding
            img_str = image_to_base64(img_tensor)
            
            # Determine if prediction is correct
            is_correct = (pred_label == true_label)
            if is_correct:
                correct += 1
                class_correct[true_label] += 1
            class_total[true_label] += 1
            total += 1
            
            # Add image card to HTML
            status_class = "correct" if is_correct else "incorrect"
            html_content += f"""
            <div class="image-card">
                <img src="data:image/png;base64,{img_str}" alt="CIFAR-10 Image">
                <p>True: {classes[true_label]}</p>
                <p class="{status_class}">Pred: {classes[pred_label]}</p>
                <p class="confidence">Confidence: {confidence:.1f}%</p>
            </div>
            """
            
            samples_added += 1
        
        if samples_added >= max_samples:
            break
    
    # Calculate overall accuracy
    accuracy = correct / total * 100
    
    # Complete the HTML
    html_content += """
        </div>
        <script>
            document.getElementById('stats').innerHTML = `
                <h2>Statistics</h2>
                <p><strong>Overall Accuracy:</strong> """ + f"{accuracy:.2f}%" + """</p>
                <p><strong>Class Accuracies:</strong></p>
                <ul>
    """
    
    # Add per-class accuracy
    for i in range(10):
        class_acc = class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0
        html_content += f"<li>{classes[i]}: {class_acc:.2f}%</li>\n"
    
    html_content += """
                </ul>
            `;
        </script>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Report generated at: {output_path}")
    print(f"Overall accuracy: {accuracy:.2f}%")

def generate_placeholder_report(output_path):
    """Generate a placeholder report when no model is available."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CIFAR-10 Test Report (Placeholder)</title>
        <style>
            body {{ 
                font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; 
                margin: 20px; 
                background: #111; 
                color: #eee; 
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2 {{ color: #4cf; }}
            .placeholder {{ 
                background-color: #222; 
                padding: 20px; 
                border-radius: 5px; 
                text-align: center; 
                border: 1px solid #444;
            }}
            .note {{ 
                color: #f88; 
                background-color: #2a1a1a; 
                padding: 10px; 
                border-radius: 5px; 
                margin-top: 20px; 
                border-left: 4px solid #f88;
            }}
        </style>
    </head>
    <body>
        <h1>CIFAR-10 Test Report (Placeholder)</h1>
        <p>Generated on: {timestamp}</p>
        
        <div class="placeholder">
            <h2>No trained model available</h2>
            <p>This is a placeholder report. No trained model was found at the expected location.</p>
            <p>To generate a complete report, please train a model first using the CIFAR-10 training scripts.</p>
        </div>
        
        <div class="note">
            <p><strong>Note:</strong> Once training is complete, run the report generation again to see model performance and predictions.</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Generated placeholder report at {output_path}")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate CIFAR-10 test report')
    parser.add_argument('--model', type=str, default=None, help='Path to the model file')
    parser.add_argument('--output', type=str, default=None, help='Output path for the HTML report')
    parser.add_argument('--model-type', type=str, default='cifa10', 
                        choices=['cifa10', 'cifa10_10', 'cifa10_20', 'cifa10_30', 'cifa10_40',
                                'cifa100', 'cifa100_10', 'cifa100_20', 'cifa100_30', 'cifa100_40',
                                'cifa100_transfer'],
                        help='Model type to determine the output directory')
    args = parser.parse_args()
    
    # Determine model path based on model type
    model_type = args.model_type
    
    # Path to the saved model - use model-specific directories with consistent naming
    if args.model:
        model_path = args.model
    else:
        # Extract dataset number (10 or 100) from model_type
        dataset_num = model_type[-2:] if 'cifa' in model_type else '10'
        
        # Determine if we're looking for a meta-model or a trained model
        is_meta_model = 'meta' in model_type.lower()
        
        # Check for model in the model-specific directory
        # For trained models (not meta-models), prioritize trained models over meta-models
        if not is_meta_model:
            model_paths = [
                # Standard naming convention for trained models
                os.path.join("trained", model_type, f"cifa{dataset_num}_model.pth"),
                os.path.join("trained", model_type, f"cifa{dataset_num}_trained.pth"),
                
                # Multisamplesize trained models
                os.path.join("trained", model_type, f"cifa{dataset_num}_multisamplesize_trained.pth"),
                os.path.join("trained", model_type, f"cifa{dataset_num}_multisamplesize_model.pth"),
                
                # Best models (especially for transfer learning)
                os.path.join("trained", model_type, f"cifa{dataset_num}_best_model.pth"),
                
                # Latest models in reports directory
                os.path.join("reports", model_type, f"latest_model.pth"),
                os.path.join("reports", model_type, f"cifa{dataset_num}_latest_model.pth"),
                
                # Legacy paths for backward compatibility
                os.path.join("models", f"cifar{dataset_num}_model.pth"),
                os.path.join("checkpoints", f"cifar{dataset_num}_model.pth")
            ]
        else:
            # Meta-models have a different architecture, so we prioritize them
            model_paths = [
                # Meta models
                os.path.join("trained", model_type, f"cifa{dataset_num}_meta_model.pth"),
                os.path.join("trained", "meta_model_trained.pth"),
                os.path.join("models", "meta_model_trained.pth"),
            ]
        
        # Try to find a model file
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found model at path: {model_path}")
                break
        
        # If no model found, default to the first path (which will fail gracefully)
        if model_path is None:
            model_path = model_paths[0]
            print(f"No model found, will try: {model_path}")

    
    # Output path for the HTML report
    model_type = args.model_type
    
    # Determine output directory based on model type
    if args.output:
        output_path = args.output
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        # Use model-specific directory with fixed filename
        output_dir = os.path.join("reports", model_type)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "latest_test_report.html")
    
    # Load model and generate report
    print("Loading model...")
    model, model_loaded = load_model(model_path)
    
    # Always load test data
    print("Loading test data...")
    test_loader = get_test_loader()
    
    if model_loaded:
        print("Generating report with test images and predictions...")
        # Set eval mode explicitly
        model.eval()
        # Generate the report with sample test images
        generate_html_report(model, test_loader, output_path, max_samples=200)
        print(f"Report generated at: {output_path}")
        
        # Evaluate on full test set to get overall accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Overall accuracy: {accuracy:.2f}%")
        print("CIFAR-10 test report generated successfully.")
    else:
        print("Generating placeholder report...")
        generate_placeholder_report(output_path)
        print("CIFAR-10 placeholder report generated (no model found).")
        
    # Return success code
    sys.exit(0)
