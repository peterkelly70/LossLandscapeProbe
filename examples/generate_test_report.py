import os
import torch
import numpy as np
import base64
from io import BytesIO
from datetime import datetime
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image

# Import the model architecture from cifar10_example.py
from cifar10_example import SimpleCNN

def get_cifar10_classes():
    """Return the class names for CIFAR-10."""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']

def load_model(model_path):
    """Load the trained model."""
    # Create the model with default architecture
    model = SimpleCNN(num_channels=32)  # Default from meta_config
    
    # Load the state dict directly
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

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
                font-family: Arial, sans-serif;
                margin: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }
            h1, h2 {
                color: #333;
            }
            .stats {
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .image-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 15px;
            }
            .image-card {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                text-align: center;
            }
            .image-card img {
                width: 100%;
                height: auto;
            }
            .correct {
                color: green;
                font-weight: bold;
            }
            .incorrect {
                color: red;
                font-weight: bold;
            }
            .confidence {
                font-size: 0.8em;
                color: #666;
            }
            .sample-notice {
                background-color: #f0f8ff;
                border-left: 4px solid #1e90ff;
                padding: 10px 15px;
                margin-bottom: 20px;
                border-radius: 3px;
            }
            .sample-notice p {
                margin: 0;
                color: #333;
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
    
    # Add the sample notice directly after the heading
    sample_notice = '''
    <div style="background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 10px; margin-bottom: 20px;">
        <p><strong>Note:</strong> The accuracy statistics shown here are calculated based only on this sample, not the full test set.</p>
    </div>
    '''
    html_content += sample_notice
    
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

if __name__ == "__main__":
    # Path to the saved model
    model_path = "meta_model_trained.pth"
    
    # Output path for the HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cifar10_test_report_{timestamp}.html")
    
    # Load model and generate report
    print("Loading model...")
    model = load_model(model_path)
    
    print("Loading test data...")
    test_loader = get_test_loader()
    
    print("Generating report...")
    generate_html_report(model, test_loader, output_path, max_samples=200)
