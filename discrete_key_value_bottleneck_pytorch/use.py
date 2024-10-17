import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Import Vision Transformer and DiscreteKeyValueBottleneck
from vit_pytorch import SimpleViT
from vit_pytorch.extractor import Extractor
from discrete_key_value_bottleneck import DiscreteKeyValueBottleneck

# Step 1: Dataset Preparation
# Define the transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match the input size of the model
    transforms.ToTensor()
])

# Load the CIFAR-10 test set
full_test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Reduce the test set to only 100 images
test_set = Subset(full_test_set, list(range(1000)))
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Step 2: Model Definition
# Define the Vision Transformer
vit = SimpleViT(
    image_size=256,
    patch_size=32,
    num_classes=10,  # Number of classes in CIFAR-10
    dim=512,
    depth=6,
    heads=16,
    mlp_dim=2048
)

# Wrap the ViT with Extractor to return embeddings
vit = Extractor(vit, return_embeddings_only=True)

# Define the encoder with the discrete key-value bottleneck
enc_with_bottleneck = DiscreteKeyValueBottleneck(
    encoder=vit,
    dim=512,
    num_memories=256,
    dim_memory=2048,
    decay=0.9,
)

# Step 3: Making Predictions
all_predictions = []
all_labels = []

# Set model to evaluation mode
enc_with_bottleneck.eval()

# Disable gradient computation for evaluation
with torch.no_grad():
    for images, labels in test_loader:
        memories = enc_with_bottleneck(images)
        
        # Get class predictions from the model's output
        class_scores = memories.mean(dim=1)  # Reducing memory output to class scores
        predictions = class_scores.argmax(dim=1).cpu().numpy()  # Get the predicted class label
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

# Step 4: Calculating Accuracy, Precision, and Recall
# Step 4: Calculating Accuracy, Precision, and Recall
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='macro', zero_division=1)
recall = recall_score(all_labels, all_predictions, average='macro', zero_division=1)

from sklearn.metrics import classification_report

# Step 4: Generating the Classification Report for Multiclass Evaluation
report = classification_report(all_labels, all_predictions, zero_division=1)

# Print the classification report
print(report)



