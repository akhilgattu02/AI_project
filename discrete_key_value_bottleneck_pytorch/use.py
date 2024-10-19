import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import numpy as np

# Import Vision Transformer and DiscreteKeyValueBottleneck
from vit_pytorch import SimpleViT
from vit_pytorch.extractor import Extractor
from Discrete_key_value_bottleneck import DiscreteKeyValueBottleneck

# Step 1: Dataset Preparation
# Define the transformations for the dataset with data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(10),
    transforms.Resize((256, 256)),  # Resize to match the input size of the model
    transforms.ToTensor()
])

# Load the CIFAR-10 training set
full_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a balanced subset of 1000 training images with equal distribution of all 10 classes
num_samples_per_class = 100  # 100 samples per class to get a total of 1000 samples
class_indices_train = {i: [] for i in range(10)}

# Distribute the indices for each class
for idx, (_, label) in enumerate(full_train_set):
    if len(class_indices_train[label]) < num_samples_per_class:
        class_indices_train[label].append(idx)
    if all(len(indices) >= num_samples_per_class for indices in class_indices_train.values()):
        break

# Flatten the list of indices and create the balanced training subset
balanced_indices_train = [idx for indices in class_indices_train.values() for idx in indices]
balanced_train_set = Subset(full_train_set, balanced_indices_train)
train_loader = DataLoader(balanced_train_set, batch_size=32, shuffle=True)

# Load the CIFAR-10 test set
full_test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create a balanced subset of 1000 test images with equal distribution of all 10 classes
class_indices_test = {i: [] for i in range(10)}

# Distribute the indices for each class
for idx, (_, label) in enumerate(full_test_set):
    if len(class_indices_test[label]) < num_samples_per_class:
        class_indices_test[label].append(idx)
    if all(len(indices) >= num_samples_per_class for indices in class_indices_test.values()):
        break

# Flatten the list of indices and create the balanced test subset
balanced_indices_test = [idx for indices in class_indices_test.values() for idx in indices]
balanced_test_set = Subset(full_test_set, balanced_indices_test)
test_loader = DataLoader(balanced_test_set, batch_size=32, shuffle=False)

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

# Step 3: Define the Classifier with the Bottleneck
class ClassifierWithBottleneck(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(ClassifierWithBottleneck, self).__init__()
        self.encoder = encoder
        # Infer the embedding dimension from a sample input
        dummy_input = torch.randn(1, 3, 256, 256)  # A dummy image to pass through the encoder
        with torch.no_grad():
            memory_output = self.encoder(dummy_input)
            embedding_dim = memory_output.shape[-1]  # Infer the dimension from the output
        self.classifier = nn.Linear(embedding_dim, num_classes)  # Linear layer to map to class scores
        self.batch_norm = nn.BatchNorm1d(embedding_dim)  # Adding Batch Normalization

    def forward(self, x):
        memories = self.encoder(x)
        memory_representation = memories.mean(dim=1)  # Reduce to a single memory slot representation
        memory_representation = self.batch_norm(memory_representation)  # Apply Batch Normalization
        class_scores = self.classifier(memory_representation)
        return class_scores

# Instantiate the classifier with the bottleneck
model = ClassifierWithBottleneck(enc_with_bottleneck, num_classes=10)

# Step 4: Define Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Using Adam optimizer
criterion = nn.CrossEntropyLoss()

# Step 5: Training the Model
def train_model(model, data_loader, optimizer, criterion, num_epochs=5):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in data_loader:
            optimizer.zero_grad()  # Zero gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            total_loss += loss.item()  # Accumulate loss
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')

# Train on the balanced training dataset
train_model(model, train_loader, optimizer, criterion, num_epochs=5)

# Step 6: Making Predictions
all_predictions = []
all_labels = []

# Set model to evaluation mode
model.eval()

# Disable gradient computation for evaluation
with torch.no_grad():
    for images, labels in test_loader:
        class_scores = model(images)
        predictions = class_scores.argmax(dim=1).cpu().numpy()  # Get the predicted class label
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

# Step 7: Generating the Classification Report
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='macro', zero_division=1)
recall = recall_score(all_labels, all_predictions, average='macro', zero_division=1)
report = classification_report(all_labels, all_predictions, zero_division=1)

# Print the accuracy, precision, recall, and classification report
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(report)

