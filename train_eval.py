import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights, DenseNet201_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
from tqdm import tqdm

DATA_DIR = './dataset'
MODEL_DIR = './models'
RESULT_DIR = './results'
BATCH_SIZE = 32
EPOCHS = 15 
NUM_CLASSES = 150

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets and create dataloaders
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=2)
               for x in ['train', 'val']}

class_names = image_datasets['train'].classes

def train_and_evaluate(model, criterion, optimizer, num_epochs=5):
    # Function to train the model, return metrics, and track learning history
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Dictionary to store loss and accuracy history for learning curves
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            dataloader_iter = tqdm(dataloaders[phase], desc=f"[{phase.capitalize()}]")
            
            for inputs, labels in dataloader_iter:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                dataloader_iter.set_postfix(loss=f"{loss.item():.4f}")

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record metrics to history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                        
    # Evaluation phase to calculate final macro metrics
    print("\nCalculating final macro metrics on validation set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        val_iter = tqdm(dataloaders['val'], desc="[Evaluating]")
        for inputs, labels in val_iter:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"Result -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}\n")
    return acc, prec, rec, f1, copy.deepcopy(model.state_dict()), history

def save_learning_curve(history, model_name):
    # Generate and save train/val learning curve plots
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'{model_name} - Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(RESULT_DIR, f'{model_name}_learning_curve.png')
    plt.savefig(plot_path)
    plt.close()

def get_experiment_models():
    # Setup 1: ResNet18 Feature Extractor
    model1 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model1.parameters():
        param.requires_grad = False
    model1.fc = nn.Linear(model1.fc.in_features, NUM_CLASSES)

    # Setup 2: ResNet18 Fine-tuning
    model2 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model2.fc = nn.Linear(model2.fc.in_features, NUM_CLASSES)

    # Setup 3: MobileNetV2 Fine-tuning
    model3 = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model3.classifier[1] = nn.Linear(model3.classifier[1].in_features, NUM_CLASSES)

    # Setup 4: DenseNet201 Fine-tuning
    model4 = models.densenet201(weights=DenseNet201_Weights.DEFAULT)
    model4.classifier = nn.Linear(model4.classifier.in_features, NUM_CLASSES)

    return {
        "ResNet18_FeatureExtract": model1,
        "ResNet18_FineTuning": model2,
        "MobileNetV2_FineTuning": model3,
        "DenseNet201_FineTuning": model4
    }

if __name__ == '__main__':
    # Run experiments
    experiments = get_experiment_models()
    results = {}

    for name, model in experiments.items():
        print(f"========== Running experiment: {name} ==========")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        
        # Only optimize parameters that require gradients
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        
        # Unpack the history variable as well
        acc, prec, rec, f1, best_weights, history = train_and_evaluate(model, criterion, optimizer, num_epochs=EPOCHS)
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
        
        # Save the best model weights
        torch.save(best_weights, os.path.join(MODEL_DIR, f"{name}.pth"))
        
        # Save the learning curve plot for this specific model
        save_learning_curve(history, name)

    # Overall Metrics Visualization
    labels = list(results.keys())
    acc_vals = [results[l]['Accuracy'] for l in labels]
    prec_vals = [results[l]['Precision'] for l in labels]
    rec_vals = [results[l]['Recall'] for l in labels]
    f1_vals = [results[l]['F1'] for l in labels]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - 1.5*width, acc_vals, width, label='Accuracy')
    rects2 = ax.bar(x - 0.5*width, prec_vals, width, label='Precision')
    rects3 = ax.bar(x + 0.5*width, rec_vals, width, label='Recall')
    rects4 = ax.bar(x + 1.5*width, f1_vals, width, label='F1 Score')

    ax.set_ylabel('Scores')
    ax.set_title('Performance Comparison of Different Transfer Learning Setups')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'metrics_comparison.png'))
    plt.close()
    
    print("All experiments completed successfully. Models and plots are saved.")