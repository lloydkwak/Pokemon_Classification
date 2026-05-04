import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Set constants
MODEL_PATH = './models/ResNet18_FineTuning.pth'
NUM_CLASSES = 150
DATA_DIR = './dataset/train'

# Check if dataset directory exists to load class names
if os.path.exists(DATA_DIR):
    class_names = sorted(os.listdir(DATA_DIR))
else:
    class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    # Initialize the model structure
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    # Load weights if file exists
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Pokemon Classifier")
st.write("Upload a Pokemon image to predict its name!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Predicting...")
    
    # Process image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get Top-5 predictions
        top_prob, top_catid = torch.topk(probabilities, 5)
    
    st.subheader("Top-5 Predictions:")
    for i in range(top_prob.size(0)):
        class_name = class_names[top_catid[i].item()]
        prob_percent = top_prob[i].item() * 100
        st.write(f"{i+1}. **{class_name}** ({prob_percent:.2f}%)")