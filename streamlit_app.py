import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import ViTSkinDiseaseClassifier

st.title("Skin Disease Detector using Vision Transformer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTSkinDiseaseClassifier(num_classes=7).to(device)
model.load_state_dict(torch.load("vit_skin_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, 1).item()

    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    st.write(f"**Predicted Disease:** {class_names[pred]}")
