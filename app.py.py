import streamlit as st
import torch, os
import torchvision.models as models
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from PIL import Image

# ------------------------------------
# Streamlit Title
# ------------------------------------
st.title("👗 Similar Product Recommender System")

# ------------------------------------
# 1. Utility: Save Uploaded File
# ------------------------------------
def save_uploaded_file(uploaded_file):
    """Save uploaded image to current working directory."""
    try:
        with open(os.path.join(os.getcwd(), uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except:
        return False

# ------------------------------------
# 2. Load Precomputed Embeddings
# ------------------------------------
all_embeddings = np.load("fashion_embeddings_modified_norm.npy")
all_filenames = np.load("filenames_22.npy")

# Fit KNN model
knn = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
knn.fit(all_embeddings)

# ------------------------------------
# 3. Image Preprocessing
# ------------------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------
# 4. Load Encoder (ResNet50 backbone)
# ------------------------------------
resnet = models.resnet50(weights=None)
modules = list(resnet.children())[:-2]
encoder = nn.Sequential(
    *modules,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)
encoder.load_state_dict(torch.load("fashion_encoder.pth", map_location="cpu"))
encoder.eval()

# ------------------------------------
# 5. Feature Extraction
# ------------------------------------
def transform_image(uploaded_file):
    """Transform uploaded image into model input tensor."""
    img_path = os.path.join(os.getcwd(), uploaded_file.name)
    img = Image.open(img_path).convert("RGB")
    img = preprocess(img).unsqueeze(0)
    return img

def similarity_calculator(uploaded_file):
    """Compute similarity between uploaded image and dataset embeddings."""
    img = transform_image(uploaded_file)
    embedding = encoder(img)
    embedding = embedding.view(embedding.size(0), -1)
    embedding = embedding / embedding.norm(dim=1, keepdim=True)
    embedding = embedding.cpu().detach().numpy()

    distances, indices = knn.kneighbors(embedding, n_neighbors=10)
    return indices[0][1:], distances[0][1:]

# ------------------------------------
# 6. Display Similar Products
# ------------------------------------
def extract_images(uploaded_file):
    indices, dists = similarity_calculator(uploaded_file)
    cols = st.columns(5)

    for i, index in enumerate(indices[:5]):
        filepath = os.path.join(
            r"C:\Users\thaku\jupyter notebook datasets\Fashion Recommender System\Dataset\fashion-dataset\images",
            all_filenames[index]
        )
        img = Image.open(filepath).convert("RGB")
        with cols[i]:
            st.image(img, caption=all_filenames[index], use_container_width=True)

# ------------------------------------
# 7. Streamlit File Uploader
# ------------------------------------
uploaded_file = st.file_uploader("📂 Upload an image to find similar products")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image")
        st.subheader("🔎 Top 5 Similar Products")
        extract_images(uploaded_file)
    else:
        st.error("⚠️ Error Occurred! Please upload again.")
