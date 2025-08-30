👗 Fashion Recommender System

A content-based image recommender system that suggests visually similar fashion products.
Upload any fashion product image, and the system retrieves the most similar items from the dataset.

Built with PyTorch + Streamlit + KNN.

🚀 Features

Upload an image via a Streamlit web app.

Extract embeddings using a ResNet50-based encoder trained on a fashion dataset.

Compare uploaded image with precomputed embeddings using KNN (Euclidean distance).

Display Top-5 most visually similar products from the dataset.

📂 Project Structure
```
├── main.py                                 # Streamlit web app (frontend)
├── Recommender System(Fashion_2).ipynb     # Notebook for training + embedding generation
├── encoder_weights.pth                     # Trained ResNet50 encoder
├── fextract_embeddings.npy    # Precomputed normalized embeddings
├── filenames_22.npy                        # Image filenames for lookup
└── dataset/
    └── images/(Kaggle Dataset)             # Fashion product images
```
⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/your-username/fashion-recommender.git
cd fashion-recommender

2. Install Dependencies
pip install streamlit torch torchvision scikit-learn pillow numpy

3. Prepare Dataset & Models

Place your fashion dataset inside:

dataset/images/(Kaggle Dataset Link is provided)


Make sure the following files exist (generated from the Jupyter Notebook):

encoder_weights.pth – trained ResNet50 encoder model

extract_embeddings.npy – normalized feature embeddings


4. Run Streamlit App
streamlit run main.py

🧑‍💻 Workflow
Notebook (Recommender System(Fashion_2).ipynb)

Preprocess fashion dataset.

Train a ResNet50 encoder (last layers replaced with Adaptive Pooling + Flatten).

Generate embeddings for all images.

Save:

encoder_weights.pth
extract_embeddings.npy


Streamlit App (main.py)

Load the encoder and embeddings.

User uploads an image → preprocessed → converted into embedding.

Compute nearest neighbors using KNN (Euclidean).

Display Top-5 most similar fashion products.

📸 Example

Upload: Red Dress

Output: Top-5 visually similar red dresses from the dataset.

🌐 Deployment

You can deploy this app on:

Streamlit Cloud (recommended)

Heroku / AWS / GCP / Azure

For Streamlit Cloud:

Push your repo to GitHub.

Go to Streamlit Cloud
.

Deploy your repo → select main.py as entry point.

Make sure your dataset and model files are included or hosted remotely.
