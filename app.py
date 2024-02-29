import os
from pinecone import Pinecone, ServerlessSpec
import torch
import clip
from PIL import Image

api_key = '29e5fac8-3f3a-4153-866c-b96cfde9d809' 
pc = Pinecone(api_key=api_key)

index_name = "smilesimulation" 
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=512, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

index = pc.Index(index_name)

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to extract image
def extract_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features.cpu().numpy().flatten()

metadata_list = [
    {"animation": "https://webview.32-stories.com/?mlink=https://onyx-uploads.s3.eu-west-2.amazonaws.com/Client983/SUBM-DZFGYZS/F4597E6637C342B4ADDC0440D140E78F.iiwgl&fg=004&bg=ddd&p=CHOWDJ"},  # Metadata for the first image
    {"animation": "https://webview.32-stories.com/?mlink=https://onyx-uploads.s3.eu-west-2.amazonaws.com/Client983/SUBM-DZFGYZS/F4597E6637C342B4ADDC0440D140E78F.iiwgl&fg=004&bg=ddd&p=CHOWDJ"},  # Metadata for the second image
    {"animation": "https://webview.32-stories.com/?mlink=https://onyx-uploads.s3.eu-west-2.amazonaws.com/Client983/SUBM-DZFGYZS/F4597E6637C342B4ADDC0440D140E78F.iiwgl&fg=004&bg=ddd&p=CHOWDJ"},  # Metadata for the third image
]

image_paths = [
    "/Users/nik/Downloads/intraoralimages/4d6a30dc-f48f-44e8-950a-de65d24e0264-intra oral 5.jpg", 
    "/Users/nik/Downloads/intraoralimages/0a660a11-cd3a-4b6c-9f31-bfa2f8f97317-_MG_9337.JPG", 
    "/Users/nik/Downloads/intraoralimages/2d2f70f2-e027-4e25-a53c-8c81c10cfb7a-IMG_9619.jpeg"
]

namespace = "simulation"

image_dir = '/Users/rajeshgauswami/Documents/Sample'

for image_id, (image_path, metadata) in enumerate(zip(image_paths, metadata_list)):
    vector = extract_features(image_path)
    index.upsert(
        vectors=[{
            "id": str(image_id), 
            "values": vector.tolist(), 
            "metadata": metadata
        }],
        namespace=namespace
    )

# Test query
query_vector = extract_features("/Users/nik/Downloads/intraoralimages/2d2f70f2-e027-4e25-a53c-8c81c10cfb7a-IMG_9619.jpeg").tolist()

# Fetch top 5 similar vectors
results = index.query(
    namespace=namespace,
    vector=query_vector,
    top_k=5,
    include_values=True,
    include_metadata=True
)

for match in results["matches"]:
    print(f"Found match: {match['id']} with score: {match['score']}, metadata: {match.get('metadata', {})}")