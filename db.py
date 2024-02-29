import os
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone
import torch
# Set your Pinecone API key and index details
index_name = 'my-index'
api_key = '29e5fac8-3f3a-4153-866c-b96cfde9d809'

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Function to generate embeddings for an image using CLIP
# Function to generate embeddings for an image using CLIP
from transformers import CLIPProcessor, CLIPModel

def generate_embeddings(image_path, text_input=None):

  try:
    image = Image.open(image_path)
  except Exception as e:
    raise ValueError(f"Error opening image: {e}")

  # Create CLIP processor and model
  processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
  model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

  # Preprocess image and text (if provided)
  inputs = processor(images=image, return_tensors="pt")
  if text_input:
    text_tokens = processor(text=text_input, return_tensors="pt")
    inputs.update(text_tokens)  # Combine image and text inputs

  # Pass preprocessed data to the model
  with torch.no_grad():
    outputs = model(**inputs)

  # Check for updated output structure (example: 'last_hidden_state')
  if 'last_hidden_state' in outputs:
    embeddings = outputs.last_hidden_state
  else:
    embeddings = outputs.image_embeds
  return embeddings
# Specify the directory containing image files
image_dir = '/Users/rajeshgauswami/Documents/Sample'

# Generate embeddings for each image in the directory
image_embeddings = []
text_input = "This is an image of a intraoral."

for image_file in os.listdir(image_dir):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, image_file)
        embeddings = generate_embeddings(image_path,text_input)
        image_embeddings.append(embeddings)

# Convert the list of embeddings to a NumPy array
image_embeddings_array = np.array(image_embeddings)
print(image_embeddings_array)


# Insert the embeddings into Pinecone
# pc.Index(index_name).upsert(
#     vectors=[
#         ("vec1", [0.1, 0.2, 0.3, 0.4], {"genre": "drama"}),
#     ],
#     namespace="intra-namespace"
# )
for i, embedding in enumerate(image_embeddings_array):
    vector_name = f"vec{i}"  # You can use a meaningful name for each vector

    pc.Index(index_name).upsert(
       vectors=[{
            "id": str(i), 
            "values": embedding.tolist(), 
        }],
        namespace="intra-namespace"
    )


# pc.Index(index_name).upsert(vectors=list(enumerate(image_embeddings_array.tolist())))

print(f"{len(image_embeddings)} image embeddings inserted into Pinecone index '{index_name}'.")
