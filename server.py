import uvicorn
from fastapi import File, UploadFile, HTTPException, FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
import base64
import io
import torch
import clip
from PIL import Image
import os
from pinecone import Pinecone, ServerlessSpec
from bson import ObjectId
from fastapi.encoders import jsonable_encoder

index_name = "smilesimulation" 
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB configuration
DATABASE_NAME = "core"
MONGO_URI = "mongodb+srv://rajesh:UP1GwVn3BoaGu3PP@main.vaapi.mongodb.net/core?retryWrites=true&w=majority"

# MongoDB connection setup
client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]

# Image processing configuration (assuming you have a model and preprocess function)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def extract_features_from_base64(base64_data):
    try:
        image_data = base64.b64decode(base64_data)
        image = preprocess(Image.open(io.BytesIO(image_data))).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


class ImageRequest(BaseModel):
    image: str


# Specify the path to the uploads folder
UPLOAD_FOLDER = "uploads"

# Create the uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/ping")
async def ping():
    collections = await db.list_collection_names()
    return JSONResponse(content={"collections": collections})

def extract_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features.cpu().numpy().flatten()
api_key = '29e5fac8-3f3a-4153-866c-b96cfde9d809' 
pc = Pinecone(api_key=api_key)
namespace = "simulation"
index = pc.Index(index_name)


@app.post("/image")
async def image(file: UploadFile = File(...)):
    try:
        # Handle file upload
        contents = await file.read()
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        query_vector = extract_features(file_path).tolist()
        result= index.query(
            namespace=namespace,
            vector=query_vector,
            top_k=5,
            include_values=False,
            include_metadata=True
        )
        print(result.matches[0].metadata)
        submission = result.matches[0].metadata['submission']
        object_id = ObjectId(submission)
        submissionInfo = await db['submissions'].find_one({"_id":object_id})
        design = await db['treatmentdesigns'].find_one({"submission":object_id,"status":"APPROVED"})
        print(submissionInfo)
        print(design)
        return {
            "result": "success",
            "submission": submission,  # Include the result in the response
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file upload: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3005)
