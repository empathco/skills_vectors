from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import CollectionStatus
import os 
import pandas as pd 
import numpy as np
import time
import requests 

QDRANT_SERVER=os.environ['QDRANT_URL']
BATCH_SIZE = 100
VEC_SIZE = 512

from qdrant_client import QdrantClient

client = QdrantClient(
    url=os.environ['QDRANT_URL'], 
    api_key=os.environ['QDRANT_API_KEY']
)

client.delete_collection(collection_name="skills")

start = time.time()
client.create_collection(
    collection_name="skills",
    vectors_config=models.VectorParams(size=VEC_SIZE, distance=models.Distance.COSINE),
)

# Load the IDs from the CSV file
ids_path = './data/skill_list.csv'
ids_df = pd.read_csv(ids_path)
vectors = np.load('./data/skill_vectors.npy')
vlists=[]
i=0
for v in vectors:
    vlists.append(v.tolist())
    i+=1
print(f"Vlists {vlists[0]}")
num_vectors = len(ids_df)
ids=[i for i in range(len(vlists))]
payloads = [{'abbreviation':ids_df['abbreviation'][i],'l':str(ids_df['level'][i])}  for i in range(num_vectors)]

print(f"{num_vectors} vectors uploading...")
client.upsert(collection_name="skills",points=models.Batch(ids=ids,payloads=payloads,vectors=vlists))
end = time.time()
tot_duration = end - start
print(f"{num_vectors} vectors uploaded in {tot_duration} seconds")

