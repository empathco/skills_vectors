import pinecone, os 
from pinecone import Pinecone
import pandas as pd 
import numpy as np
import time
import sys 

provider = sys.argv[1]
if provider == "openai":
  SKILLS_DIM = 1536
else:
  SKILLS_DIM = 768

api_key = os.environ['PINECONE_API_KEY']
index_name = 'skills-' + provider  

env = os.environ['PINECONE_ENV']
BATCH_SIZE=300

pc=Pinecone(api_key=api_key, environment = env)
result=pc.list_indexes()
print(result)

# Load the IDs from the CSV file
ids_path = './data/skill_list.csv'
ids_df = pd.read_csv(ids_path)
ids = ids_df['abbreviation'].values

skill_vectors_file = "./data/" + provider + "_skill_vectors.npy"
vectors = np.load(skill_vectors_file)
num_vectors = vectors.shape[0]

# Generate a list of dictionaries for upsert
upsert_data = [{"id": ids[i], "values": vectors[i].tolist(),"metadata": {'level':str(ids_df['level'][i])}} for i in range(len(vectors))]
print(f"{num_vectors} vectors uploading to the '{index_name}' index.")
with pc.Index(index_name) as index:
    index.delete(delete_all=True)
    first = 0
    tot_duration = 0
    while first < num_vectors: 
        start = time.time()
        next = first + BATCH_SIZE 
        print(f"Upserting from {first} to {next} to the {index_name} index.")
        index.upsert(vectors=upsert_data[first:next],namespace='')
        first = next
        end = time.time()
        duration = end - start
        print(f"{BATCH_SIZE} vectors uploaded to the {index_name} index in {duration} seconds")
        tot_duration += duration 
        
print(f"{num_vectors} vectors uploaded to the {index_name} index in {tot_duration} seconds")
   