import os 
import pandas as pd 
import numpy as np
import time
import sys 

provider = sys.argv[1]
if provider == "openai":
  SKILLS_DIM = 1536
else:
  SKILLS_DIM = 768

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
#just API key at Starter level, token = os.environ['MILVUS_API_KEY']
# this is how you connect with Standard (not Starter) level
token = "db_admin:"+ os.environ['MILVUS_PASSWORD']
uri = os.environ['MILVUS_URL']
connections.connect("default", uri=uri, token=token)
collection_name = provider + "_skills"
utility.drop_collection(collection_name)

# Load the IDs from the CSV file
ids_path = './data/skill_list.csv'
ids_df = pd.read_csv(ids_path)
ids = ids_df['abbreviation'].values
skill_vectors_file = "./data/" + provider + "_skill_vectors.npy"
vectors = np.load(skill_vectors_file)
levels = [str(x) for x in ids_df['level'].values]
num_vectors = vectors.shape[0]
data=[ids,vectors,levels]

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=SKILLS_DIM),
    FieldSchema(name="level", dtype=DataType.VARCHAR,max_length=16)
]
schema = CollectionSchema(fields, "skills")

skills_collection = Collection(collection_name, schema)
tot_duration = 0
start = time.time()
result = skills_collection.insert(data)
end = time.time() 
duration = end - start
tot_duration += duration 
print(f"Result of data upsert: {result} in {duration} seconds") 

# creating index
start = time.time()
index_params = {
  "metric_type":"L2",
  "index_type":"HNSW",
  "params":{}
} 
skills_collection.create_index(
  field_name="embeddings", 
  index_params=index_params
)
result = utility.index_building_progress(collection_name)
end = time.time()
duration = end - start 
print(f"Result of index creation: {result} in {duration} seconds") 
tot_duration += duration 

print(f"{num_vectors} vectors uploaded in {tot_duration} seconds")


