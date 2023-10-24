import os 
import pandas as pd 
import numpy as np
import time

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

# Load the IDs from the CSV file
ids_path = './data/epl_skill_list.csv'
ids_df = pd.read_csv(ids_path)
ids = ids_df['abbreviation'].values
vectors = np.load('./data/skill_vectors.npy')
num_vectors = vectors.shape[0]
data=[ids,vectors]
SKILLS_DIM = 512
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=SKILLS_DIM)
]
schema = CollectionSchema(fields, "skills")
print(f"Create collection: skills")
skills_collection = Collection("skills", schema, consistency_level="Strong")
start = time.time()
result = skills_collection.insert(data)
collection = Collection("skills") 
index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024}
} 
# don't need to recreate index here    
#collection.create_index(
#  field_name="embeddings", 
#  index_params=index_params
#)
print(f"Result of data insert: {result}")
end = time.time()
tot_duration = end - start

print(f"{num_vectors} vectors uploaded in {tot_duration} seconds")


