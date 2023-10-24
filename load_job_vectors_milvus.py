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
token = os.environ['MILVUS_API_KEY']
uri = os.environ['MILVUS_URL']
connections.connect("default", uri=uri, token=token)

# Load the IDs from the CSV file
ids_path = './data/job_title_desc.csv'
ids_df = pd.read_csv(ids_path)
ids = ids_df['job_code'].values
# load the vectors fron the numpy file
vectors = np.load('./data/jd_and_title_sem_vec.npy')

num_vectors = vectors.shape[0]

data=[ids,vectors]

SKILLS_DIM = 512
fields = [
    FieldSchema(name="abbreviation", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=SKILLS_DIM)
]

schema = CollectionSchema(fields, "jobs")

print(f"Create collection: jobs")
jobs_collection = Collection("jobs", schema, consistency_level="Strong")
start = time.time()
result = jobs_collection.insert(data)
print(f"Result of data insert: {result}")
end = time.time()
tot_duration = end - start
print(f"{num_vectors} vectors uploaded in {tot_duration} seconds")

