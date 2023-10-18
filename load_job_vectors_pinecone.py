import pinecone, os 
import pandas as pd 
import numpy as np
import time

pinecone.api_key = os.environ['PINECONE_API_KEY']
index_name = 'jobs'
env = os.environ['PINECONE_ENV']
BATCH_SIZE=1000

pinecone.init(api_key=pinecone.api_key, environment = env)
# Load the unique job IDs (job codes) from the CSV file
ids_path = './data/all_internal_job_title_desc.csv'
ids_df = pd.read_csv(ids_path)
ids = ids_df['job_code'].values

vectors = np.load('./data/jd_sem_vec.npy')
num_vectors = vectors.shape[0]

# Generate a list of dictionaries for upsert
upsert_data = [{"id": ids[i], "values": vectors[i].tolist()} for i in range(len(vectors))]

with pinecone.Index(index_name=index_name) as index:
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
    
