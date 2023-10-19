import weaviate
import os 
import pandas as pd 
import numpy as np
import time

MAX_JOBS=100

def save_job_skills(job_skills,filename):
    rows=[]
    for key,job in job_skills.items():
        row = {"job":key,
               "skill1":job.ids[0],"skill1_score":job.distances[0],
               "skill2":job.ids[1],"skill2_score":job.distances[1],
               "skill3":job.ids[2],"skill3_score":job.distances[2],
            }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filename)

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
token = os.environ['MILVUS_API_KEY']
uri = os.environ['MILVUS_URL']
connections.connect("default", uri=uri, token=token)

jobs_path = './data/all_internal_job_title_desc.csv'
jobs_df = pd.read_csv(jobs_path)
jobs_vectors = np.load('./data/jd_sem_vec.npy')

job_skills={}
tot_duration = 0 
num_queries = 0
for i, job in jobs_df.iterrows():
    if i >= MAX_JOBS:
        break 
    job_vec = jobs_vectors[i]
    print(f"Finding skills for job {job.loc['job_title']}")
    search_params = {
        "metric_type": "L2", 
        "offset": 5, 
        "ignore_growing": False, 
        "params": {"nprobe": 10}
    }
    skills_collection = Collection("skills")
    print(f"Skills collection: {skills_collection}")

    skills_collection.load(replica=2)
    # Check the loading progress and loading status
    utility.load_state("skills")
    # Output: <LoadState: Loaded>

    utility.loading_progress("skills")
    # Output: {'loading_progress': 100%}
    
    start = time.time()
    results = skills_collection.search(
        data=[job_vec], 
        anns_field="embeddings",
        # the sum of `offset` in `param` and `limit` 
        # should be less than 16384.
        param=search_params,
        limit=3,
        expr=None,
        # set the names of the fields you want to 
        # retrieve from the search result.
        output_fields=['id'],
        consistency_level="Strong"
    ) 
    end = time.time()
    duration = end - start
    print (f"Query time: {duration} seconds")
    tot_duration += duration 
    num_queries += 1
    job_skills[job['job_code']]=results[0]

avg_query_time = tot_duration / num_queries
print(f"Total query time {tot_duration} seconds for {num_queries} queries, average {avg_query_time} seconds")
save_job_skills(job_skills,'job_skills.csv')
