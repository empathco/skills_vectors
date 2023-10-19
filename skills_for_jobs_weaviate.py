import weaviate
import os 
import pandas as pd 
import numpy as np
import time

WEAVIATE_SERVER='https://skills-322ahpq1.weaviate.network'
MAX_JOBS=100

def save_job_skills(job_skills,filename):
    rows=[]
    for key,job in job_skills.items():
        skills=job['data']['Get']['Skill']
        row = {"job":key,
               "skill1":skills[0]['abbreviation'],"skill1_score":skills[0]['_additional']['distance'],
               "skill2":skills[1]['abbreviation'],"skill2_score":skills[1]['_additional']['distance'],
               "skill3":skills[2]['abbreviation'],"skill3_score":skills[2]['_additional']['distance'],
            }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filename)

resource_owner_config = weaviate.AuthClientPassword(
        username = os.environ['WEAVIATE_USER'],
        password = os.environ['WEAVIATE_PASSWORD'],
        scope = "offline_access" # optional, depends on the configuration of your identity provider (not required with WCS)
    )

client = weaviate.Client(
        url = WEAVIATE_SERVER,  
        auth_client_secret=resource_owner_config
    )

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
    start = time.time()
    result = (
        client.query
        .get("Skill", ["abbreviation", "title"])
        .with_near_vector({
            "vector": job_vec
        })
        .with_limit(3)
        .with_additional(["distance"])
        .do()
    )
    print(f"Result {result}")
    end = time.time()
    duration = end - start
    job_skills[job['job_code']]=result
    print (f"Query time: {duration} seconds")
    tot_duration += duration 
    num_queries += 1

avg_query_time = tot_duration / num_queries
print(f"Total query time {tot_duration} for {num_queries} seconds, average {avg_query_time} seconds")
save_job_skills(job_skills,'job_skills.csv')
