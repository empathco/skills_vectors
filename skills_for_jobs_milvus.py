import weaviate
import os 
import pandas as pd 
import numpy as np
import time

MAX_JOBS=10
TARGET_ORG='wg'
MAX_SKILLS=10

labeled_job_skills = pd.read_csv('./data/labeled_job_skills.csv')

def job_has_skill(job_title,skill_id):
    print(f"Looking for skills for job {job_title} to match {skill_id} ")
    for i,skill in labeled_job_skills.loc[labeled_job_skills['job_title']==job_title].iterrows():
        #print(f"Skill {skill}")
        if skill['abbreviation']==skill_id:
            print(f"Found abbreviation")
            return True 
    #print("Didn't find abbreviation")
    return False

def save_job_skills(job_skills,filename):
    rows=[]
    tot_quality = 0
    for key,job in job_skills.items():
        row = {"job":key}
        i=0
        count = 0 
        for id in job.ids:
            value = row["skill"+str(i+1)] = id
            i+=1
            if job_has_skill(key,value):
                count += 1
        row["quality"] = count 
        tot_quality += count 
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
#just API key at Starter level, token = os.environ['MILVUS_API_KEY']
# this is how you connect with Standard (not Starter) level
token = "db_admin:"+ os.environ['MILVUS_PASSWORD']
uri = os.environ['MILVUS_URL']
connections.connect("default", uri=uri, token=token)

jobs_path = './data/job_title_desc.csv'
jobs_df = pd.read_csv(jobs_path)
if (jobs_df.size == 0) or (jobs_df[jobs_df['org_name']==TARGET_ORG].size==0):
    print ("No jobs")
    exit()
jobs_vectors = np.load('./data/jd_sem_vec.npy')

job_skills={}
tot_duration = 0 
skills_collection = Collection("skills")
print(f"Skills collection: {skills_collection}")

skills_collection.load(replica=2)
# Check the loading progress and loading status
utility.load_state("skills")
# Output: <LoadState: Loaded>

utility.loading_progress("skills")
# Output: {'loading_progress': 100%}
 
for i, job in jobs_df.iterrows():
    if job['org_name']!=TARGET_ORG: 
        continue
    if len(job_skills)>=MAX_JOBS:
        break
    job_vec = jobs_vectors[i]
    print(f"Finding skills for job {job.loc['job_title']}")
    search_params = {
        "metric_type": "L2", 
        "offset": 5, 
        "ignore_growing": False, 
        "params": {"nprobe": 10}
    }
    start = time.time()
    results = skills_collection.search(
        data=[job_vec], 
        anns_field="embeddings",
        # the sum of `offset` in `param` and `limit` 
        # should be less than 16384.
        param=search_params,
        limit=MAX_SKILLS,
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
    job_skills[job['job_code']]=results[0]

num_queries = len(job_skills)
avg_query_time = tot_duration / num_queries
print(f"Total query time {tot_duration} seconds for {num_queries} queries, average {avg_query_time} seconds")
save_job_skills(job_skills,'job_skills_milvus.csv')
