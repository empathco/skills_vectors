import pinecone, os 
import pandas as pd 
import numpy as np
import time

MAX_JOBS=100

def save_job_skills(job_skills,filename):
    rows=[]
    for key,job in job_skills.items():
        row = {"job":key,
               "skill1":job['matches'][0]['id'],"skill1_score":job['matches'][0]['score'],
               "skill2":job['matches'][1]['id'],"skill2_score":job['matches'][1]['score'],
               "skill3":job['matches'][2]['id'],"skill3_score":job['matches'][2]['score'],
            }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filename)

pinecone.api_key = os.environ['PINECONE_API_KEY']
skill_index_name = 'skills'
job_index_name = 'jobs'
env = os.environ['PINECONE_ENV']

pinecone.init(api_key=pinecone.api_key, environment = env)
skills_index = pinecone.Index('skills')
jobs_index = pinecone.Index('jobs')

jobs_path = './data/all_internal_job_title_desc.csv'
jobs_df = pd.read_csv(jobs_path)
jobs_vectors = np.load('./data/jd_sem_vec.npy')

job_skills={}
tot_duration = 0 
num_queries = 0
for i, job in jobs_df.iterrows():
    job_vec = jobs_vectors[i]
    print(f"Finding skills for job {job.loc['job_title']}")
    start = time.time()
    result = skills_index.query(vector=job_vec.tolist(),top_k=3,include_values=True)
    #print(f"Result {result}")
    end = time.time()
    duration = end - start
    job_skills[job['job_code']]=result
    print (f"Query time: {duration} seconds")
    tot_duration += duration 
    num_queries += 1
    if i >= MAX_JOBS:
        break 
avg_query_time = tot_duration / num_queries
print(f"Total query time {tot_duration} for {num_queries}, average {avg_query_time}")
save_job_skills(job_skills,'job_skills_pinecone.csv')
