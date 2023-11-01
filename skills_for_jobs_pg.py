import psycopg2
import os 
import pandas as pd 
import numpy as np
import time

MAX_JOBS=10
TARGET_ORG='wg'
MAX_SKILLS=10

def save_job_skills(job_skills,filename):
    rows=[]
    for key,job in job_skills.items():
        row = {"job":key}
        skills=job_skills[key]
        i=0
        for skill in job:
            row["skill"+str(i)] = skill[0]
            i+=1
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filename)

jobs_path = './data/job_title_desc.csv'
jobs_df = pd.read_csv(jobs_path)
if (jobs_df.size == 0) or (jobs_df[jobs_df['org_name']==TARGET_ORG].size==0):
    print(f"No jobs")
    exit()

print("Loading vectors")

jobs_vectors = np.load('./data/jd_sem_vec.npy')

job_skills={}
tot_duration = 0 

host = os.environ['STACKHERO_POSTGRESQL_HOST']
db_name = "skills_vectors"
db_user = "admin"
db_password = os.environ['STACKHERO_POSTGRESQL_ADMIN_PASSWORD']
conn = psycopg2.connect(database=db_name,
                        host=host,
                        user=db_user,
                        password=db_password)
cursor = conn.cursor()

for i, job in jobs_df.iterrows():
    print(f"Processing job {job}")
    if job['org_name']!=TARGET_ORG: 
        continue
    if len(job_skills)>=MAX_JOBS:
        break
    job_vec = jobs_vectors[i]
    vec_list = job_vec.tolist()
    vec_str =  ",".join(str(num) for num in vec_list)
    print(f"Finding skills for job {job.loc['job_title']}")
    query = "SELECT abbreviation, embedding <=> '[" + vec_str +"]' AS score FROM skills ORDER BY score DESC LIMIT "+str(MAX_SKILLS)
    print(f"Query {query}")
    start = time.time()
    cursor.execute(query)
    skills = cursor.fetchall()
    end = time.time()
    duration = end - start
    print(f"Query time: {duration} seconds")
    tot_duration += duration 
    print(f"Skills {skills}")
    job_skills[job['org_job_code']]=skills
cursor.close()
conn.close()

num_queries = len(job_skills)
avg_query_time = tot_duration / num_queries
print(f"Total query time {tot_duration} seconds for {num_queries} queries, average {avg_query_time} seconds")
save_job_skills(job_skills,'job_skills_pg.csv')
