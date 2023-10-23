import psycopg2
import os 
import pandas as pd 
import numpy as np
import time

MAX_JOBS=10

def save_job_skills(job_skills,filename):
    rows=[]
    for key in job_skills:
        row = {"job":key}
        print(f"Job key {key}")
        skills= job_skills[key]
        print(f"Skills {skills}")
        for i in range(len(skills)):
            subkey1 = "skill"+str(i+1)
            row[subkey1] = skills[i][0]
            subkey2 = "skill"+str(i+1)+"_score"
            row[subkey2] = skills[i][1]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filename)

jobs_path = './data/all_internal_job_title_desc.csv'
jobs_df = pd.read_csv(jobs_path)
jobs_vectors = np.load('./data/jd_sem_vec.npy')

job_skills={}
tot_duration = 0 
num_queries = 0

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
    if i >= MAX_JOBS:
        break 
    job_vec = jobs_vectors[i]
    vec_list = job_vec.tolist()
    vec_str =  ", ".join(str(num) for num in vec_list)
    print(f"Vector string {vec_str}")
    print(f"Finding skills for job {job.loc['job_title']}")
    query = "SELECT abbreviation, embedding <=> '[" + vec_str +"]' AS score FROM skills ORDER BY score DESC LIMIT 3"
    print(f"Query {query}")
    start = time.time()
    cursor.execute(query)
    skills = cursor.fetchall()
    end = time.time()
    duration = end - start
    print(f"Query time: {duration} seconds")

    tot_duration += duration 
    num_queries += 1
    print(f"Skills {skills}")
    job_skills[job['job_code']]=skills
cursor.close()
conn.close()

avg_query_time = tot_duration / num_queries
print(f"Total query time {tot_duration} seconds for {num_queries} queries, average {avg_query_time} seconds")
save_job_skills(job_skills,'job_skills_pg.csv')
