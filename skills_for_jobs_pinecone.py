import pinecone, os 
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
    tot_quality=0
    for key,job in job_skills.items():
        count = 0 
        row = {"job":key}
        for i in range(MAX_SKILLS):
            #print(f"Matches {job['matches']}")
            if i< len(job['matches']):
                value = row["skill"+str(i)]=job['matches'][i]['id']
                if job_has_skill(key,value):
                    count += 1
        row["quality"] = count 
        tot_quality += count 
        rows.append(row)
    avg_quality = tot_quality/len(job_skills)
    print(f"Average quality {avg_quality}")
    df = pd.DataFrame(rows)
    df.to_csv(filename)

pinecone.api_key = os.environ['PINECONE_API_KEY']
skill_index_name = 'skills'
job_index_name = 'jobs'
env = os.environ['PINECONE_ENV']

pinecone.init(api_key=pinecone.api_key, environment = env)
skills_index = pinecone.Index('skills')
jobs_index = pinecone.Index('jobs')

jobs_path = './data/job_title_desc.csv'
jobs_df = pd.read_csv(jobs_path)
if (jobs_df.size == 0) or (jobs_df[jobs_df['org_name']==TARGET_ORG].size==0):
    print ("No jobs")
    exit()
jobs_vectors = np.load('./data/jd_sem_vec.npy')

job_skills={}
tot_duration = 0 

for i, job in jobs_df.iterrows():
    if job['org_name']!=TARGET_ORG: 
        continue
    if len(job_skills)>=MAX_JOBS:
        break
    job_vec = jobs_vectors[i]
    #print(f"Finding skills for job {job.loc['job_title']}")
    start = time.time()
    result = skills_index.query(vector=job_vec.tolist(),top_k=MAX_JOBS,include_values=True)
    #print(f"Result {result}")
    end = time.time()
    duration = end - start
    job_skills[job['job_title']]=result
    print (f"Query time: {duration} seconds")
    tot_duration += duration 

num_queries = len(job_skills)
avg_query_time = tot_duration / num_queries
print(f"Total query time {tot_duration} for {num_queries}, average {avg_query_time}")
save_job_skills(job_skills,'job_skills_pinecone.csv')
