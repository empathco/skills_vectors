import pinecone, weaviate, os 
import pandas as pd 
import numpy as np
import time

MAX_JOBS=10
TARGET_ORG='wg'
MAX_SKILLS=100
WEAVIATE_SERVER='https://skills-322ahpq1.weaviate.network'

labeled_job_skills = pd.read_csv('./data/labeled_job_skills.csv')

def job_has_skill(job_title,skill_id):
    #print(f"Looking for skills for job {job_title} to match {skill_id} ")
    for i,skill in labeled_job_skills.loc[labeled_job_skills['job_title']==job_title].iterrows():
        #print(f"Does skill {skill['abbreviation']}={skill_id}?")
        if skill['abbreviation']==skill_id:
            #print(f"Found abbreviation")
            return True 
    #print("Didn't find abbreviation")
    return False

def save_job_skills_pinecone(job_skills,filename):
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

def save_job_skills_weaviate(job_skills,filename):
    rows=[]
    tot_quality=0
    for key,job in job_skills.items():
        count=0
        row = {"job":key}
        skills=job['data']['Get']['Skill']
        i=0
        for skill in skills:
            value = row["skill"+str(i)]=skill['abbreviation'] 
            i+=1 
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
pinecone_skills_index = pinecone.Index('skills')

resource_owner_config = weaviate.AuthClientPassword(
        username = os.environ['WEAVIATE_USER'],
        password = os.environ['WEAVIATE_PASSWORD'],
        scope = "offline_access" # optional, depends on the configuration of your identity provider (not required with WCS)
    )

weaviate_client = weaviate.Client(
        url = WEAVIATE_SERVER,  
        auth_client_secret=resource_owner_config
    )

jobs_path = './data/job_title_desc.csv'
jobs_df = pd.read_csv(jobs_path)
if (jobs_df.size == 0) or (jobs_df[jobs_df['org_name']==TARGET_ORG].size==0):
    print ("No jobs")
    exit()
jobs_vectors = np.load('./data/jd_sem_vec.npy')

job_skills_pinecone={}
job_skills_weaviate={}
tot_durations = {'pinecone':0,'weaviate':0,'milvus':0,'pg':'0'}
for i, job in jobs_df.iterrows():
    if job['org_name']!=TARGET_ORG: 
        continue
    if len(job_skills_pinecone)>=MAX_JOBS:
        break
    job_vec = jobs_vectors[i]
    ######################
    # pinecone search
    #print(f"Finding skills for job {job.loc['job_title']}")
    start = time.time()
    result = pinecone_skills_index.query(vector=job_vec.tolist(),top_k=MAX_JOBS,include_values=True)
    #print(f"Result {result}")
    end = time.time()
    duration = end - start
    job_skills_pinecone[job['job_title']]=result
    print (f"Query time Pinecone: {duration} seconds")
    tot_durations['pinecone'] += duration 

    start = time.time()
    result = (
        weaviate_client.query
        .get("Skill", ["abbreviation", "title"])
        .with_near_vector({
            "vector": job_vec
        })
        .with_limit(MAX_SKILLS)
        .with_additional(["distance"])
        .do()
    )
    #print(f"Result {result}")
    end = time.time()
    duration = end - start
    job_skills_weaviate[job['job_title']]=result
    print (f"Query time Weaviate: {duration} seconds")

avg_query_time = tot_durations['pinecone'] / len(job_skills_pinecone)
print(f"Total query time {tot_durations['pinecone']}, average {avg_query_time}")
save_job_skills_pinecone(job_skills_pinecone,'job_skills_pinecone.csv')

avg_query_time = tot_durations['pinecone'] / len(job_skills_weaviate)
print(f"Total query time {tot_durations['weaviate']}, average {avg_query_time}")
save_job_skills_weaviate(job_skills_weaviate,'job_skills_weaviate.csv')

