import pinecone, weaviate, os 
import pandas as pd 
import numpy as np
import time
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

MAX_JOBS=10
TARGET_ORG='wg'
MAX_SKILLS=10
WEAVIATE_SERVER='https://skills-322ahpq1.weaviate.network'

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
        prev_skill = None 
        for i in range(len(job['matches'])):
            #print(f"Matches {job['matches']}")
            if i< len(job['matches']):
                value = row["skill"+str(i)]=job['matches'][i]['id']
                if value == prev_skill:
                    continue
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
        prev_skill = None
        for skill in skills:
            value = row["skill"+str(i)]=skill['abbreviation'] 
            if value == prev_skill:
                continue
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

def save_job_skills_milvus(job_skills,filename):
    rows=[]
    tot_quality = 0
    for key,job in job_skills.items():
        row = {"job":key}
        i=0
        count = 0 
        #print(f"Job: {job}")
        for id in job.ids:
            value = row["skill"+str(i+1)] = id
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

token = os.environ['MILVUS_API_KEY']
uri = os.environ['MILVUS_URL']
token = "db_admin:"+ os.environ['MILVUS_PASSWORD']
uri = os.environ['MILVUS_URL']
connections.connect("default", uri=uri, token=token)
# TODO: create the Milvus index via API if it doesn't exist yet

skills_collection = Collection("skills")

labeled_job_skills = pd.read_csv('./data/labeled_job_skills.csv')

jobs_path = './data/job_title_desc.csv'
jobs_df = pd.read_csv(jobs_path)
if (jobs_df.size == 0) or (jobs_df[jobs_df['org_name']==TARGET_ORG].size==0):
    print ("No jobs are available.")
    exit()
jobs_vectors = np.load('./data/jd_sem_vec.npy')

job_skills_pinecone = {}
job_skills_weaviate = {}
job_skills_milvus = {}
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
    result = pinecone_skills_index.query(vector=job_vec.tolist(),top_k=MAX_SKILLS*5,include_values=True)
    #print(f"Result {result}")
    end = time.time()
    duration = end - start
    job_skills_pinecone[job['job_title']]=result
    print (f"Query time Pinecone: {duration} seconds")
    tot_durations['pinecone'] += duration 
    ##############################
    # weaviate search
    start = time.time()
    result = (
        weaviate_client.query
        .get("Skill", ["abbreviation", "title"])
        .with_near_vector({
            "vector": job_vec
        })
        .with_limit(MAX_SKILLS*5)
        .with_additional(["distance"])
        .do()
    )
    #print(f"Result {result}")
    end = time.time()
    duration = end - start
    job_skills_weaviate[job['job_title']]=result
    print (f"Query time Weaviate: {duration} seconds")
    tot_durations['weaviate'] += duration 

    ##############################
    # milvus search 
    search_params = {
        "metric_type": "L2", 
        "offset": 5, 
        "ignore_growing": False, 
        "params": {"nprobe": 10}
    }
    start = time.time()
    milvus_result = skills_collection.search(
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
    job_skills_milvus[job['job_title']]=milvus_result[0]
    print (f"Query time Milvus: {duration} seconds")
    tot_durations['milvus'] += duration 


# report on the search times
avg_query_time = tot_durations['pinecone'] / len(job_skills_pinecone)
print(f"Total query time {tot_durations['pinecone']}, average {avg_query_time}")
save_job_skills_pinecone(job_skills_pinecone,'job_skills_pinecone.csv')

avg_query_time = tot_durations['weaviate'] / len(job_skills_weaviate)
print(f"Total query time {tot_durations['weaviate']}, average {avg_query_time}")
save_job_skills_weaviate(job_skills_weaviate,'job_skills_weaviate.csv')

avg_query_time = tot_durations['milvus'] / len(job_skills_milvus)
print(f"Total query time {tot_durations['milvus']}, average {avg_query_time}")
save_job_skills_milvus(job_skills_milvus,'job_skills_milvus.csv')


