import pinecone, weaviate, psycopg2, os 
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

def init_pinecone():
    pinecone.api_key = os.environ['PINECONE_API_KEY']
    skill_index_name = 'skills'
    env = os.environ['PINECONE_ENV']
    pinecone.init(api_key=pinecone.api_key, environment = env)
    index = pinecone.Index(skill_index_name)
    return index

def init_weaviate():
    WEAVIATE_SERVER='https://skills-322ahpq1.weaviate.network'
    resource_owner_config = weaviate.AuthClientPassword(
        username = os.environ['WEAVIATE_USER'],
        password = os.environ['WEAVIATE_PASSWORD'],
        scope = "offline_access" # optional, depends on the configuration of your identity provider (not required with WCS)
    )
    client = weaviate.Client(
        url = WEAVIATE_SERVER,  
        auth_client_secret=resource_owner_config
    )
    return client 

def init_pg():
    host = os.environ['STACKHERO_POSTGRESQL_HOST']
    db_name = "skills_vectors"
    db_user = "admin"
    db_password = os.environ['STACKHERO_POSTGRESQL_ADMIN_PASSWORD']
    conn = psycopg2.connect(database=db_name,
                            host=host,
                            user=db_user,
                            password=db_password)
    cursor = conn.cursor()
    return cursor 

def init_milvus():
    token = os.environ['MILVUS_API_KEY']
    uri = os.environ['MILVUS_URL']
    token = "db_admin:"+ os.environ['MILVUS_PASSWORD']
    uri = os.environ['MILVUS_URL']
    connections.connect("default", uri=uri, token=token)
    collection = Collection("skills")
    return collection

def pinecone_search(index,job_vec):
    start = time.time()
    result = index.query(vector=job_vec.tolist(),top_k=MAX_SKILLS*5,include_values=True)
    #print(f"Result {result}")
    end = time.time()
    duration = end - start

    job_skills_pinecone[job['job_title']]=result
    print (f"Query time Pinecone: {duration} seconds")
    tot_durations['pinecone'] += duration 
    return result

def weaviate_search(client,job_vec):
    if len(job_skills_weaviate)>=MAX_JOBS:
        return None 
    start = time.time()
    result = (
        client.query
        .get("Skill", ["abbreviation", "title"])
        .with_near_vector({
            "vector": job_vec
        })
        .with_limit(MAX_SKILLS*5)
        .with_additional(["distance"])
        .do()
    )
    end = time.time()
    duration = end - start
    job_skills_weaviate[job['job_title']]=result
    print (f"Query time Weaviate: {duration} seconds")
    tot_durations['weaviate'] += duration 
    return result

def milvus_search(collection,job_vec):
    if len(job_skills_weaviate)>=MAX_JOBS:
        return None 
    search_params = {
        "metric_type": "L2", 
    }
    start = time.time()
    results = collection.search(
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
    result=results[0]
    print (f"Query time Milvus: {duration} seconds")
    tot_durations['milvus'] += duration 
    #print(f"Milvus result {results}")
    return result 

def pg_search(cursor,job_vec):
    vec_list = job_vec.tolist()
    vec_str =  ",".join(str(num) for num in vec_list)
    #print(f"Finding skills for job {job.loc['job_title']}")
    query = "SELECT abbreviation, embedding <=> '[" + vec_str +"]' AS score FROM skills ORDER BY score DESC LIMIT "+str(MAX_SKILLS)
    #print(f"Query {query}")
    start = time.time()
    cursor.execute(query)
    skills = cursor.fetchall()
    end = time.time()
    duration = end - start
    print(f"Query time Postgres: {duration} seconds")
    tot_durations['pg'] += duration
    return skills 

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
    #print(f"Looping through {len(job_skills)} items")
    for key,job in job_skills.items():
        count=0
        row = {"job":key}
        #print(f"Job {job}")
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
        if job is None:
            break 
        for hit in job:
            if hit is None: 
                break
            if hit: 
                #print(f"Hit {hit}")
                value = row["skill"+str(i+1)] = hit.id
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

def save_job_skills_pg(job_skills,filename):
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
    print ("No jobs are available.")
    exit()
jobs_vectors = np.load('./data/jd_sem_vec.npy')

job_skills_pinecone = {}
job_skills_weaviate = {}
job_skills_milvus = {}
job_skills_pg = {}
tot_durations = {'pinecone':0,'weaviate':0,'milvus':0,'pg':0}

pinecone_skills_index = init_pinecone() 
weaviate_client = init_weaviate()
milvus_collection = init_milvus()
pg_cursor = init_pg() 
for i, job in jobs_df.iterrows():
    if job['org_name']!=TARGET_ORG: 
        continue
    job_vec = jobs_vectors[i]
    if len(job_skills_pinecone)>=MAX_JOBS:
        break
    job_skills_pinecone[job['job_title']] = pinecone_search(pinecone_skills_index,job_vec)
    job_skills_weaviate[job['job_title']] = weaviate_search(weaviate_client,job_vec)
    job_skills_milvus[job['job_title']] = milvus_search(milvus_collection,job_vec)
    job_skills_pg[job['job_title']] = pg_search(pg_cursor,job_vec)   

# save the job skills alignments and report on the search times and quality
labeled_job_skills = pd.read_csv('./data/labeled_job_skills.csv')
avg_query_time = tot_durations['pinecone'] / len(job_skills_pinecone)
print(f"Pinecone: total query time {tot_durations['pinecone']}, average {avg_query_time}")
save_job_skills_pinecone(job_skills_pinecone,'job_skills_pinecone.csv')

avg_query_time = tot_durations['weaviate'] / len(job_skills_weaviate)
print(f"Weaviate: total query time {tot_durations['weaviate']}, average {avg_query_time}")
save_job_skills_weaviate(job_skills_weaviate,'job_skills_weaviate.csv')

avg_query_time = tot_durations['milvus'] / len(job_skills_milvus)
print(f"Milvus: Total query time {tot_durations['milvus']}, average {avg_query_time}")
save_job_skills_milvus(job_skills_milvus,'job_skills_milvus.csv')

avg_query_time = tot_durations['pg'] / len(job_skills_pg)
print(f"Postgres: Total query time {tot_durations['pg']}, average {avg_query_time}")
save_job_skills_milvus(job_skills_milvus,'job_skills_pg.csv')