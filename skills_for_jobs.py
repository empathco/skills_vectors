import pinecone, weaviate, psycopg2, os 
import pandas as pd 
import numpy as np
import time
from numpy import dot
from numpy.linalg import norm
import ast 

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

MAX_JOBS = 5000 # all jobs
TARGET_ORG = 'wg'
MAX_SKILLS = 10
NUM_LISTS = 4

WEAVIATE_SERVER=os.environ['WEAVIATE_CLUSTER']

def init_pinecone():
    pinecone.api_key = os.environ['PINECONE_API_KEY']
    skill_index_name = 'skills'
    env = os.environ['PINECONE_ENV']
    pinecone.init(api_key=pinecone.api_key, environment = env)
    index = pinecone.Index(skill_index_name)
    return index

def init_weaviate():

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
    collection.load()
    return collection

def pinecone_search(index,job_vec):
    start = time.time()
    result = index.query(vector=job_vec.tolist(),top_k=MAX_SKILLS,include_values=True,include_metadata=True)
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
    result = None 
    try: 
        result = (
            client.query
            .get("Skill", ["abbreviation", "title","level"])
            .with_near_vector({
                "vector": job_vec
            })
            .with_limit(MAX_SKILLS*5)
            .with_additional(["distance","vector"])
            .do()
        )
    except Exception as e:
        print(f"Query failed: {e}")
        errors+=1

    end = time.time()
    duration = end - start
    job_skills_weaviate[job['job_title']]=result
    print (f"Query time Weaviate: {duration} seconds")
    tot_durations['weaviate'] += duration 
    return result

def milvus_search(collection,job_vec):
    if len(job_skills_milvus)>=MAX_JOBS:
        return None 
    search_params = {
        "metric_type": "L2", # this is actually the only option on the Zillig host
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
        output_fields=['id','embeddings','level'],
        consistency_level="Strong"
    ) 
    #print(f"Milvus search results {results}")
    end = time.time()
    duration = end - start
    result=results[0]
    print (f"Query time Milvus: {duration} seconds")
    tot_durations['milvus'] += duration 
    return result 

def pg_search(cursor,job_vec):
    vec_list = job_vec.tolist()
    vec_str =  ",".join(str(num) for num in vec_list)
    #print(f"Finding skills for job {job.loc['job_title']}")
    query = "SELECT abbreviation,level,embedding <=> '[" + vec_str +"]',embedding AS score FROM skills ORDER BY score DESC LIMIT "+str(MAX_SKILLS)
    #print(f"Query {query}")
    start = time.time()
    cursor.execute(query)
    skills = cursor.fetchall()
    end = time.time()
    duration = end - start
    print(f"Query time Postgres: {duration} seconds")
    tot_durations['pg'] += duration
    return skills 

def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def average(lst): 
    return sum(lst) / len(lst) 

def save_job_skills_pinecone(job_skills,best_vector,filename):
    #print(f"Pinecone: evaluating {len(job_skills)} skills")
    rows=[]
    for key,job in job_skills.items():
        count = 0 
        row = {"job":key}
        prev_skill = None 
        similarities=[]
        avg_similarities=[]
        for i in range(len(job['matches'])):
            #print(f"Matches {job['matches']}")
            if i< len(job['matches']):
                value = row["skill"+str(i)]=job['matches'][i]['id']
                row["level"+str(i)]=job['matches'][i]['metadata']['level']
                if value == prev_skill:
                    continue
                #print(f"Best vector {len(best_vector)}, current match vector {len(job['matches'][i]['values'])}")
                similarities.append(cos_sim(job['matches'][i]['values'],best_vector))
 
        avg_similarities.append(average(similarities))
        rows.append(row)
    print(f"Average similarity {average(avg_similarities)}")
    df = pd.DataFrame(rows)
    df.to_csv(filename)

def save_job_skills_weaviate(job_skills,best_vector,filename):
    #print(f"Weaviate: evaluating {len(job_skills)} skills")
    rows=[]
    avg_similarities=[]
    for key,job in job_skills.items():
        row = {"job":key}
        #print(f"Job {job}")
        skills=job['data']['Get']['Skill']
        i=0
        prev_skill = None
        similarities=[]
        for skill in skills:
            value = row["skill"+str(i)]=skill['abbreviation'] 
            row["level"+str(i)]=skill['level']
            if value == prev_skill:
                continue
            i+=1 
            similarities.append(cos_sim(skill['_additional']['vector'],best_vector))
        avg_similarities.append(average(similarities))
        rows.append(row)
    print(f"Average similarity {average(avg_similarities)}")
    df = pd.DataFrame(rows)
    df.to_csv(filename)

def save_job_skills_milvus(job_skills,best_vector,filename):
    rows=[]
    similarities=[]
    avg_similarities=[]
    for key,job in job_skills.items():
        row = {"job":key}
        #print(f"Job: {job}")
        if job is None:
            break 
        similarities=[]
        i=0
        for hit in job:
            if hit is None: 
                break
            value = row["skill"+str(i)] = hit.entity.id
            row["level"+str(i)]=hit.entity.get('level')
            i+=1
            similarities.append(cos_sim(hit.entity.embeddings,best_vector))
        avg_similarities.append(average(similarities))
        rows.append(row)
    print(f"Average similarity {average(avg_similarities)}")
    df = pd.DataFrame(rows)
    df.to_csv(filename)

def save_job_skills_pg(job_skills,best_vector,filename):
    rows=[]
    similarities = []
    avg_similarities=[]
    for key,job in job_skills.items():
        row = {"job":key}
        skills=job_skills[key]
        i=0
        for skill in skills:
            value = row["skill"+str(i)] = skill[0]
            row["level"]=skill[2]
            i+=1
            similarities.append(cos_sim(np.array(ast.literal_eval(skill[3])),best_vector))
        #print(f"Average similarities: {average(similarities)}")
        avg_similarities.append(average(similarities))
        rows.append(row)
    print(f"Average similarity {average(avg_similarities)}")
    df = pd.DataFrame(rows)
    df.to_csv(filename)

# use the Postgres exact nearest neighbor search to find the "right" answers 
# to evaluate the index Approximate Nearest Neighbor accuracy
def get_nearest_neighbor_skills(cursor,job_vec):
    vec_list = job_vec.tolist()
    vec_str =  ",".join(str(num) for num in vec_list)
    #print(f"Finding skills for job {job.loc['job_title']}")
    # setting probes to the numnber of lists forces Exact Nearest Neighbor search
    query = "BEGIN;SET LOCAL ivfflat.probes = "+str(NUM_LISTS)+";"
    query += "SELECT abbreviation,level,embedding <=> '[" + vec_str +"]' AS score,embedding FROM skills ORDER BY score DESC LIMIT "+str(MAX_SKILLS*10) 
    #query += ";COMMIT;"
    #print(f"Query {query}")
    start = time.time()
    cursor.execute(query)
    results = cursor.fetchall()
    end = time.time()
    nn_skills = []
    for result in results:
        if (len(nn_skills)>0) and result[0]==nn_skills[-1]:
            continue
        nn_skills.append(result)
        if len(nn_skills)>=MAX_SKILLS:
            break       
    duration = end - start
    print(f"Query time Postgres exact nearest neighbor search: {duration} seconds\n")
    cursor.execute("COMMIT")
    tot_durations['pg'] += duration
    #print(f"Best skills: {nn_skills}")
    closest_vector = np.array(ast.literal_eval(results[0][3]))
    #print(f"Closest vector {results[0][2]}")
    return nn_skills,closest_vector

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
job_skills_best = {}
tot_durations = {'pinecone':0,'weaviate':0,'milvus':0,'pg':0}

pinecone_skills_index = init_pinecone() 
weaviate_client = init_weaviate()
milvus_collection = init_milvus()
pg_cursor = init_pg() 
pinecone_errors = 0 
for i, job in jobs_df.iterrows():
    if job['org_name']!=TARGET_ORG: 
        continue
    if len(job_skills_pinecone)>=MAX_JOBS:
        break
    job_vec = jobs_vectors[i]

    result = pinecone_search(pinecone_skills_index,job_vec)
    if result is None:
        print("No result from Pinecone search")
        pinecone_errors+=1
    else: 
        job_skills_pinecone[job['job_title']] = result

    job_skills_weaviate[job['job_title']] = weaviate_search(weaviate_client,job_vec)
    job_skills_milvus[job['job_title']] = milvus_search(milvus_collection,job_vec)
    job_skills_pg[job['job_title']] = pg_search(pg_cursor,job_vec)  
    best_skills,best_vector = get_nearest_neighbor_skills(pg_cursor,job_vec) 
    job_skills_best[job['job_title']]= best_skills

df = pd.DataFrame(job_skills_best)
df.to_csv("job_skills_best.csv")

avg_query_time = tot_durations['pinecone'] / len(job_skills_pinecone)
print(f"Pinecone: total query time {tot_durations['pinecone']}, average {avg_query_time}, error count {pinecone_errors}")
save_job_skills_pinecone(job_skills_pinecone,best_vector,'job_skills_pinecone.csv')

avg_query_time = tot_durations['weaviate'] / len(job_skills_weaviate)
print(f"Weaviate: total query time {tot_durations['weaviate']}, average {avg_query_time}")
save_job_skills_weaviate(job_skills_weaviate,best_vector,'job_skills_weaviate.csv')

avg_query_time = tot_durations['milvus'] / len(job_skills_milvus)
print(f"Milvus: Total query time {tot_durations['milvus']}, average {avg_query_time}")
save_job_skills_milvus(job_skills_milvus,best_vector,'job_skills_milvus.csv')

avg_query_time = tot_durations['pg'] / len(job_skills_pg)
print(f"Postgres: Total query time {tot_durations['pg']}, average {avg_query_time}")
save_job_skills_pg(job_skills_pg,best_vector,'job_skills_pg.csv')

