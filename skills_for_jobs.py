import pinecone, weaviate, psycopg2, os 
from pinecone import Pinecone
import pandas as pd 
import numpy as np
import time
from numpy import dot
from numpy.linalg import norm
import ast
import sys 

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

from qdrant_client import QdrantClient

from qdrant_client.models import Filter, FieldCondition, FilterSelector, MatchValue

MAX_JOBS = 5000 # all jobs
MAX_SKILLS = 10
NUM_LISTS = 4

WEAVIATE_SERVER=os.environ['WEAVIATE_CLUSTER']

def init_pinecone(provider="gemini"):
    api_key = os.environ['PINECONE_API_KEY']
    skill_index_name = 'skills-'+ provider
    env = os.environ['PINECONE_ENV']
    pc=Pinecone(api_key=api_key, environment = env)
    index = pc.Index(skill_index_name)
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

def init_milvus():
    token = os.environ['MILVUS_API_KEY']
    uri = os.environ['MILVUS_URL']
    token = "db_admin:"+ os.environ['MILVUS_PASSWORD']
    uri = os.environ['MILVUS_URL']
    connections.connect("default", uri=uri, token=token)
    collection_name = provider + "_skills"
    collection = Collection(collection_name)
    collection.load()
    return collection

def init_pg():
    DATABASE_URL = os.environ['DATABASE_URL']
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cursor = conn.cursor()
    return cursor 

def init_qdrant():
    client = QdrantClient(
        url=os.environ['QDRANT_URL'], 
        api_key=os.environ['QDRANT_API_KEY']
    )
    return client 

def pinecone_search(index,job_vec,provider="gemini"):
    start = time.time()
    try:
        result = index.query(vector=job_vec.tolist(),top_k=MAX_SKILLS,include_values=True,include_metadata=True)
        job_skills_pinecone[job['job_code']]=result
    except Exception as e:
        print(f"Pinecone query error {e}")
        result = None
    end = time.time()
    duration = end - start
    print (f"Query time Pinecone: {duration} seconds")
    tot_durations['pinecone'] += duration 
    return result

def weaviate_search(client,job_vec,provider="gemini"):
    if len(job_skills_weaviate)>=MAX_JOBS:
        return None 
    start = time.time()
    result = None 
    errors = 0
    try: 
        class_name=provider+"_Skill"
        result = (
            client.query
            .get(class_name, ["abbreviation", "title","level"])
            .with_near_vector({
                "vector": job_vec
            })
            .with_limit(MAX_SKILLS)
            .with_additional(["distance","vector"])
            .do()
        )
    except Exception as e:
        print(f"Query failed: {e}")
        errors+=1

    end = time.time()
    duration = end - start
    #print(f"Weaviate search result {result}")
    job_skills_weaviate[job['job_code']]=result
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

def pg_search(cursor,job_vec,provider):
    vec_list = job_vec.tolist()
    vec_str =  ",".join(str(num) for num in vec_list)
    #print(f"Finding skills for job {job.loc['job_code']}")
    table_name=provider+"_skills"
    query = "SELECT abbreviation,level,embedding <=> '[" + vec_str +"]',embedding AS score FROM "+table_name+" ORDER BY score DESC LIMIT "+str(MAX_SKILLS)
    #print(f"Query {query}")
    start = time.time()
    cursor.execute(query)
    skills = cursor.fetchall()
    end = time.time()
    duration = end - start
    print(f"Query time Postgres: {duration} seconds")
    tot_durations['pg'] += duration
    return skills 

def qdrant_search(client,job_vec,provider="gemini"):
    collection_name=provider+"-skills"
    start = time.time()   
    try: 
        results = client.search(collection_name=collection_name,query_vector=job_vec,with_payload=True,with_vectors=True, limit=MAX_SKILLS)
    except Exception as e:
        print(f"Failed Qdrant search {e}")
        results = None
    end = time.time()
    duration = end - start
    print(f"Query time Qdrant: {duration} seconds")
    tot_durations['qdrant'] += duration
    return results

def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def average(lst): 
    return sum(lst) / len(lst) 

def save_job_skills_pinecone(job_skills,best_vector,filename,job_skills_best=None):
    #print(f"Pinecone: evaluating {len(job_skills)} skills")
    rows=[]
    all_matches = 0 
    avg_similarities=[]
    for key,job in job_skills.items():
        count = 0 
        row = {"job":key}
        similarities=[]
        skill_matches = 0 
        skill_names = []
        for i in range(len(job['matches'])):
            #print(f"Matches {job['matches']}")
            if i< len(job['matches']):
                value = row["skill"+str(i)]=job['matches'][i]['id']
                if value in skill_names: # don't add same skill twice
                    continue
                skill_names.append(value)
                if job_skills_best and (value in job_skills_best[key]):
                    skill_matches += 1
                row["level"+str(i)]=job['matches'][i]['metadata']['level']

                #print(f"Best vector {len(best_vector)}, current match vector {len(job['matches'][i]['values'])}")
                similarities.append(cos_sim(job['matches'][i]['values'],best_vector))
        all_matches += skill_matches   
        avg_similarities.append(average(similarities))
        rows.append(row)

    avg_matches = all_matches/len(job_skills)
    print(f"All matches {all_matches}, average {avg_matches}")
    print(f"Pinecone average similarity {average(avg_similarities)}")
    df = pd.DataFrame(rows)
    df.to_csv(filename)

def save_job_skills_weaviate(job_skills,best_vector,filename,job_skills_best=None,provider="gemini"):
    #print(f"Weaviate: evaluating {len(job_skills)} skills")
    rows=[]
    avg_similarities=[]
    all_matches = 0 
    if provider == "openai":
        class_name = "Openai_Skill"
    else:
        class_name = "Gemini_Skill"
    for key,job in job_skills.items():
        row = {"job":key}
        #print(f"Job data {job['data']}")
        skills=job['data']['Get'][class_name]
        i=0
        similarities=[]
        skill_matches=0
        #print(f"Number of skills {len(skills)}")
        skill_names=[]
        for skill in skills:
            value = row["skill"+str(i)]=skill['abbreviation']
            #print(f"Skill {value}") 
            if value in skill_names:
                #print(f"Duplicate skill {value}")
                continue
            skill_names.append(value)
            if job_skills_best and (value in job_skills_best[key]):
                skill_matches += 1
            row["level"+str(i)]=skill['level']
            i+=1 
            similarities.append(cos_sim(skill['_additional']['vector'],best_vector))
        if len(similarities)>0:
            all_matches += skill_matches 
            #print(f"Similarities {similarities} ({len(similarities)})") 
            avg_similarities.append(average(similarities))
        rows.append(row)

    avg_matches = all_matches/len(job_skills)
    print(f"All matches {all_matches}, average {avg_matches}")
    if len(avg_similarities)>0:
        print(f"Weaviate average similarity {average(avg_similarities)}")
    df = pd.DataFrame(rows)
    df.to_csv(filename)

def save_job_skills_milvus(job_skills,best_vector,filename,job_skills_best=None):
    rows=[]
    avg_similarities=[]
    all_matches = 0 
    for key,job in job_skills.items():
        row = {"job":key}
        #print(f"Job: {job}")
        if job is None:
            break 
        i=0
        skill_matches = 0 
        similarities=[]
        skill_names=[]
        for hit in job:
            if hit is None: 
                break
            value = row["skill"+str(i)] = hit.entity.id
            if value in skill_names:
                continue
            skill_names.append(value )
            if job_skills_best and (value in job_skills_best[key]):
                skill_matches += 1
            row["level"+str(i)]=hit.entity.get('level')
            i+=1
            similarities.append(cos_sim(hit.entity.embeddings,best_vector))
        all_matches += skill_matches    
        avg_similarities.append(average(similarities))
        rows.append(row)

    avg_matches = all_matches/len(job_skills)
    print(f"All matches {all_matches}, average {avg_matches}")
    print(f"Milvus average similarity {average(avg_similarities)}")
    df = pd.DataFrame(rows)
    df.to_csv(filename)

def save_job_skills_pg(job_skills,best_vector,filename,job_skills_best=None):
    rows=[]
    avg_similarities=[]
    all_matches = 0 
    for key,job in job_skills.items():
        row = {"job":key}
        skills=job_skills[key]
        i=0
        skill_matches = 0 
        skill_names = []
        similarities=[]
        for skill in skills:
            value = row["skill"+str(i)] = skill[0]
            if value in skill_names:
                continue
            skill_names.append(value)
            if job_skills_best and (value in job_skills_best[key]):
                skill_matches += 1
            row["level"+str(i)]=skill[1]
            i+=1
            similarities.append(cos_sim(np.array(ast.literal_eval(skill[3])),best_vector))
        all_matches += skill_matches    
        avg_similarities.append(average(similarities))
        rows.append(row)

    avg_matches = all_matches/len(job_skills)
    print(f"All matches {all_matches}, average {avg_matches}")
    print(f"Postgres average similarity {average(avg_similarities)}")
    df = pd.DataFrame(rows)
    df.to_csv(filename)

def save_job_skills_qdrant(job_skills,best_vector,filename,job_skills_best=None):
    rows=[]
    avg_similarities=[]
    i=0
    all_matches = 0 
    for key,value in job_skills.items():
        row={"job":key}
        skill_matches = 0 
        skill_names = []
        similarities=[]
        for skill in value: 
            value = row["skill"+str(i)] = skill.payload['abbreviation']
            if value in skill_names:
                continue
            skill_names.append(value)
            if job_skills_best and (value in job_skills_best[key]):
                skill_matches += 1
            row["level"+str(i)] = skill.payload['l']
            i+=1
            similarities.append(cos_sim(skill.vector,best_vector))
        all_matches += skill_matches        
        avg_similarities.append(average(similarities))
        rows.append(row)

    avg_matches = all_matches/len(job_skills)
    print(f"All matches {all_matches}, average {avg_matches}")
    print(f"Qdrant average similarity {average(avg_similarities)}")
    df = pd.DataFrame(rows)
    df.to_csv(filename)  

# use the Postgres exact nearest neighbor search to find the "right" answers 
# to evaluate the index Approximate Nearest Neighbor accuracy
def get_nearest_neighbor_skills(cursor,job_vec,provider):
    vec_list = job_vec.tolist()
    vec_str =  ",".join(str(num) for num in vec_list)
    #print(f"Finding skills for job {job.loc['job_code']}")
    # setting probes to the numnber of lists forces Exact Nearest Neighbor search
    table_name=provider+"_skills"
    #query = "BEGIN;SET LOCAL ivfflat.probes = "+str(NUM_LISTS)+";"
    query = "SELECT abbreviation,level,embedding <=> '[" + vec_str +"]' AS score,embedding FROM "+table_name+" ORDER BY score DESC LIMIT "+str(MAX_SKILLS*10) 
    #query += ";COMMIT;"
    #print(f"Query {query}")
    start = time.time()
    cursor.execute(query)
    results = cursor.fetchall()
    end = time.time()
    nn_skills = []
    skill_names = []
    for result in results:
        value = result[0]
        if value in skill_names:
            continue
        skill_names.append(value)
        nn_skills.append(result)
        if len(nn_skills)>=MAX_SKILLS:
            break       
    duration = end - start
    print(f"Query time Postgres exact nearest neighbor search: {duration} seconds\n")
    #cursor.execute("COMMIT")
    tot_durations['best'] += duration
    #print(f"Best skills: {nn_skills}")
    closest_vector = np.array(ast.literal_eval(results[0][3]))
    print(f"Closest vector {results[0][2]}")
    return nn_skills,closest_vector

if len(sys.argv)<2:
    provider="gemini"
else:
    provider = sys.argv[1]

jobs_path = './data/generic_job_list.csv'
jobs_df = pd.read_csv(jobs_path)
if jobs_df.size == 0:
    print ("No jobs are available.")
    exit()
job_embeddings = './data/generic_job_desc_' + provider + '.npy' 
jobs_vectors = np.load(job_embeddings)

job_skills_pinecone = {}
job_skills_weaviate = {}
job_skills_milvus = {}
job_skills_pg = {}
job_skills_qdrant = {}
job_skills_best = {}
tot_durations = {'pinecone':0,'weaviate':0,'milvus':0,'pg':0,'qdrant':0,'best':0}

pinecone_skills_index = init_pinecone(provider) 
weaviate_client = init_weaviate()
milvus_collection = init_milvus()
pg_cursor = init_pg() 
qdrant_client = init_qdrant()
pinecone_errors = 0 
for i, job in jobs_df.iterrows():
    if len(job_skills_pinecone)>=MAX_JOBS:
        break
    job_vec = jobs_vectors[i]

    result = pinecone_search(pinecone_skills_index,job_vec,provider)
    if result is None:
        print("No result from Pinecone search")
        pinecone_errors+=1
    else: 
        job_skills_pinecone[job['job_code']] = result

    job_skills_weaviate[job['job_code']] = weaviate_search(weaviate_client,job_vec,provider)
    job_skills_milvus[job['job_code']] = milvus_search(milvus_collection,job_vec)
    job_skills_pg[job['job_code']] = pg_search(pg_cursor,job_vec,provider)  
    job_skills_qdrant[job['job_code']] = qdrant_search(qdrant_client,job_vec,provider)
    best_skills,best_vector = get_nearest_neighbor_skills(pg_cursor,job_vec,provider) 
    job_skills_best[job['job_code']]= best_skills

avg_query_time = tot_durations['pinecone'] / len(job_skills_pinecone)
print(f"Pinecone: total query time {tot_durations['pinecone']}, average {avg_query_time}, error count {pinecone_errors}")
save_job_skills_pinecone(job_skills_pinecone,best_vector,'job_skills_pinecone'+provider+'.csv',job_skills_best)

avg_query_time = tot_durations['weaviate'] / len(job_skills_weaviate)
print(f"Weaviate: total query time {tot_durations['weaviate']}, average {avg_query_time}")
save_job_skills_weaviate(job_skills_weaviate,best_vector,'job_skills_weaviate_'+provider+'.csv',job_skills_best,provider)

avg_query_time = tot_durations['milvus'] / len(job_skills_milvus)
print(f"Milvus: Total query time {tot_durations['milvus']}, average {avg_query_time}")
save_job_skills_milvus(job_skills_milvus,best_vector,'job_skills_milvus_'+provider+'.csv',job_skills_best)

avg_query_time = tot_durations['pg'] / len(job_skills_pg)
print(f"Postgres: Total query time {tot_durations['pg']}, average {avg_query_time}")
save_job_skills_pg(job_skills_pg,best_vector,'job_skills_pg_'+provider+'.csv',job_skills_best)

avg_query_time = tot_durations['qdrant'] / len(job_skills_qdrant)
print(f"Qdrant: Total query time {tot_durations['qdrant']}, average {avg_query_time}")
save_job_skills_qdrant(job_skills_qdrant,best_vector,'job_skills_qdrant_'+provider+'.csv',job_skills_best)

avg_query_time = tot_durations['best'] / len(job_skills_best)
print(f"Best skills with Postgres ENN: Total query time {tot_durations['qdrant']}, average {avg_query_time}")
save_job_skills_pg(job_skills_best,best_vector,'job_skills_best_'+provider+'.csv')


