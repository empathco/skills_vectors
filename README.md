# skills_vectors
This project demonstrates storing and searching skills and jobs vectors in various vector databases:
* Find skills related to job descriptions (semantic search)
* Uploading to Pinecone, Weaviate, Milvus, Postgres
* Given a job find related skills using semantic similarity (vector distance)
* Good broad, “real world complexity” yet replicable test of data loading and semantic search 
* Code flavor is important for devs (at least for me)

## Loading Skills Vectors
First load the skills vectors to your DB of choice. 

### Pinecone
    python3 load_skill_vectors_pinecone.py

### Weaviate
    python load_skill_vectors_weaviate.py

### Milvus
    python load_skill_vectors_milvus.py

### pgvector
    python load_skill_vectors_pg.py

## Performing Semantic Search
Then perform semantic search with the skills_for_jobs*.py scripts

### Pinecone
    python3 skills_for_jobs_pinecone.py

### Weaviate
    python skills_for_jobs_weaviate.py

### Milvus
    python skills_for_jobs_milvus.py

### pgvector
    python skills_for_jobs_pg.py

## Configure Environment Variables 
Set the following env vars to access and use the various vector DBs.

    export PINECONE_API_KEY=''
    export PINECONE_ENV='us-west1-gcp'

    export WEAVIATE_USER=''
    export WEAVIATE_PASSWORD=''
    export MILVUS_URL='https://'
    export MILVUS_API_KEY=''

    export STACKHERO_POSTGRESQL_HOST=''export STACKHERO_POSTGRESQL_ADMIN_PASSWORD=''

## Create Skills and Jobs Files 
Grab some text for skill descriptions and job descriptions from your site of choice (or have ChatGPT write them)

### ./data/job_title_desc.csv 
    job_code,org_name,org_job_code,job_title,jd_plain_text,processed_title
    your text here

### jd_and_title_sem_vec.npy 
    put your embeddings here one to a line 

### ./data/epl_skill_list.csv 
    abbreviation,title,text_type,content

### skill_vectors.npy 
    embeddings from your LLM of choice

