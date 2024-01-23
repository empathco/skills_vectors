import os 
import pandas as pd 
import numpy as np
import time
import psycopg2
import psycopg2.extras as extras
import sys 

provider = sys.argv[1]
if provider == "openai":
    SKILLS_DIM = 1536
else:
    SKILLS_DIM = 768

NUM_LISTS = 4 
ids_path = './data/skill_list.csv'
ids_df = pd.read_csv(ids_path)
ids = ids_df['abbreviation'].values
levels = ids_df['level'].values
skill_vectors_file = "./data/" + provider + "_skill_vectors.npy"
vectors = np.load(skill_vectors_file)
print(f"Shape of skill_vectors.npy {vectors.shape}")
num_vectors = vectors.shape[0]

DATABASE_URL = os.environ['DATABASE_URL']
conn = psycopg2.connect(DATABASE_URL, sslmode='require')

cursor = conn.cursor()
table_name=provider+"_skills"
cmd = "DELETE FROM "+table_name+";"
cursor.execute(cmd)

# search for skills
start = time.time()
try:
    for i,row in ids_df.iterrows():
      #print(f"Row {row}")
      query = "INSERT INTO "+table_name+"(ABBREVIATION,EMBEDDING,LEVEL,PROVIDER) VALUES('" + row.iloc[0] + "','[" + ','.join([str(j) for j in vectors[i]]) + "]',"+str(row['level']) + ",'" + provider +"');"
      cursor.execute(query)  
    conn.commit()
except (Exception, psycopg2.DatabaseError) as error:
    print("Error: %s" % error)
    conn.rollback()
end = time.time()
duration = end - start
print(f"{num_vectors} vectors uploaded in {duration} seconds")

# create index for skills
start = time.time()
try:
    cmd = "CREATE INDEX IF NOT EXISTS SKILLS_INDEX ON "+ table_name +" USING ivfflat (embedding vector_cosine_ops) WITH (lists = "+ str(NUM_LISTS) + ");"
    cursor.execute(cmd)  
    conn.commit()
except (Exception, psycopg2.DatabaseError) as error:
    print("Error: %s" % error)
    conn.rollback()
end = time.time()
duration = end - start
print(f"Index created in {duration} seconds")
cursor.close()