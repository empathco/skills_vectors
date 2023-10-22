import os 
import pandas as pd 
import numpy as np
import time
import psycopg2
import psycopg2.extras as extras
SKILLS_DIM = 512
ids_path = './data/epl_skill_list_melted.csv'
ids_df = pd.read_csv(ids_path)
ids = ids_df['abbreviation'].values
vectors = np.load('./data/skill_vectors.npy')
num_vectors = vectors.shape[0]
host = os.environ['STACKHERO_POSTGRESQL_HOST']
db_name = "skills_vectors"
db_user = "admin"
db_password = os.environ['STACKHERO_POSTGRESQL_ADMIN_PASSWORD']
conn = psycopg2.connect(database=db_name,
                        host=host,
                        user=db_user,
                        password=db_password)
cursor = conn.cursor()
start = time.time()
try:
    for i,row in ids_df.iterrows():
      query = "INSERT INTO SKILLS(ABBREVIATION,EMBEDDING) VALUES('" + row[0] + "','[" + ','.join([str(j) for j in vectors[i]]) + "]')"
      cursor.execute(query)  
    conn.commit()
except (Exception, psycopg2.DatabaseError) as error:
    print("Error: %s" % error)
    conn.rollback()
    cursor.close()
print("execute_values() done")
cursor.close()
end = time.time()
duration = end - start
print(f"{num_vectors} vectors uploaded in {duration} seconds")