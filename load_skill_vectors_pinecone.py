import pinecone, os 
import pandas as pd 
import numpy as np

pinecone.api_key = os.environ['PINECONE_API_KEY']
index_name = os.environ['PINECONE_SKILLS_INDEX']
env = os.environ['PINECONE_ENV']
BATCH_SIZE=1000

pinecone.init(api_key=pinecone.api_key, environment = env)
pinecone.list_indexes()
# Load the IDs from the CSV file
ids_path = './data/epl_skill_list_melted.csv'
ids_df = pd.read_csv(ids_path)
ids = ids_df['abbreviation'].values

vectors = np.load('./data/skill_vectors.npy')
num_vectors = vectors.shape[0]



# Generate a list of dictionaries for upsert
upsert_data = [{"id": ids[i], "values": vectors[i].tolist()} for i in range(len(vectors))]
print(f"{num_vectors} vectors uploading to the '{index_name}' index.")

with pinecone.Index(index_name=index_name) as index:
    index.upsert(vectors=upsert_data[:BATCH_SIZE],namespace='')

print(f"{num_vectors} vectors uploaded to the '{index_name}' index.")
