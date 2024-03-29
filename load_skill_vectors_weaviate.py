import weaviate
import os 
import pandas as pd 
import numpy as np
import time
import requests 
import sys 

WEAVIATE_SERVER=os.environ['WEAVIATE_CLUSTER']
BATCH_SIZE = 100

def cs_check_batch_result(results):
    """
    Check batch results for errors.
    Parameters
    ----------
    results : dict
        The Weaviate batch creation return value.
    """

    if results is None:
        return
    for result in results:
        if "result" in result and "errors" in result["result"]:
            if "error" in result["result"]["errors"]:
                print(result["result"]["errors"])
                raise Exception(result["result"]["errors"])
            
provider = sys.argv[1]
if provider == "openai":
  SKILLS_DIM = 1536
else:
  SKILLS_DIM = 768

resource_owner_config = weaviate.AuthClientPassword(
        username = os.environ['WEAVIATE_USER'],
        password = os.environ['WEAVIATE_PASSWORD'],
        scope = "offline_access" # optional, depends on the configuration of your identity provider (not required with WCS)
    )

client = weaviate.Client(
        url = WEAVIATE_SERVER,  
        auth_client_secret=resource_owner_config
    )

print("Deleting existing skills")
result=client.batch.delete_objects(
    class_name='Skill',
    where={
        'path': ['id'],
        'operator': 'Like',
        'valueText': '*'
    },
)
print(f"Deleted skills {result}")

# Load the IDs from the CSV file
ids_path = './data/skill_list.csv'
ids_df = pd.read_csv(ids_path)
ids = ids_df['abbreviation'].values

skill_vectors_file = "./data/" + provider + "_skill_vectors.npy"
vectors = np.load(skill_vectors_file)
num_vectors = vectors.shape[0]

# Generate a list of dictionaries for upsert
upsert_data = [{"id": ids[i], "values": vectors[i].tolist()} for i in range(len(vectors))]
 
start = time.time()
print(f"{num_vectors} vectors uploading...")
class_name = provider + "_Skill"
with client.batch(batch_size = BATCH_SIZE, callback=cs_check_batch_result) as batch:
    for i,v in enumerate(vectors):
        properties = {
            "abbreviation": ids_df['abbreviation'][i],
            "title": ids_df['title'][i],
            "level_description": ids_df['level_description'][i],
            "level": str(ids_df['level'][i]),
            "provider": provider
        }
        client.batch.add_data_object(properties,class_name=class_name,vector=v)
end = time.time()
tot_duration = end - start
print(f"{num_vectors} vectors uploaded in {tot_duration} seconds")

