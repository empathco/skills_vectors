import weaviate
import os 
import pandas as pd 
import numpy as np
import time

WEAVIATE_SERVER='https://skills-322ahpq1.weaviate.network'
BATCH_SIZE = 10

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

resource_owner_config = weaviate.AuthClientPassword(
        username = os.environ['WEAVIATE_USER'],
        password = os.environ['WEAVIATE_PASSWORD'],
        scope = "offline_access" # optional, depends on the configuration of your identity provider (not required with WCS)
    )

client = weaviate.Client(
        url = WEAVIATE_SERVER,  
        auth_client_secret=resource_owner_config
    )

# Load the IDs from the CSV file
ids_path = './data/epl_skill_list_melted.csv'
ids_df = pd.read_csv(ids_path)
ids = ids_df['abbreviation'].values

vectors = np.load('./data/skill_vectors.npy')
num_vectors = vectors.shape[0]

# Generate a list of dictionaries for upsert
upsert_data = [{"id": ids[i], "values": vectors[i].tolist()} for i in range(len(vectors))]
 
start = time.time()
print(f"{num_vectors} vectors uploading...")
with client.batch(batch_size = BATCH_SIZE, callback=cs_check_batch_result) as batch:
    for i,v in enumerate(vectors):
        properties = {
            "abbreviation": ids_df['abbreviation'][i],
            "title": ids_df['title'][i],
            "content": ids_df['content'][i]
        }
        client.batch.add_data_object(properties,class_name='Skill',vector=v)
end = time.time()
tot_duration = end - start
print(f"{num_vectors} vectors uploaded in {tot_duration} seconds")

