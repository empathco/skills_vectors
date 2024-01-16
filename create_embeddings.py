import os
import time
import sys 

from openai import OpenAI
import google.generativeai as genai
import pandas as pd
import numpy as np
from dotenv import load_dotenv

def get_embedding_openai(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.embeddings.create(input = [text], model=model).data[0].embedding

def get_embedding_gemini(text, model="models/embedding-001"):
   text = text.replace("\n", " ")
   embedding = genai.embed_content(model=model,
                                   content=text,
                                   task_type="SEMANTIC_SIMILARITY")
   return embedding["embedding"]


load_dotenv()
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
gemini = genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
providers = ["openai", "gemini"]

# change providers for embeddings here
provider = providers[1]
print(f'Using {provider} as the provider of embeddings.')
response = str(input("continue? y/n\n"))
if response not in ("y","Y","yes","Yes"):
   exit()

# read from csv to get skill descriptions
df = pd.read_csv('./data/skill_list.csv')
embeddings = []
df_size = len(df)
total_duration = 0
print(f'creating embeddings from {provider} for {df_size} descriptions')

for i, row in df.iterrows():
   description = df['level_description'][i]
   start = time.time()
   if provider == "openai":
      embeddings.append(get_embedding_openai(description))
   else:
      embeddings.append(get_embedding_gemini(description))
   end = time.time()
   duration = end - start
   total_duration += duration

   if len(embeddings) % 25 == 0:
      est_time = (total_duration/len(embeddings))*(df_size-len(embeddings))
      print(f'{len(embeddings)}/{df_size} estimated remaining time: {est_time} sec')

print(f'created embeddings using {provider}. Average time to create embedding: {total_duration/len(embeddings)}')

with open('./data/skill_vectors.npy', 'wb') as f:
   np.save(f, embeddings)
