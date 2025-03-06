import datetime
import chromadb
import traceback

from chromadb.utils import embedding_functions
import pandas as pd

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
filepath = "COA_OpenData.csv"
def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    if collection.count() <= 0:
        
        df = pd.read_csv(filepath)
    

        for idx, row in df.iterrows():
            metadata = {
                "file_name": filepath,
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": int(datetime.datetime.strptime(row['CreateDate'], '%Y-%m-%d').timestamp())  
            }
        
            collection.add(
                ids=[str(idx)],
                metadatas=[metadata],
                documents=[row["HostWords"]]
            )
    return collection



def generate_hw02(question, city, store_type, start_date, end_date):
    collections = generate_hw01()

    query_results = collections.query(
        query_texts=[question],
        n_results=10,
        include=["metadatas", "distances"],
        where={
            "$and": [
                {"date": {"$gte": int(start_date.timestamp())}}, 
                {"date": {"$lte": int(end_date.timestamp())}}, 
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )

    filtered_similarity = []
    filtered_store_name = []
    for index in range(len(query_results['ids'])):
        for distance, metadata in zip(query_results['distances'][index], query_results['metadatas'][index]):
            similarity = 1 - distance
            # print(str(similarity)+","+str(metadata['name'])) 
            if similarity >= 0.8:
                filtered_similarity.append(similarity)
                filtered_store_name.append(metadata['name'])
                
    filtered_results = sorted(zip(filtered_similarity, filtered_store_name), key=lambda x: x[0], reverse=True)


    sorted_store_names = [name for _, name in filtered_results]
    # print("\n"+str(sorted_store_names))

    return sorted_store_names


    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    collections = generate_hw01()
    get_selected_store = collections.get(where={"name": store_name})
    metadatas = [{**meta, "new_store_name": new_store_name} for meta in get_selected_store.get("metadatas", [])]
    collections.upsert(ids=get_selected_store.get("ids", []), metadatas=metadatas, documents=get_selected_store.get("documents", []))
    
    query_results = collections.query(
        query_texts=[question],
        n_results=10,
        include=["metadatas", "distances"],
        where={
            "$and": [
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )


    filtered_similarity = []
    filtered_store_name = []
    for index in range(len(query_results["ids"])):
        for distance, metadata in zip(query_results['distances'][index], query_results['metadatas'][index]):
            similarity = 1 - distance
            if similarity >= 0.8:
                # print(str(similarity)+","+str(metadata['name'])) 
                filtered_similarity.append(similarity)
                new_store_name = metadata.get('new_store_name', "")
                name = metadata['name']
                filtered_store_name.append(new_store_name if new_store_name else name)

    filtered_results = sorted(zip(filtered_similarity, filtered_store_name), key=lambda x: x[0], reverse=True)

    sorted_store_names = [name for _, name in filtered_results] 
    # print("\n"+str(sorted_store_names))
    return sorted_store_names
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection
