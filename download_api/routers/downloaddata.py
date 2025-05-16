from fastapi import APIRouter
from pydantic import BaseModel
import requests
from services import databases
import pandas as pd

router = APIRouter()

class AneelRequest(BaseModel):
    base_url: str = "https://dadosabertos.aneel.gov.br/api/3/action/datastore_search"
    resource_id: str = '3a7aee00-b6ee-4913-9670-f6b60f4a7bea'

def get_aneel_data(base_url, resource_id, limit=5, query=None):
    params = {
        'resource_id': resource_id,
        'limit': limit
    }
    if query:
        params['q'] = query

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro: {e}")
        return None
    
def save_files_locally(df_pd):
    data_geracao = df_pd['DatGeracaoConjuntoDados'].unique().tolist()
    for data in data_geracao:
        data_str = data.replace('-', '')[:-2]
        df_pd[df_pd['DatGeracaoConjuntoDados'] == data].to_csv(f'data/aneel_{data_str}.csv', index=False)
    return 0

@router.post("/downloaddata/")
def get_dataframe(request: AneelRequest):
    base_url = request.base_url
    resource_id = request.resource_id

    data = get_aneel_data(base_url, resource_id, limit=10000)

    if data:
        if 'result' in data and 'records' in data['result']:
            records = data['result']['records']
            df_pd = pd.DataFrame(records)
            save_files_locally(df_pd)
        else:
            raise ValueError("Sem dados na resposta.")

    databases.save_tables_in_db()
    return "Base salva com sucesso"
