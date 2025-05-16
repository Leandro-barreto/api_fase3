from routers import downloaddata
from pydantic import BaseModel

class AneelRequest(BaseModel):
    base_url: str = "https://dadosabertos.aneel.gov.br/api/3/action/datastore_search"
    resource_id: str = '3a7aee00-b6ee-4913-9670-f6b60f4a7bea'

def local_download():
    request = AneelRequest()
    print(downloaddata.get_dataframe(request))

if __name__=='__main__':
    local_download()