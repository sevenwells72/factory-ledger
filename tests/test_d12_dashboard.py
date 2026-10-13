"""D1 receipt projections are verified against real posted ticket routes."""
import subprocess
from pathlib import Path
import pytest
import main
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import payload, prepare, commit  # noqa: F401


def test_d12_node_contract():
    subprocess.run(['node','--test','tests/test_d12_ui.js'],cwd=Path(__file__).parents[1],check=True)


@pytest.mark.db
def test_receipt_id_reaches_all_read_models(client, payload, actors):
    key=actors['floor']['key']
    drafted=prepare(client,payload,key)
    result=commit(client,drafted,key).json()
    number,txn=result['receipt_number'],result['transaction_id']
    day=main.get_plant_now().date().isoformat()
    for path,collection,id_key in [('/transactions/history?limit=1000','transactions','id'),('/ledger/recent?limit=50','events','transaction_id'),('/dashboard/api/activity/receipts','receipts','transaction_id'),('/dashboard/api/activity/daily-entries?date='+day,'entries','transaction_id')]:
        response=client.get(path,headers={'X-API-Key':key})
        assert response.status_code==200,response.text
        row=next(r for r in response.json()[collection] if r[id_key]==txn)
        assert row['receipt_number']==number
        assert row['void_receipt_number'] is None
    assert client.get('/receipts/'+number,headers={'X-API-Key':key}).json()['status']=='committed'
