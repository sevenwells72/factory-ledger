"""Optional D2 office catalog and extracted documents in the sole local browser DB."""
from psycopg2.extras import Json


def seed_office(cur, data):
    cur.execute("INSERT INTO customers(name,active) VALUES('Sunshine Bakery',true) RETURNING id")
    data['similar_customer']=cur.fetchone()['id']
    cur.execute("INSERT INTO products(name,odoo_code,type,is_service,active) VALUES('D2 Pallet Charge','D2PALLET','finished',true,true) RETURNING id")
    data['pallet']=cur.fetchone()['id']
    cur.execute("""INSERT INTO products(name,odoo_code,type,pack_format,case_size_lb,bags_per_case,units_per_case,retail_bag_oz,active,is_service)
                   VALUES('D2 Pouches','D2POUCH','finished','bagged',7.5,12,12,10,true,false) RETURNING id""")
    data['pouch']=cur.fetchone()['id']
    data['documents']={}
    for kind in ['sales','purchase']:
        extraction={'customer_name':'D12 Test Customer','supplier_name':'D12 Test Supplier','po_number':'000DOC-'+kind,
                    'lines':[{'description':'D12 Test Cases','quantity':5,'unit':'lb','quantity_lb':5,'expected_qty_lb':5}]}
        cur.execute("""INSERT INTO purchase_documents(storage_path,original_filename,mime_type,file_sha256,byte_size,status,document_kind,extraction)
                       VALUES(%s,%s,'image/png',%s,10,'extracted',%s,%s) RETURNING id""",
                    ('d2-local/'+kind,kind+'.png','d2-local-'+kind,kind,Json(extraction)))
        data['documents'][kind]=cur.fetchone()['id']
