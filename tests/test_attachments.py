import hashlib
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
import attachments

PNG=b'\x89PNG\r\n\x1a\nfixture'

def test_private_volume_roundtrip_permissions_and_integrity(tmp_path,monkeypatch):
    monkeypatch.setenv('EVIDENCE_STORAGE_DIR',str(tmp_path/'evidence'))
    attachments.store(None,'object',PNG,'image/png')
    row={'storage_path':'object','bytes':len(PNG),'sha256':hashlib.sha256(PNG).hexdigest()}
    assert attachments.load(None,row)==PNG
    assert os.stat(tmp_path/'evidence').st_mode & 0o777 == 0o700
    assert os.stat(tmp_path/'evidence/object').st_mode & 0o777 == 0o600
    row['sha256']='0'*64
    with pytest.raises(HTTPException) as e: attachments.load(None,row)
    assert e.value.detail['error_code']=='ATTACHMENT_CORRUPT'

@pytest.mark.parametrize('content,mime,kind,code',[(b'', 'image/png','bol','ATTACHMENT_SIZE_LIMIT'),
    (b'a'*(attachments.MAX_BYTES+1),'image/png','bol','ATTACHMENT_SIZE_LIMIT'),
    (PNG,'text/html','bol','ATTACHMENT_TYPE_INVALID'),(b'<script>','image/png','bol','ATTACHMENT_TYPE_INVALID'),
    (b'%PDF-1.7','application/pdf','correction_photo','ATTACHMENT_TYPE_INVALID')])
def test_type_and_size_limits(content,mime,kind,code):
    with pytest.raises(HTTPException) as e: attachments.validate_file(content,mime,kind)
    assert e.value.detail['error_code']==code

def test_public_bucket_refused(monkeypatch):
    monkeypatch.delenv('EVIDENCE_STORAGE_DIR',raising=False)
    monkeypatch.setattr(attachments.httpx,'get',lambda *a,**k:SimpleNamespace(status_code=200,json=lambda:{'public':True}))
    api=SimpleNamespace(_storage_config=lambda:('https://storage.invalid',{}))
    with pytest.raises(HTTPException) as e: attachments.store(api,'object',PNG,'image/png')
    assert e.value.detail['error_code']=='STORAGE_NOT_PRIVATE'

@pytest.mark.parametrize('row,code',[(None,'ATTACHMENT_NOT_FOUND'),
    ({'uploaded_by_actor_id':2},'ATTACHMENT_WRONG_ACTOR'),
    ({'uploaded_by_actor_id':1,'ticket_id':3,'exception_id':None,'shipment_id':None},'ATTACHMENT_WRONG_CONTEXT')])
def test_reference_cannot_release_hold_without_existence_actor_and_ticket(row,code):
    cur=Mock();cur.fetchone.return_value=row
    with pytest.raises(HTTPException) as e: attachments.verified(cur,'bare string',{'id':1},ticket_id=4)
    assert e.value.detail['error_code']==code
