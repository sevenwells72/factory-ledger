"""A6 private evidence storage. Binary objects never have public/signed URLs.

Supabase's evidence bucket is private and checked on every storage operation.
An explicit EVIDENCE_STORAGE_DIR may instead name a private persistent volume.
Neither backend is enabled implicitly. Database rows authorize every download.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Literal
from uuid import uuid4

import httpx
from fastapi import Depends, File, Form, Request, UploadFile, Response
from pydantic import BaseModel, Field
from psycopg2.extras import Json

import permissions
import write_tickets as wt

MAX_BYTES = 15 * 1024 * 1024
KINDS = ('bol','truck_photo','correction_photo','lot_tag','count_sheet','signed_packing_slip','other')
ROUTES = frozenset({('POST','/attachments'), ('GET','/attachments/{attachment_id}'),
                   ('GET','/attachments/{attachment_id}/file'), ('POST','/shipments/{shipment_id}/attachments')})

def named(actor):
    if actor['id'] is None or permissions.role_of(actor) not in permissions.NAMED:
        wt.fail(403, 'ROLE_NOT_ALLOWED', 'Evidence requires a named actor.')

def _volume():
    value = os.environ.get('EVIDENCE_STORAGE_DIR')
    if not value:
        return None
    root = Path(value)
    if not root.is_absolute() or root.is_symlink():
        wt.fail(503, 'STORAGE_NOT_CONFIGURED', 'Evidence volume must be an absolute private directory.')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    return root

def _bucket(api):
    url, headers = api._storage_config()
    try:
        r = httpx.get(url+'/storage/v1/bucket/evidence', headers=headers, timeout=30)
        if r.status_code == 404:
            r = httpx.post(url+'/storage/v1/bucket', headers=headers,
                           json={'id':'evidence','name':'evidence','public':False,
                                 'file_size_limit':MAX_BYTES}, timeout=30)
            if r.status_code not in (200,201,409):
                wt.fail(502, 'STORAGE_FAILED', 'Private evidence bucket could not be created.')
            r = httpx.get(url+'/storage/v1/bucket/evidence', headers=headers, timeout=30)
        if r.status_code != 200 or r.json().get('public') is not False:
            wt.fail(503, 'STORAGE_NOT_PRIVATE', 'Evidence bucket must be private.')
    except (httpx.HTTPError, ValueError):
        wt.fail(502, 'STORAGE_FAILED', 'Private evidence storage is unavailable.')
    return url, headers

def store(api, path, content, content_type):
    root = _volume()
    if root:
        dest = root/path
        with dest.open('xb') as f:
            os.chmod(dest, 0o600); f.write(content); f.flush(); os.fsync(f.fileno())
        return
    url, headers = _bucket(api)
    try:
        r = httpx.post(url+'/storage/v1/object/evidence/'+path,
                       headers={**headers,'Content-Type':content_type,'x-upsert':'false'}, content=content, timeout=30)
        if r.status_code not in (200,201):
            wt.fail(502, 'STORAGE_FAILED', 'Evidence upload failed.')
    except httpx.HTTPError:
        wt.fail(502, 'STORAGE_FAILED', 'Evidence upload failed.')

def load(api, row):
    root = _volume()
    if root:
        try:
            content = (root/row['storage_path']).read_bytes()
        except OSError:
            wt.fail(502, 'STORAGE_FAILED', 'Evidence file is unavailable.')
    else:
        url, headers = _bucket(api)
        try:
            r = httpx.get(url+'/storage/v1/object/evidence/'+row['storage_path'], headers=headers, timeout=30)
            if r.status_code != 200:
                wt.fail(502, 'STORAGE_FAILED', 'Evidence download failed.')
            content = r.content
        except httpx.HTTPError:
            wt.fail(502, 'STORAGE_FAILED', 'Evidence download failed.')
    if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
        wt.fail(502, 'ATTACHMENT_CORRUPT', 'Evidence integrity check failed.')
    return content

def validate_file(content, content_type, kind):
    signatures = {'image/png':content.startswith(b'\x89PNG\r\n\x1a\n'),
                  'image/jpeg':content.startswith(b'\xff\xd8\xff'),
                  'image/webp':content.startswith(b'RIFF') and content[8:12] == b'WEBP',
                  'application/pdf':content.startswith(b'%PDF-')}
    if not content or len(content)>MAX_BYTES:
        wt.fail(413,'ATTACHMENT_SIZE_LIMIT','Evidence must be nonempty and at most 15 MiB.')
    if not signatures.get(content_type):
        wt.fail(422,'ATTACHMENT_TYPE_INVALID','Upload a JPEG, PNG, WebP or PDF with matching content type.')
    if kind in ('correction_photo','truck_photo','lot_tag') and not content_type.startswith('image/'):
        wt.fail(422,'ATTACHMENT_TYPE_INVALID','This evidence kind requires a photo.')

def verified(cur, ref, actor, *, ticket_id=None, exception_id=None, kinds=None, claim=False):
    """P1.13: existence AND uploader AND context. Unbound own uploads are claimed atomically."""
    cur.execute('SELECT * FROM attachments WHERE attachment_ref=%s OR id::text=%s FOR UPDATE', (str(ref),str(ref)))
    row = cur.fetchone()
    if not row:
        wt.fail(422,'ATTACHMENT_NOT_FOUND','Upload evidence first; a reference alone is not evidence.')
    if actor['id'] is None or row['uploaded_by_actor_id'] != actor['id']:
        wt.fail(403,'ATTACHMENT_WRONG_ACTOR','Evidence must belong to the authenticated actor.')
    if ((row['ticket_id'] is not None and row['ticket_id'] != ticket_id)
            or (row['exception_id'] is not None and row['exception_id'] != exception_id)
            or (row['shipment_id'] is not None and ticket_id is None)):
        wt.fail(422,'ATTACHMENT_WRONG_CONTEXT','Evidence belongs to another ticket or exception.')
    if kinds and row['kind'] not in kinds:
        wt.fail(422,'ATTACHMENT_KIND_INVALID','Upload the required evidence kind.')
    if claim:
        cur.execute('UPDATE attachments SET ticket_id=COALESCE(ticket_id,%s),exception_id=COALESCE(exception_id,%s) WHERE id=%s',
                    (ticket_id,exception_id,row['id']))
    return row

def public(row):
    return {k:row[k] for k in ('id','attachment_ref','kind','sha256','content_type','bytes','uploaded_by_actor_id',
                              'created_at','ticket_id','shipment_id','exception_id','extracted')}

def read_allowed(actor, row):
    named(actor)
    if row['uploaded_by_actor_id'] != actor['id'] and permissions.role_of(actor) not in ('office','owner'):
        wt.fail(403,'ATTACHMENT_WRONG_ACTOR','You may only read your own evidence.')

class AttachRequest(BaseModel):
    attachment_id: int = Field(gt=0)
    class Config:
        extra = 'forbid'

def attach_shipment(api, cur, shipment_id, ref, actor):
    named(actor)
    cur.execute('SELECT * FROM shipments WHERE id=%s FOR UPDATE', (shipment_id,))
    shipment = cur.fetchone()
    if not shipment:
        wt.fail(404,'SHIPMENT_NOT_FOUND','Shipment not found.')
    if shipment['entered_by_actor_id'] != actor['id'] and permissions.role_of(actor) not in ('office','owner'):
        wt.fail(403,'ROLE_NOT_ALLOWED','Only the shipping actor or office/owner may attach shipment evidence.')
    row = verified(cur,ref,actor,ticket_id=shipment['ticket_id'],kinds=('bol','truck_photo','signed_packing_slip'),claim=True)
    if row['shipment_id'] not in (None,shipment_id):
        wt.fail(422,'ATTACHMENT_WRONG_CONTEXT','Evidence belongs to another shipment.')
    # Verify bytes still exist before a photo can satisfy proof.
    load(api,row)
    cur.execute('UPDATE attachments SET shipment_id=%s WHERE id=%s',(shipment_id,row['id']))
    if row['kind']=='bol' and row['content_type'].startswith('image/'):
        cur.execute("UPDATE shipments SET proof_status='complete' WHERE id=%s",(shipment_id,))
        cur.execute("""UPDATE exceptions SET status='resolved',resolution_kind='photo_attached',resolved_at=clock_timestamp(),
                       resolved_by_actor_id=%s,detail=detail || %s::jsonb
                       WHERE shipment_id=%s AND kind='BOL_PHOTO_MISSING' AND status IN ('open','escalated')""",
                    (actor['id'],json.dumps({'attachment_id':row['id']}),shipment_id))
    return dict(public(row),shipment_id=shipment_id)

def register_routes(app, api):
    @app.post('/attachments')
    def upload(request: Request, file: UploadFile=File(...), kind: Literal['bol','truck_photo','correction_photo','lot_tag','count_sheet','signed_packing_slip','other']=Form(...),
               ticket_id: Optional[int]=Form(None), extracted: str=Form('{}'), _: bool=Depends(api.verify_api_key)):
        actor=wt.identity(api,request); named(actor)
        content=file.file.read(MAX_BYTES+1); validate_file(content,file.content_type,kind)
        try:
            info=json.loads(extracted)
        except ValueError:
            wt.fail(422,'EXTRACTED_INVALID','Extracted information must be a JSON object.')
        if not isinstance(info,dict) or len(extracted)>10000:
            wt.fail(422,'EXTRACTED_INVALID','Extracted information must be a small JSON object.')
        with api.get_transaction() as cur:
            if ticket_id is not None:
                cur.execute('SELECT actor_id,status FROM write_tickets WHERE id=%s FOR SHARE',(ticket_id,))
                ticket=cur.fetchone()
                if not ticket or ticket['actor_id']!=actor['id'] or ticket['status'] not in ('prepared','awaiting_approval','committed'):
                    wt.fail(403,'ATTACHMENT_WRONG_CONTEXT','Upload only to your own ticket.')
            ref='att_'+uuid4().hex; path=uuid4().hex
            store(api,path,content,file.content_type)
            cur.execute('''INSERT INTO attachments(attachment_ref,kind,storage_path,sha256,content_type,bytes,uploaded_by_actor_id,ticket_id,extracted)
                           VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *''',
                        (ref,kind,path,hashlib.sha256(content).hexdigest(),file.content_type,len(content),actor['id'],ticket_id,Json(info)))
            row=cur.fetchone()
            return wt.json_value(api,dict(public(row),attachment_id=row['id']))

    def get(cur,attachment_id,actor):
        cur.execute('SELECT * FROM attachments WHERE id=%s',(attachment_id,)); row=cur.fetchone()
        if not row: wt.fail(404,'ATTACHMENT_NOT_FOUND','Attachment not found.')
        read_allowed(actor,row); return row

    @app.get('/attachments/{attachment_id}')
    def metadata(attachment_id:int,request:Request,_:bool=Depends(api.verify_api_key)):
        with api.get_transaction() as cur:
            return wt.json_value(api,public(get(cur,attachment_id,wt.identity(api,request))))

    @app.get('/attachments/{attachment_id}/file')
    def download(attachment_id:int,request:Request,_:bool=Depends(api.verify_api_key)):
        with api.get_transaction() as cur:
            row=get(cur,attachment_id,wt.identity(api,request))
            return Response(load(api,row),media_type=row['content_type'],headers={'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'})

    @app.post('/shipments/{shipment_id}/attachments')
    def attach(shipment_id:int,body:AttachRequest,request:Request,_:bool=Depends(api.verify_api_key)):
        with api.get_transaction() as cur:
            return wt.json_value(api,attach_shipment(api,cur,shipment_id,body.attachment_id,wt.identity(api,request)))
