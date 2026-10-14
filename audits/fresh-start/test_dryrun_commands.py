"""The production dry-run commands in HOSTED-REHEARSAL-2026-10-14.md, checked as written.

Offline part: every documented input/output path passes the tools' audits/fresh-start/ guard,
is gitignored, and no command carries --apply. Local-DB part (skipped unless TEST_DATABASE_URL
names a localhost server): the documented preview, apply dry run and verify run unchanged except
for the folder, against a throwaway database built from tests/schema/schema.sql. API calls and
sockets are blocked; only psql to localhost runs. The scratch database is dropped afterwards."""
import contextlib,csv,hashlib,io,json,os,re,shlex,shutil,subprocess,tempfile,unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit,urlunsplit
import fresh_start_common as f
import fresh_start_v3 as v
import apply_reset as app
import reset_preview,verify_reset

DOC=f.OUT/'HOSTED-REHEARSAL-2026-10-14.md'
TOOLS={'reset_preview.py':reset_preview.main,'apply_reset.py':app.main,'verify_reset.py':verify_reset.main}
PATH_FLAGS=('--coverage','--moves','--output-dir','--approval','--count-csv')
FAKE_KEY='FAKE-dryrun-owner-key-not-a-secret'


def documented():
    """(folder expression, [cp/python argv lists]) from the dry-run shell block."""
    text=DOC.read_text()
    block=re.search(r"## Michael's production dry run.*?```sh\n(.*?)```",text,re.S).group(1)
    lines=[x.split('   #')[0].rstrip() for x in block.replace('\\\n',' ').splitlines()]
    lines=[x for x in lines if x and not x.lstrip().startswith('#')]
    folder=lines[0].removeprefix('D=')
    assert lines[0].startswith('D=audits/fresh-start/'),lines[0]
    return folder,[shlex.split(x) for x in lines[1:]]


def expand(argv,folder):
    return [x.replace('$D',folder) for x in argv]


class DocumentedPathTests(unittest.TestCase):
    def test_commands_are_preview_dry_run_verify_and_cp(self):
        _,cmds=documented()
        self.assertEqual([c[0] for c in cmds],['python3','cp','python3','python3'])
        self.assertEqual([Path(c[2]).name for c in cmds if c[0]=='python3'],list(TOOLS))
        for c in cmds:
            self.assertNotIn('--apply',c)
            if c[0]=='python3':self.assertEqual(c[c.index('--target')+1],'production')

    def test_every_path_stays_under_fresh_start_and_is_gitignored(self):
        folder,cmds=documented()
        folder=folder.replace('$(date +%F)','2026-10-15')
        paths=[]
        for c in map(lambda c:expand(c,folder),cmds):
            if c[0]=='cp':paths.append(c[2]);continue
            args=c[3:]
            paths+=[args[i+1] for i,a in enumerate(args) if a in PATH_FLAGS]
            paths+=[a for i,a in enumerate(args) if not a.startswith('-') and (i==0 or args[i-1] not in PATH_FLAGS+('--area','--target'))]
        self.assertEqual(len(paths),12)
        with contextlib.chdir(f.ROOT):
            for p in paths:
                self.assertTrue(f.local_path(p).is_relative_to(f.OUT),p)
            ignored=subprocess.run(['git','check-ignore','-q',folder+'/preview/reset-preview-v3.json'],cwd=f.ROOT)
        self.assertEqual(ignored.returncode,0,'dry-run folder must be gitignored')


def local_url():
    url=os.environ.get('TEST_DATABASE_URL','')
    parts=urlsplit(url)
    if not url or parts.hostname not in ('localhost','127.0.0.1') or (parts.port or 5432)!=5432:
        return None
    return parts


@unittest.skipUnless(local_url() and shutil.which('psql') and shutil.which('createdb'),
                     'needs TEST_DATABASE_URL on localhost:5432 and psql/createdb')
class DocumentedDryRunTests(unittest.TestCase):
    def setUp(self):
        parts=local_url();self.db=f'fl_dryrun_doc_{os.getpid()}'
        base=['-h',parts.hostname,'-p','5432']+(['-U',parts.username] if parts.username else [])
        env=dict(os.environ,PGPASSWORD=parts.password or '');env.pop('PGOPTIONS',None);env.pop('DATABASE_URL',None)
        def run(*a,**k):
            p=subprocess.run(*a,capture_output=True,env=env,**k)
            self.assertEqual(p.returncode,0,p.stderr[-2000:]);return p
        run(['dropdb',*base,'--if-exists',self.db]);run(['createdb',*base,self.db])
        self.addCleanup(lambda:subprocess.run(['dropdb',*base,'--if-exists','--force',self.db],capture_output=True,env=env))
        psql=['psql',*base,'-d',self.db,'-X','-q','-v','ON_ERROR_STOP=1']
        run(psql+['-c','CREATE EXTENSION IF NOT EXISTS pg_trgm'])
        run(psql+['-f',str(f.ROOT/'tests/schema/schema.sql')])
        self.received=(datetime.now(timezone.utc)-timedelta(days=2)).replace(microsecond=0)
        # A historical receipt, entered when it happened: bypass the stamp-created_at-now
        # triggers for this seed only (scratch localhost database, single transaction).
        run(psql,input=f"""
BEGIN; SET LOCAL session_replication_role=replica;
INSERT INTO actors(name,role,key_hash,active) VALUES ('Blubber','owner','{hashlib.sha256(FAKE_KEY.encode()).hexdigest()}',true);
INSERT INTO products(id,name,odoo_code,type,uom,active) VALUES (9001,'FAKE dry-run oats','99001','ingredient','lb',true);
INSERT INTO lots(id,product_id,lot_code,status) VALUES (9001,9001,'FAKE-DRY-1','active');
INSERT INTO transactions(id,type,timestamp,occurred_at,created_at,business_date,notes) VALUES (9001,'receive','{self.received.isoformat()}',
 '{self.received.isoformat()}','{self.received.isoformat()}','{self.received.astimezone(f.PLANT).date()}','FAKE dry-run receipt');
INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb,created_at) VALUES (9001,9001,9001,100,'{self.received.isoformat()}');
COMMIT;
""",text=True)
        tmp=tempfile.TemporaryDirectory(prefix='.FAKE-dryrun-',dir=f.OUT);self.addCleanup(tmp.cleanup)
        self.tmp=Path(tmp.name)
        credential=self.tmp/'db_url';credential.write_text(urlunsplit(parts._replace(path='/'+self.db)))
        key=self.tmp/'owner_key';key.write_text(FAKE_KEY);key.chmod(0o600)
        targets=dict(f.TARGETS,production=dict(f.TARGETS['production'],credential=credential,key=key))
        for obj,name,value in [(f,'TARGETS',targets),(app,'JOURNAL_ROOT',self.tmp/'apply-state'),
                               (app,'RESULTS',self.tmp/'apply-results'),(app,'LOCK',self.tmp/'lock')]:
            c=patch.object(obj,name,value);c.start();self.addCleanup(c.stop)
        for target in ('socket.create_connection','socket.socket.connect','apply_reset.LiveBackendV3.call',
                       'apply_reset.LiveBackend.request'):
            c=patch(target,side_effect=AssertionError('LIVE API ACCESS FORBIDDEN'));c.start();self.addCleanup(c.stop)

    def write_inputs(self,folder):
        folder.mkdir(parents=True)
        t=self.received+timedelta(days=1)
        row={k:'' for k in v.COUNT_COLUMNS}
        row.update(sheet_id='S1',area='Dry storage',count_date=t.astimezone(f.PLANT).date().isoformat(),counter='FAKE',
                   start_time=(t-timedelta(minutes=10)).isoformat(),end_time=(t+timedelta(minutes=10)).isoformat(),
                   row_id='R1',product_id='9001',SKU='99001',name='FAKE dry-run oats',row_kind='COUNT',lot_code='FAKE-DRY-1',
                   quantity='90',unit='lb',location='Dry storage shelf',counted_time=t.isoformat())
        coverage=dict(product_id='9001',SKU='99001',name='FAKE dry-run oats',all_locations_searched='yes',
                      completed_at=(t+timedelta(minutes=20)).isoformat(),owner_initials='FAKE',notes='')
        moves=['move_id','product_id','SKU','lot_code','quantity','unit','from_area','from_location','to_area',
               'to_location','moved_at','tag_id','fl_transaction_id','notes']
        for name,cols,rows in [('count',v.COUNT_COLUMNS,[row]),('coverage',list(coverage),[coverage]),('moves',moves,[])]:
            with (folder/(name+'.csv')).open('w',newline='') as out:
                w=csv.DictWriter(out,fieldnames=cols);w.writeheader();w.writerows(rows)

    def sign(self,approval,preview):
        raw=preview.read_bytes();rows=json.loads(raw)['rows']
        ready=sorted({r['product_id'] for r in rows if r['status']=='READY'})
        self.assertEqual(ready,[9001],[(r['product_id'],r['status'],r['holds']) for r in rows])
        fields={'Preview file':preview.name,'SHA-256':hashlib.sha256(raw).hexdigest(),'Approved product IDs':'9001',
                'Owner name':'Blubber','Named actor':'Blubber','Date':datetime.now(f.PLANT).date().isoformat(),
                'Signature':'Blubber','Owner approves held entries':'N','Extra-kosher lots':'none'}
        text=approval.read_text()
        for k,val in fields.items():
            text=re.sub(rf'^{re.escape(k)}:.*$',f'{k}: {val}',text,count=1,flags=re.M)
        approval.write_text(text)

    def test_documented_commands_run_in_dry_run_mode(self):
        _,cmds=documented()
        folder=str(self.tmp.relative_to(f.ROOT)/'dryrun-doc')
        self.write_inputs(f.ROOT/folder)
        results=[]
        with contextlib.chdir(f.ROOT):
            for c in (expand(c,folder) for c in cmds):
                if c[0]=='cp':
                    shutil.copyfile(c[1],c[2]);self.sign(Path(c[2]),Path(c[2]).with_name('reset-preview-v3.json'));continue
                out=io.StringIO()
                with patch('sys.argv',[c[2],*c[3:]]),contextlib.redirect_stdout(out):
                    results.append((TOOLS[Path(c[2]).name](),out.getvalue()))
        (preview_code,_),(dry_code,dry),(verify_code,_)=results
        self.assertEqual(preview_code,0)
        self.assertEqual(dry_code,0)
        self.assertIn('Target: PRODUCTION',dry)
        self.assertIn('Product 9001 / lot FAKE-DRY-1: 100.0000 + (-10) = 90.0000 lb;',dry)
        self.assertIn('DRY RUN ONLY: no API calls, no inventory changes. Named actor verified by read-only lookup.',dry)
        self.assertFalse(any((self.tmp/'apply-state').rglob('journal.jsonl')),'dry run must not write a journal')
        self.assertFalse((self.tmp/'apply-results').exists())
        self.assertEqual(verify_code,2,'nothing posted: the proposed difference must remain')
        self.assertTrue((f.ROOT/folder/'verify').is_dir())


if __name__=='__main__':unittest.main()
