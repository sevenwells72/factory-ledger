const {test}=require('node:test');const assert=require('node:assert/strict');const ui=require('../dashboard/ledger-ui.js');
test('a hold with a reserved receipt is not a recorded entry',()=>{
 for(const r of [{}, {success:true},{receipt_number:' '},{receipt_number:123},{receipt_number:'MK-1',success:false},{receipt_number:'MK-1',posted:false},{receipt_number:'MK-1',held:true},{receipt_number:'MK-1',status:'awaiting_approval'},{receipt_number:'MK-1',ticket_status:'rejected'}])assert.equal(ui.recorded(r),false);
 assert.equal(ui.recorded({receipt_number:'MK-261013-001',replayed:true}),true);
});
test('FL messages remain verbatim and unsafe labels remain text',()=>{const message='<img src=x onerror=alert(1)>';assert.equal(ui.errorText({detail:{message,message_es:'Falta una foto.'}},'en'),message);assert.equal(ui.errorText({detail:{message,message_es:'Falta una foto.'}},'es'),'Falta una foto.');assert.ok(!ui.dataHtml({product:message}).includes('<img'));assert.ok(!ui.dataHtml({ticket:'secret-token',payload_hash:'hash'}).includes('secret-token'));});
test('open queue groups follow server overdue flags and due dates',()=>{const rows=[{id:1,kind:'B',overdue:false},{id:2,kind:'A',overdue:true,due_at:'2026-10-11'},{id:3,kind:'A',overdue:true,due_at:'2026-10-10'},{id:4,kind:'B',overdue:true,due_at:'2026-10-09'}];assert.deepEqual(ui.groups(rows).map(([k,v])=>[k,v.map(r=>r.id)]),[['B',[4,1]],['A',[3,2]]]);assert.equal(rows[0].id,1);});

test('receipt times display the New York plant day across UTC midnight',()=>{assert.match(ui.plantTime('2026-10-14T00:00:00Z'),/Oct 13, 2026/);assert.match(ui.plantTime('2026-10-14T00:00:00Z'),/8:00 PM/);});

test('commit failure retains nested FL blockers verbatim',()=>{assert.equal(ui.errorText({detail:{message:'Draft no longer valid; prepare again.',blockers:[{message:'Enter the BOL number before recording.'}]}},'en'),'Draft no longer valid; prepare again.\nEnter the BOL number before recording.');});

// Review fix round 1 (PR #107). Each test failed on 71125d9.
test('resolve and reject use their XR receipt, never the source posting',()=>{
 const source={id:10,kind:'SHORTAGE',status:'resolved',receipt_number:'MK-261013-003'},xr='XR-261013-001';
 assert.deepEqual(ui.decisionOutcome('resolve',source),{tone:'bad'});
 assert.deepEqual(ui.decisionOutcome('resolve',{...source,resolution_receipt_number:xr}),{tone:'good',receipt:xr,status:'resolved'});
 assert.deepEqual(ui.decisionOutcome('resolve',{...source,status:'waived',resolution_receipt_number:xr}),{tone:'good',receipt:xr,status:'waived'});
 assert.deepEqual(ui.decisionOutcome('resolve',{...source,status:'open',resolution_receipt_number:xr}),{tone:'good',receipt:xr,status:'still_open'});
 assert.deepEqual(ui.decisionOutcome('reject',{rejected:true,posted:false,ticket_status:'rejected',receipt_number:'ADJ-261013-004',resolution_receipt_number:xr}),{tone:'good',receipt:xr,status:'declined'});
 assert.equal(ui.decisionOutcome('reject',{rejected:true,receipt_number:'ADJ-261013-004'}).tone,'bad');
});
test('decision receipts fail closed on missing or malformed XR and failed actions',()=>{
 for(const receipt of [undefined,null,'',123,'ADJ-261013-004','XR-',' XR-261013-001'])assert.equal(ui.decisionOutcome('resolve',{status:'resolved',resolution_receipt_number:receipt}).tone,'bad');
 for(const extra of [{success:false},{held:true},{status:'unknown'}])assert.equal(ui.decisionOutcome('resolve',{status:'resolved',resolution_receipt_number:'XR-261013-001',...extra}).tone,'bad');
 assert.equal(ui.decisionOutcome('reject',{rejected:false,resolution_receipt_number:'XR-261013-001'}).tone,'bad');
});
test('approve and counted decisions distinguish XR from the posting receipt',()=>{
 const xr='XR-261013-001',adj='ADJ-261013-004';
 assert.deepEqual(ui.decisionOutcome('approve',{receipt_number:adj,resolution_receipt_number:xr,transaction_id:24}),{tone:'good',receipt:xr,status:'approved',postingReceipt:adj});
 assert.deepEqual(ui.decisionOutcome('resolve',{status:'resolved',receipt_number:'MK-261013-003',resolution_receipt_number:xr,resolution_posting_receipt_number:adj}),{tone:'good',receipt:xr,status:'resolved',postingReceipt:adj});
 assert.equal(ui.decisionOutcome('approve',{receipt_number:adj,resolution_receipt_number:xr,held:true}).tone,'bad');
 assert.equal(ui.decisionOutcome('approve',{receipt_number:adj}).tone,'bad');
});
test('only server allowed_actions are offered, with held-decision mappings',()=>{
 assert.deepEqual(ui.resolutionChoices({kind:'SHORTAGE',allowed_actions:['counted']}),['counted'],'FL list wins');
 assert.deepEqual(ui.resolutionChoices({kind:'LARGE_CORRECTION',allowed_actions:['approve','reject']}),['approved','declined']);
 assert.deepEqual(ui.resolutionChoices({kind:'KOSHER_ATTESTATION',allowed_actions:['reject']}),['declined']);
 assert.deepEqual(ui.resolutionChoices({kind:'UNIDENTIFIED_LOT',allowed_actions:['identified']}),['identified']);
 for(const item of [{},{allowed_actions:[]},{allowed_actions:null},{allowed_resolutions:['counted']},{allowed_actions:[],allowed_resolutions:['counted']}])assert.deepEqual(ui.resolutionChoices(item,{approve_exception:true,resolve_exception:true},['counted']),[],'no frontend fallback');
});
test('fix 3: owner approve note is optional; reject and write-off require one',()=>{
 assert.deepEqual(ui.decisionRequest('approved','   '),{action:'approve',body:{}});
 assert.deepEqual(ui.decisionRequest('approved',' ok '),{action:'approve',body:{note:'ok'}});
 assert.deepEqual(ui.decisionRequest('declined','wrong count'),{action:'reject',body:{resolution_kind:'declined',note:'wrong count'}});
 assert.deepEqual(ui.decisionRequest('counted','recount',{counted_lb:0}),{action:'resolve',body:{resolution_kind:'counted',note:'recount',counted_lb:0}});
 assert.equal(ui.noteRequired('approved'),false);for(const k of ['declined','written_off','waived','counted'])assert.equal(ui.noteRequired(k),true,k);
});
test('fix 4: a voided receipt is VOIDED, never recorded-green',()=>{
 const base={status:'committed',transactions:[{id:24,effective_status:'voided'}]};
 assert.deepEqual(ui.receiptState(base),{tone:'voided',state:'voided',voidReceipt:null});
 assert.equal(ui.receiptState({...base,void_receipt_number:'VD-261013-001'}).voidReceipt,'VD-261013-001');
 assert.equal(ui.receiptState({status:'committed',transactions:[{effective_status:'voided'},{effective_status:'posted'}]}).state,'partly_voided');
 assert.deepEqual(ui.receiptState({status:'committed',transactions:[{effective_status:'posted'}]}),{tone:'good',state:'recorded'});
 assert.equal(ui.receiptState({status:'awaiting_approval',transactions:[]}).tone,'bad');
 assert.deepEqual(ui.receiptState({status:'committed',effective_status:'voided',void_receipt_number:'VD-261013-001',transactions:[]}),{tone:'voided',state:'voided',voidReceipt:'VD-261013-001'});
 assert.equal(ui.receiptState({status:'committed',effective_status:'partially_voided',void_receipt_number:'VD-261013-002'}).state,'partly_voided');
 assert.equal(ui.receiptState({...base,transactions:[{effective_status:'voided',void_receipt_number:'VD-261013-003'}]}).voidReceipt,'VD-261013-003');
});
test('fix 5: a timeout or lost connection reads as plain language, never browser text',()=>{
 const abort=Object.assign(Error('signal is aborted without reason'),{name:'AbortError'});
 assert.match(ui.transportText(abort,'en'),/^No answer from FL within 20 seconds — this is NOT confirmed\. Press Record again/);
 assert.match(ui.transportText(abort,'es'),/^FL no respondió/);
 assert.ok(!ui.transportText(abort,'en').includes('signal'));
 assert.match(ui.transportText(new TypeError('Failed to fetch'),'en'),/^Could not reach FL/);
 assert.equal(ui.transportText(Error('FL said no'),'en'),null);
});
test('fix 6: draft labels carry both languages so a switch relabels without rebuilding',()=>{
 const html=ui.draftHtml({delta_lb:5});assert.match(html,/<dt data-en="Change \(lb\)" data-es="Cambio \(lb\)">/);
 const entry=require('node:fs').readFileSync(require('node:path').join(__dirname,'../dashboard/entry.js'),'utf8');
 assert.ok(!/addEventListener\('ledger-language',render\)/.test(entry),'language switch must not re-render the form');
});
test('fix 0: supplier-lot resolve uses the ticketed A5 route and reports resolved exceptions',()=>{
 const src=require('node:fs').readFileSync(require('node:path').join(__dirname,'../dashboard/exceptions.js'),'utf8');
 assert.match(src,/'\/lots\/'\+item\.lot\.id\+'\/supplier-lot\/prepare'/);assert.match(src,/resolved_exception_ids/);assert.ok(!/PATCH/.test(src));
});
