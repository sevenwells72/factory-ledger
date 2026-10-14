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
test('fix 1: a resolve or reject result never borrows the source receipt and is never green',()=>{
 const source={id:10,kind:'SHORTAGE',status:'resolved',receipt_number:'MK-261013-003'};
 assert.deepEqual(ui.decisionOutcome('resolve',source),{tone:'neutral',status:'resolved'});
 assert.deepEqual(ui.decisionOutcome('resolve',{...source,status:'waived'}),{tone:'neutral',status:'waived'});
 assert.deepEqual(ui.decisionOutcome('resolve',{...source,status:'open'}),{tone:'neutral',status:'still_open'});
 assert.deepEqual(ui.decisionOutcome('reject',{rejected:true,receipt_number:'ADJ-261013-004'}),{tone:'neutral',status:'declined'});
 assert.equal(ui.decisionOutcome('resolve',{detail:{error_code:'X'}}).tone,'bad');
 assert.deepEqual(ui.decisionOutcome('approve',{receipt_number:'ADJ-261013-004',transaction_id:24}),{tone:'good',receipt:'ADJ-261013-004'});
 assert.equal(ui.decisionOutcome('approve',{receipt_number:'ADJ-261013-004',held:true}).tone,'bad');
});
test('fix 2: only resolutions this person may use are offered',()=>{
 const floor={resolve_exception:true,approve_exception:false},owner={resolve_exception:true,approve_exception:true},office={resolve_exception:false,approve_exception:false};
 assert.deepEqual(ui.resolutionChoices({kind:'LARGE_CORRECTION'},floor,['approved','declined']),[]);
 assert.deepEqual(ui.resolutionChoices({kind:'UNIDENTIFIED_LOT'},floor,['identified','written_off']),['identified']);
 assert.deepEqual(ui.resolutionChoices({kind:'UNIDENTIFIED_LOT'},owner,['identified','written_off']),['identified','written_off']);
 assert.deepEqual(ui.resolutionChoices({kind:'SHORTAGE'},office,['counted','missing_movement','voided']),[]);
 assert.deepEqual(ui.resolutionChoices({kind:'SHORTAGE',allowed_resolutions:['counted']},owner,['counted','voided']),['counted'],'FL list wins');
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
