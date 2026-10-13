const {test}=require('node:test');const assert=require('node:assert/strict');const ui=require('../dashboard/ledger-ui.js');
test('a hold with a reserved receipt is not a recorded entry',()=>{
 for(const r of [{}, {success:true},{receipt_number:' '},{receipt_number:123},{receipt_number:'MK-1',success:false},{receipt_number:'MK-1',posted:false},{receipt_number:'MK-1',held:true},{receipt_number:'MK-1',status:'awaiting_approval'},{receipt_number:'MK-1',ticket_status:'rejected'}])assert.equal(ui.recorded(r),false);
 assert.equal(ui.recorded({receipt_number:'MK-261013-001',replayed:true}),true);
});
test('FL messages remain verbatim and unsafe labels remain text',()=>{const message='<img src=x onerror=alert(1)>';assert.equal(ui.errorText({detail:{message,message_es:'Falta una foto.'}},'en'),message);assert.equal(ui.errorText({detail:{message,message_es:'Falta una foto.'}},'es'),'Falta una foto.');assert.ok(!ui.dataHtml({product:message}).includes('<img'));assert.ok(!ui.dataHtml({ticket:'secret-token',payload_hash:'hash'}).includes('secret-token'));});
test('open queue groups follow server overdue flags and due dates',()=>{const rows=[{id:1,kind:'B',overdue:false},{id:2,kind:'A',overdue:true,due_at:'2026-10-11'},{id:3,kind:'A',overdue:true,due_at:'2026-10-10'},{id:4,kind:'B',overdue:true,due_at:'2026-10-09'}];assert.deepEqual(ui.groups(rows).map(([k,v])=>[k,v.map(r=>r.id)]),[['B',[4,1]],['A',[3,2]]]);assert.equal(rows[0].id,1);});

test('receipt times display the New York plant day across UTC midnight',()=>{assert.match(ui.plantTime('2026-10-14T00:00:00Z'),/Oct 13, 2026/);assert.match(ui.plantTime('2026-10-14T00:00:00Z'),/8:00 PM/);});

test('commit failure retains nested FL blockers verbatim',()=>{assert.equal(ui.errorText({detail:{message:'Draft no longer valid; prepare again.',blockers:[{message:'Enter the BOL number before recording.'}]}},'en'),'Draft no longer valid; prepare again.\nEnter the BOL number before recording.');});
