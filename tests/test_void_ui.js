const {test}=require('node:test');
const assert=require('node:assert/strict');
const ui=require('../dashboard/ledger-ui.js');
test('void controls fail closed and rely on per-entry FL actions',()=>{
 const receipt={transactions:[{id:12,allowed_actions:['void']},{id:13,allowed_actions:[]},{id:14}]};
 for(const flag of [false,undefined,null,1,'true'])assert.deepEqual(ui.voidChoices(receipt,flag),[]);
 assert.deepEqual(ui.voidChoices(receipt,true).map(t=>t.id),[12]);
 assert.deepEqual(ui.voidChoices({},true),[]);
});
test('void submits the fixed reason only through prepare',()=>{
 assert.deepEqual(ui.voidRequest(12),{path:'/void/12/prepare',body:{reason:'data_entry_error',client_source:'dashboard'}});
 for(const id of [0,-1,'12',NaN])assert.throws(()=>ui.voidRequest(id));
});
test('voided original preserves exact VD backlink',()=>{
 assert.deepEqual(ui.receiptState({status:'committed',transactions:[{id:12,effective_status:'voided',void_receipt_number:'VD-261014-001'}]}),
 {tone:'voided',state:'voided',voidReceipt:'VD-261014-001'});
});
