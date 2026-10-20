const {test}=require('node:test');
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('dashboard/ledger-ticket.js','utf8'),ledger=require('../dashboard/ledger-ui.js');
const flush=async()=>{for(let i=0;i<5;i++)await new Promise(resolve=>setImmediate(resolve));};
class Element{
  constructor(tag,text=''){this.tag=tag;this.textContent=String(text??'');this.children=[];this.dataset={};this.value='';this.disabled=false;this.hidden=false;this.checked=false;}
  append(node){node.parent=this;this.children.push(node);}
  replaceChildren(...nodes){this.children=[];nodes.forEach(node=>this.append(node));}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(node=>node!==this);}
  setAttribute(name,value){this[name]=value;}
}
const descendants=node=>[node,...node.children.flatMap(descendants)];
const plain=value=>JSON.parse(JSON.stringify(value));
async function harness({language='en',path='/make/prepare',draft={},canCommit=true,onPrepare,initialAttachment=null,photoKind='correction_photo',payload={}}={}){
  const host=new Element('main'),notice=new Element('div'),requests=[],failures=[],uploads=[],busy=[],attachments=[];let cancelled=0,edited=0;
  const ui={...ledger,lang:language,t:(en,es)=>language==='es'?es:en,el(tag,text,parent){const node=new Element(tag,text);parent?.append(node);return node;},makeDraftHtml:()=>'',draftHtml:()=>'',autoSubstitutionsHtml:()=>'',plantTime:()=>'',failure(error){failures.push(error.message);},notice(){},
    async upload(file,kind){uploads.push({file,kind});return {attachment_id:91,kind};},
    async request(url,body){requests.push({url,body:plain(body)});if(url.includes('/commit'))return {receipt_number:'MK-261020-001',success:true};return {ticket:'ticket-1',payload_hash:'hash-1',draft:plain(draft),actor:{name:'Arturo'},can_commit:canCommit,...(onPrepare?onPrepare(body):{})};}};
  const root={FLLedger:ui,addEventListener(){},matchMedia:()=>({matches:true})};vm.runInNewContext(source,{window:root,document:{getElementById:()=>notice},FLReceipts:{link:receipt=>'receipt.html?number='+receipt,page:name=>name}});
  const controller=ui.ticketCard(host,path,{client_source:'dashboard',...plain(payload)},{initialAttachment,photoKind,onAttachment(attachment){attachments.push(attachment);},onCancel(){cancelled++;},onEdit(){edited++;},onBusy(value){busy.push(value);}});await flush();
  const all=predicate=>descendants(host).filter(predicate),find=predicate=>all(predicate)[0],button=text=>find(node=>node.tag==='button'&&node.textContent===text);
  return {host,ui,controller,requests,failures,uploads,busy,attachments,all,find,button,get cancelled(){return cancelled;},get edited(){return edited;}};
}
test('Cancel closes the draft without a commit',async()=>{
  const h=await harness();h.button('Cancel').onclick();assert.equal(h.cancelled,1);assert.equal(h.host.children.length,0);assert.equal(h.requests.length,1);assert.equal(h.requests[0].url,'/make/prepare');
});
for(const [path,photoKind,array] of [['/adjust/prepare','correction_photo',false],['/receive/prepare','lot_tag',true],['/receive/prepare','bol',true],['/ship/prepare','bol',true]])test(`form ${path} ${photoKind} attachment is carried to the exact commit request`,async()=>{
  const h=await harness({path,photoKind,initialAttachment:{attachment_id:88,kind:photoKind}});assert.ok(h.find(node=>node.textContent==='Photo attached'));await h.button('Record').onclick();const body=h.requests.at(-1).body;assert.deepEqual(array?body.attachment_ids:[body.attachment_id],[88]);assert.equal(array?'attachment_id' in body:'attachment_ids' in body,false);assert.equal(h.uploads.length,0);
});
test('an unfinished draft upload disables Record and blocks Cancel until it finishes',async()=>{
  const h=await harness();let finish;h.ui.upload=()=>new Promise(resolve=>finish=resolve);const photo=h.find(node=>node.type==='file');photo.files=[{name:'tag.png'}];const uploading=photo.onchange();assert.equal(h.button('Record').disabled,true);h.button('Cancel').onclick();assert.equal(h.cancelled,0);await h.button('Record').onclick();assert.equal(h.requests.length,1);finish({attachment_id:91});await uploading;await h.button('Record').onclick();assert.equal(h.requests.at(-1).body.attachment_id,91);
});
const makeDraft={input_plan:[{product_id:1,product_name:'Oats',lot_id:11,lot_code:'OATS-A-001'},{product_id:2,product_name:'Sunflower Seeds',lot_id:21,lot_code:'SUN-SUN1'}],lot_confirmation_group:{ingredients:[{product_id:1,candidates:[{lot_id:11,lot_code:'OATS-A-001'},{lot_id:12,lot_code:'OATS-B-001'}]},{product_id:2,candidates:[{lot_id:21,lot_code:'SUN-SUN1'}]}]}};
function fillTags(h,{sun=true}={}){const methods=h.all(node=>node.tag==='select'),inputs=h.all(node=>node.tag==='input'&&node.autocomplete==='off');methods.forEach(node=>node.value='last4');inputs[0].value='-001';inputs[0].oninput();if(sun){inputs[1].value='SUN1';inputs[1].oninput();}}
test('ambiguous suffix shows full codes and the suggested choice still needs every tag checked',async()=>{
  const h=await harness({draft:makeDraft,canCommit:false,onPrepare:body=>({can_commit:body.lot_confirmations?.length===2})});assert.equal('lot_confirmations' in h.requests[0].body,false);fillTags(h);await h.button('Check confirmations → new draft').onclick();assert.equal(h.requests.length,1);assert.ok(h.button('OATS-A-001'));assert.ok(h.button('OATS-B-001'));h.button('OATS-A-001').onclick();assert.equal(h.button('Record').disabled,true);await h.button('Check confirmations → new draft').onclick();await flush();assert.deepEqual(h.requests[1].body.lot_confirmations,[{lot_id:11,method:'full_code',value:'OATS-A-001'},{lot_id:21,method:'last4',value:'SUN1'}]);assert.equal(h.button('Record').disabled,false);
});
test('choosing a different full lot refreshes the plan and preserves only typed evidence for other ingredients',async()=>{
  const h=await harness({draft:makeDraft,canCommit:false});fillTags(h);await h.button('Check confirmations → new draft').onclick();h.button('OATS-B-001').onclick();await flush();const body=h.requests[1].body;assert.deepEqual(body.ingredient_lots,[{ingredient_product_id:1,lot_id:12}]);assert.deepEqual(body.lot_confirmations,[{lot_id:21,method:'last4',value:'SUN1'},{lot_id:12,method:'full_code',value:'OATS-B-001'}]);
});
test('choosing a different full lot never invents confirmation for an untouched ingredient',async()=>{
  const h=await harness({draft:makeDraft,canCommit:false});fillTags(h,{sun:false});await h.button('Check confirmations → new draft').onclick();h.button('OATS-B-001').onclick();await flush();assert.deepEqual(h.requests[1].body.lot_confirmations,[{lot_id:12,method:'full_code',value:'OATS-B-001'}]);assert.equal(h.button('Record').disabled,true);
});
for(const language of ['en','es'])test(`zero-stock draft warning is obvious and respects FL's permission to record in ${language}`,async()=>{
  const h=await harness({language,draft:{shortages:[{product_name:'Coconut Macaroon',on_hand_lb:0,short_lb:10}]}});const warning=h.find(node=>node.dataset.shortageWarning!==undefined);assert.match(warning.textContent,language==='en'?/Only 0 lb of Coconut Macaroon.*shortage/:/Solo hay 0 lb de Coconut Macaroon.*faltante/);assert.equal(h.button(language==='en'?'Record':'Registrar').disabled,false);
});

test('Edit cannot discard a draft while its replacement photo is uploading',async()=>{
  const h=await harness();let finish;h.ui.upload=()=>new Promise(resolve=>finish=resolve);const photo=h.find(node=>node.type==='file');photo.files=[{name:'replacement.png'}];const pending=photo.onchange();h.button('Edit entry').onclick();assert.equal(h.edited,0);finish({attachment_id:92});await pending;h.button('Edit entry').onclick();assert.equal(h.edited,1);
});

test('a replacement draft upload notifies the form once and commits the new attachment',async()=>{
  const h=await harness({path:'/receive/prepare',photoKind:'lot_tag',initialAttachment:{attachment_id:88,kind:'lot_tag'}}),photo=h.find(node=>node.type==='file');photo.files=[{name:'draft-B.png'}];await photo.onchange();assert.deepEqual(h.attachments,[{attachment_id:91,kind:'lot_tag'}]);assert.equal(h.uploads.length,1);await h.button('Record').onclick();assert.deepEqual(h.requests.at(-1).body.attachment_ids,[91]);assert.equal('attachment_id' in h.requests.at(-1).body,false);
});
test('an invalidated draft never reports a late photo upload back to the form',async()=>{
  const h=await harness();let finish;h.ui.upload=()=>new Promise(resolve=>finish=resolve);const photo=h.find(node=>node.type==='file');photo.files=[{name:'late.png'}];const pending=photo.onchange();h.controller.invalidate();finish({attachment_id:92});await pending;assert.deepEqual(h.attachments,[]);assert.equal(h.requests.length,1);
});

test('an alternate physical lot moves every substitution using that product and its override together without confirming untouched tags',async()=>{
  const payload={substitutions:[{ingredient_product_id:3,substitute_product_id:1,lot_id:11,reason_code:'physical_substitution'},{ingredient_product_id:4,substitute_product_id:1,lot_id:11,reason_code:'physical_substitution'},{ingredient_product_id:5,substitute_product_id:2,lot_id:21,reason_code:'physical_substitution'}],ingredient_lots:[{ingredient_product_id:1,lot_id:11},{ingredient_product_id:2,lot_id:21}]};
  const h=await harness({draft:makeDraft,canCommit:false,payload});fillTags(h,{sun:false});await h.button('Check confirmations → new draft').onclick();h.button('OATS-B-001').onclick();await flush();const body=h.requests[1].body;
  assert.deepEqual(body.substitutions,[{...payload.substitutions[0],lot_id:12},{...payload.substitutions[1],lot_id:12},payload.substitutions[2]]);assert.deepEqual(body.ingredient_lots,[{ingredient_product_id:1,lot_id:12},{ingredient_product_id:2,lot_id:21}]);assert.deepEqual(body.lot_confirmations,[{lot_id:12,method:'full_code',value:'OATS-B-001'}]);assert.equal(h.button('Record').disabled,true);assert.equal(h.requests.some(request=>request.url.includes('/commit')),false);
});

for(const kind of ['lot_tag','bol'])test(`Receive draft uploads ${kind} with the existing evidence type and commits an attachment array`,async()=>{
  const h=await harness({path:'/receive/prepare',photoKind:'lot_tag'}),type=h.find(node=>node.name==='draft_photo_kind'),photo=h.find(node=>node.type==='file');type.value=kind;type.onchange();photo.files=[{name:'receive.png'}];await photo.onchange();assert.equal(h.uploads[0].kind,kind);assert.deepEqual(h.attachments,[{attachment_id:91,kind}]);await h.button('Record').onclick();assert.deepEqual(h.requests.at(-1).body.attachment_ids,[91]);assert.equal('attachment_id' in h.requests.at(-1).body,false);
});
