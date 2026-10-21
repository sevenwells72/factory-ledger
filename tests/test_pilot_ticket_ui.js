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
async function harness({language='en',path='/make/prepare',draft={},canCommit=true,onPrepare,initialAttachment=null,initialPhoto=null,photoKind='correction_photo',payload={},result={receipt_number:'MK-261020-001',success:true}}={}){
  const host=new Element('main'),notice=new Element('div'),requests=[],failures=[],uploads=[],busy=[],attachments=[];let cancelled=0,edited=0;
  const labelled=new Set();
  const label=(node,value)=>{node.textContent=String(value??'');if(value?.en!==undefined){node.labels=[value.en,value.es];labelled.add(node);}else{delete node.labels;labelled.delete(node);}return node;};
  const ui={...ledger,lang:language,t:(en,es)=>Object.assign(new String(ui.lang==='es'?es:en),{en,es}),setText:label,relabel(){for(const node of labelled)if(descendants(host).includes(node))node.textContent=node.labels[ui.lang==='es'?1:0];},el(tag,text,parent){const node=label(new Element(tag),text);parent?.append(node);return node;},makeDraftHtml:()=>'',draftHtml:()=>'',autoSubstitutionsHtml:()=>'',plantTime:()=>'',failure(error){failures.push(error.message);},notice(){},
    async upload(file,kind){uploads.push({file,kind});return {attachment_id:91,kind};},
    async request(url,body){requests.push({url,body:plain(body)});if(url.includes('/commit'))return plain(result);return {ticket:'ticket-1',payload_hash:'hash-1',draft:plain(draft),actor:{name:'Arturo'},can_commit:canCommit,...(onPrepare?onPrepare(body):{})};}};
  const root={FLLedger:ui,addEventListener(){},matchMedia:()=>({matches:true})};vm.runInNewContext(source,{window:root,document:{getElementById:()=>notice},FLReceipts:{link:receipt=>'receipt.html?number='+receipt,page:name=>name}});
  const controller=ui.ticketCard(host,path,{client_source:'dashboard',...plain(payload)},{initialAttachment,initialPhoto,photoKind,onAttachment(attachment){attachments.push(attachment);},onCancel(){cancelled++;},onEdit(){edited++;},onBusy(value){busy.push(value);}});await flush();
  const all=predicate=>descendants(host).filter(predicate),find=predicate=>all(predicate)[0],button=text=>find(node=>node.tag==='button'&&node.textContent===text);
  return {host,ui,controller,requests,failures,uploads,busy,attachments,all,find,button,switchLanguage(lang){ui.lang=lang;ui.relabel();},get cancelled(){return cancelled;},get edited(){return edited;}};
}
test('Cancel closes the draft without a commit',async()=>{
  const h=await harness();h.button('Cancel').onclick();assert.equal(h.cancelled,1);assert.equal(h.host.children.length,0);assert.equal(h.requests.length,1);assert.equal(h.requests[0].url,'/make/prepare');
});
for(const [path,photoKind,array] of [['/adjust/prepare','correction_photo',false],['/receive/prepare','lot_tag',true],['/receive/prepare','bol',true],['/ship/prepare','bol',true]])test(`form ${path} ${photoKind} attachment is carried to the exact commit request`,async()=>{
  const h=await harness({path,photoKind,initialAttachment:{attachment_id:88,kind:photoKind}});assert.ok(h.find(node=>node.textContent==='Photo attached'));await h.button('Record').onclick();const body=h.requests.at(-1).body;assert.deepEqual(array?body.attachment_ids:[body.attachment_id],[88]);assert.equal(array?'attachment_id' in body:'attachment_ids' in body,false);assert.equal(h.uploads.length,0);
});
test('an unfinished draft upload disables Record and blocks Cancel until it finishes',async()=>{
  const h=await harness({path:'/adjust/prepare'});let finish;h.ui.upload=()=>new Promise(resolve=>finish=resolve);const photo=h.find(node=>node.type==='file');photo.files=[{name:'tag.png'}];const uploading=photo.onchange();assert.equal(h.button('Record').disabled,true);h.button('Cancel').onclick();assert.equal(h.cancelled,0);await h.button('Record').onclick();assert.equal(h.requests.length,1);finish({attachment_id:91});await uploading;await h.button('Record').onclick();assert.equal(h.requests.at(-1).body.attachment_id,91);
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
  const h=await harness({path:'/adjust/prepare'});let finish;h.ui.upload=()=>new Promise(resolve=>finish=resolve);const photo=h.find(node=>node.type==='file');photo.files=[{name:'replacement.png'}];const pending=photo.onchange();h.button('Edit entry').onclick();assert.equal(h.edited,0);finish({attachment_id:92});await pending;h.button('Edit entry').onclick();assert.equal(h.edited,1);
});

test('a replacement draft upload notifies the form once and commits the new attachment',async()=>{
  const h=await harness({path:'/receive/prepare',photoKind:'lot_tag',initialAttachment:{attachment_id:88,kind:'lot_tag'}}),photo=h.find(node=>node.type==='file');photo.files=[{name:'draft-B.png'}];await photo.onchange();assert.deepEqual(h.attachments,[{attachment_id:91,kind:'lot_tag'}]);assert.equal(h.uploads.length,1);await h.button('Record').onclick();assert.deepEqual(h.requests.at(-1).body.attachment_ids,[91]);assert.equal('attachment_id' in h.requests.at(-1).body,false);
});
test('an invalidated draft never reports a late photo upload back to the form',async()=>{
  const h=await harness({path:'/adjust/prepare'});let finish;h.ui.upload=()=>new Promise(resolve=>finish=resolve);const photo=h.find(node=>node.type==='file');photo.files=[{name:'late.png'}];const pending=photo.onchange();h.controller.invalidate();finish({attachment_id:92});await pending;assert.deepEqual(h.attachments,[]);assert.equal(h.requests.length,1);
});

test('an alternate physical lot moves every substitution using that product and its override together without confirming untouched tags',async()=>{
  const payload={substitutions:[{ingredient_product_id:3,substitute_product_id:1,lot_id:11,reason_code:'physical_substitution'},{ingredient_product_id:4,substitute_product_id:1,lot_id:11,reason_code:'physical_substitution'},{ingredient_product_id:5,substitute_product_id:2,lot_id:21,reason_code:'physical_substitution'}],ingredient_lots:[{ingredient_product_id:1,lot_id:11},{ingredient_product_id:2,lot_id:21}]};
  const h=await harness({draft:makeDraft,canCommit:false,payload});fillTags(h,{sun:false});await h.button('Check confirmations → new draft').onclick();h.button('OATS-B-001').onclick();await flush();const body=h.requests[1].body;
  assert.deepEqual(body.substitutions,[{...payload.substitutions[0],lot_id:12},{...payload.substitutions[1],lot_id:12},payload.substitutions[2]]);assert.deepEqual(body.ingredient_lots,[{ingredient_product_id:1,lot_id:12},{ingredient_product_id:2,lot_id:21}]);assert.deepEqual(body.lot_confirmations,[{lot_id:12,method:'full_code',value:'OATS-B-001'}]);assert.equal(h.button('Record').disabled,true);assert.equal(h.requests.some(request=>request.url.includes('/commit')),false);
});

for(const kind of ['lot_tag','bol'])test(`Receive draft uploads ${kind} with the existing evidence type and commits an attachment array`,async()=>{
  const h=await harness({path:'/receive/prepare',photoKind:'lot_tag'}),type=h.find(node=>node.name==='draft_photo_kind'),photo=h.find(node=>node.type==='file');type.value=kind;type.onchange();photo.files=[{name:'receive.png'}];await photo.onchange();assert.equal(h.uploads[0].kind,kind);assert.deepEqual(h.attachments,[{attachment_id:91,kind}]);await h.button('Record').onclick();assert.deepEqual(h.requests.at(-1).body.attachment_ids,[91]);assert.equal('attachment_id' in h.requests.at(-1).body,false);
});

for(const language of ['en','es'])for(const path of ['/make/prepare','/pack/prepare'])test(`${path} shared card never creates a photo control in ${language}`,async()=>{
  const h=await harness({language,path,initialAttachment:{attachment_id:88,kind:'correction_photo'}});
  assert.equal(h.find(node=>node.type==='file'),undefined);
  assert.equal(h.find(node=>/Attach photo|Adjuntar foto/.test(node.textContent)),undefined);
  await h.button(language==='en'?'Record':'Registrar').onclick();const body=h.requests.at(-1).body;
  assert.equal('attachment_id' in body,false);assert.equal('attachment_ids' in body,false);assert.equal(h.uploads.length,0);
});
for(const language of ['en','es'])for(const path of ['/receive/prepare','/adjust/prepare','/inventory/found/prepare'])test(`${path} keeps working photo upload and Record in ${language}`,async()=>{
  const h=await harness({language,path}),photo=h.find(node=>node.type==='file');assert.ok(photo);assert.equal(photo.parent.hidden,false);photo.files=[{name:'evidence.png'}];await photo.onchange();
  const kind=path==='/receive/prepare'?'lot_tag':'correction_photo';assert.equal(h.uploads[0].kind,kind);
  await h.button(language==='en'?'Record':'Registrar').onclick();const body=h.requests.at(-1).body;
  if(path==='/receive/prepare')assert.deepEqual(body.attachment_ids,[91]);else assert.equal(body.attachment_id,91);
  assert.ok(h.find(node=>node.className==='good'));assert.deepEqual(h.failures,[]);
});

for(const language of ['en','es'])test(`Receive draft changes the attached photo kind and commits the replacement in ${language}`,async()=>{
  const file={name:'same-label.png'},h=await harness({language,path:'/receive/prepare',initialAttachment:{attachment_id:88,kind:'lot_tag'},initialPhoto:file}),type=h.find(node=>node.name==='draft_photo_kind');
  let finish;h.ui.upload=(selected,kind)=>{h.uploads.push({file:selected,kind});return new Promise(resolve=>finish=resolve);};
  type.value='bol';const pending=type.onchange();assert.equal(h.button(language==='en'?'Record':'Registrar').disabled,true);h.button(language==='en'?'Cancel':'Cancelar').onclick();assert.equal(h.cancelled,0);await h.button(language==='en'?'Record':'Registrar').onclick();assert.equal(h.requests.length,1);
  assert.equal(h.uploads[0].file,file);assert.equal(h.uploads[0].kind,'bol');finish({attachment_id:92,kind:'bol'});await pending;
  assert.deepEqual(h.attachments,[{attachment_id:92,kind:'bol'}]);await h.button(language==='en'?'Record':'Registrar').onclick();assert.deepEqual(h.requests.at(-1).body.attachment_ids,[92]);
});
test('failed photo reclassification keeps the attached evidence and restores its kind',async()=>{
  const h=await harness({path:'/receive/prepare',initialAttachment:{attachment_id:88,kind:'lot_tag'},initialPhoto:{name:'label.png'}}),type=h.find(node=>node.name==='draft_photo_kind');h.ui.upload=async()=>{throw Error('offline');};type.value='bol';await type.onchange();assert.equal(type.value,'lot_tag');assert.deepEqual(h.attachments,[]);assert.match(h.find(node=>/previous type/.test(node.textContent)).textContent,/not changed/);await h.button('Record').onclick();assert.deepEqual(h.requests.at(-1).body.attachment_ids,[88]);
});
test('switching language refreshes physical-tag legends, actor and photo status without clearing the draft',async()=>{
  const h=await harness({path:'/receive/prepare',draft:makeDraft}),photo=h.find(node=>node.type==='file');photo.files=[{name:'label.png'}];await photo.onchange();
  h.switchLanguage('es');assert.ok(h.find(node=>node.textContent==='Foto adjunta'));assert.ok(h.find(node=>node.tag==='legend'&&node.textContent==='Confirma lote físico: Oats · OATS-A-001'));assert.ok(h.find(node=>node.textContent==='Registrando como Arturo'));assert.ok(h.button('Registrar'));
  h.switchLanguage('en');assert.ok(h.find(node=>node.textContent==='Photo attached'));assert.ok(h.find(node=>node.tag==='legend'&&node.textContent==='Confirm physical lot: Oats · OATS-A-001'));assert.equal(h.uploads.length,1);assert.equal(h.requests.length,1);
});

for(const language of ['en','es'])test(`receive success card puts the large CNS code before the receipt in ${language}`,async()=>{const h=await harness({language,path:'/receive/prepare',result:{receipt_number:'RCV-261021-001',success:true,lot_code:'26-10-21-OATS-001',supplier_lot_code:'SUP-22'}});h.ui.receiveLotHtml=receipt=>{ledger.lang=language;return ledger.receiveLotHtml(receipt);};await h.button(language==='en'?'Record':'Registrar').onclick();const card=h.host.children[0],tag=card.children[0];assert.match(tag.innerHTML,/class="receive-lot-code">26-10-21-OATS-001/);assert.match(tag.innerHTML,language==='en'?/>Write this on the pallet tag</:/>Escribe esto en la etiqueta del pallet</);assert.equal(card.children[1].className,'good');});
