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
  const host=new Element('main'),notice=new Element('div'),requests=[],failures=[],uploads=[],busy=[],attachments=[],renderedDrafts=[];let cancelled=0,edited=0;
  const labelled=new Set();
  const label=(node,value)=>{node.textContent=String(value??'');if(value?.en!==undefined){node.labels=[value.en,value.es];labelled.add(node);}else{delete node.labels;labelled.delete(node);}return node;};
  const ui={...ledger,lang:language,t:(en,es)=>Object.assign(new String(ui.lang==='es'?es:en),{en,es}),setText:label,relabel(){for(const node of labelled)if(descendants(host).includes(node))node.textContent=node.labels[ui.lang==='es'?1:0];},el(tag,text,parent){const node=label(new Element(tag),text);parent?.append(node);return node;},makeDraftHtml:()=>'',draftHtml:value=>{renderedDrafts.push(value);return '';},autoSubstitutionsHtml:()=>'',plantTime:value=>String(value||''),failure(error){failures.push(error.message);},notice(){},
    async upload(file,kind){uploads.push({file,kind});return {attachment_id:91,kind};},
    async request(url,body){requests.push({url,body:plain(body)});if(url.includes('/commit'))return plain(result);return {ticket:'ticket-1',payload_hash:'hash-1',draft:plain(draft),actor:{name:'Arturo'},can_commit:canCommit,...(onPrepare?onPrepare(body):{})};}};
  const root={FLLedger:ui,addEventListener(){},matchMedia:()=>({matches:true})};vm.runInNewContext(source,{window:root,document:{getElementById:()=>notice},FLReceipts:{link:receipt=>'receipt.html?number='+receipt,page:name=>name}});
  const controller=ui.ticketCard(host,path,{client_source:'dashboard',...plain(payload)},{initialAttachment,initialPhoto,photoKind,onAttachment(attachment){attachments.push(attachment);},onCancel(){cancelled++;},onEdit(){edited++;},onBusy(value){busy.push(value);}});await flush();
  const all=predicate=>descendants(host).filter(predicate),find=predicate=>all(predicate)[0],button=text=>find(node=>node.tag==='button'&&node.textContent===text);
  return {host,ui,controller,requests,failures,uploads,busy,attachments,renderedDrafts,all,find,button,switchLanguage(lang){ui.lang=lang;ui.relabel();},get cancelled(){return cancelled;},get edited(){return edited;}};
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


const stockedDraft={input_plan:[{product_id:1,product_name:'Oats',lot_id:11,lot_code:'CNS-A-001',quantity_lb:100}],lot_confirmation_group:{ingredients:[{product_id:1,product_name:'Oats',needed_lb:100,candidates:[
  {lot_id:12,lot_code:'CNS-B-002',supplier_lot_code:'BAG 200',supplier_lot_codes:['BAG 200','Alias B'],supplier_name:'Second Supplier',received_at:'2026-10-20',available_lb:120},
  {lot_id:11,lot_code:'CNS-A-001',supplier_lot_code:'BAG 100',supplier_lot_codes:['BAG 100'],supplier_name:'First Supplier',received_at:'2026-10-01',available_lb:100,fifo_suggested:true},
  {lot_id:10,lot_code:'CNS-OPENING',supplier_lot_code:null,supplier_lot_codes:[],supplier_name:'Opening balance',received_at:'2026-10-21',available_lb:100,opening_stock:true},
  {lot_id:13,lot_code:'EMPTY',supplier_lot_code:'EMPTY',received_at:'2026-09-01',available_lb:0}
]}]}};
function stockRow(h,id){return h.find(node=>node.dataset.stockLot===String(id));}
function chooseStock(h,id,amount){const row=stockRow(h,id),check=descendants(row).find(node=>node.name==='ingredient_lot_id');check.checked=true;check.onchange();const qty=descendants(row).find(node=>node.name==='ingredient_lot_quantity');if(amount!==undefined){qty.value=String(amount);qty.oninput();}return {row,check,qty};}
function typeStock(h,value,method='supplier_lot'){const select=h.find(node=>node.name==='ingredient_confirmation_method');select.value=method;select.onchange();const input=h.find(node=>node.name==='ingredient_confirmation_value');input.value=value;input.oninput();}
const checkStock=async h=>{h.button(h.ui.lang==='es'?'Verificar confirmaciones → nuevo borrador':'Check confirmations → new draft').onclick();await flush();};
for(const path of ['/make/prepare','/pack/prepare'])for(const language of ['en','es'])test(`${path} stock list shows opening first, FIFO highlight, supplier identity and no confirmed default in ${language}`,async()=>{
  const h=await harness({path,language,draft:stockedDraft});
  assert.deepEqual(h.all(node=>node.dataset.stockLot).map(node=>node.dataset.stockLot),['10','11','12']);
  assert.equal(h.all(node=>node.name==='ingredient_lot_id').every(node=>!node.checked),true);
  assert.match(stockRow(h,11).className,/fifo-suggested/);
  assert.ok(h.find(node=>node.tag==='strong'&&node.textContent==='BAG 100'));
  assert.ok(h.find(node=>node.textContent===(language==='es'?'Proveedor: First Supplier':'Supplier: First Supplier')));
  assert.ok(h.find(node=>node.textContent===(language==='es'?'Recibido: 1 oct 2026':'Received: Oct 1, 2026')));
  assert.equal(h.button(language==='es'?'Registrar':'Record').disabled,true);
  await h.button(language==='es'?'Registrar':'Record').onclick();await checkStock(h);
  assert.equal(h.requests.length,1);assert.ok(h.find(node=>/FIFO.*(?:confirmed|confirmado)/.test(node.textContent)));
});
test('typed supplier alias matches case and whitespace while selecting the matching physical stock',async()=>{
  const h=await harness({draft:stockedDraft,canCommit:false});typeStock(h,' aLiAs   b ');await checkStock(h);
  assert.deepEqual(h.requests[1].body.ingredient_lot_allocations,[{ingredient_product_id:1,lot_id:12,quantity_lb:100}]);
  assert.deepEqual(h.requests[1].body.lot_confirmations,[{lot_id:12,method:'supplier_lot',value:'aLiAs   b',selected:false}]);
});
for(const language of ['en','es'])test(`unknown supplier code leaves the list visible and gives opening-stock guidance in ${language}`,async()=>{
  const h=await harness({draft:stockedDraft,language});typeStock(h,'UNKNOWN');await checkStock(h);assert.equal(h.requests.length,1);
  assert.ok(h.find(node=>node.textContent===(language==='es'?'Ese lote del proveedor no tiene existencias en FL — si son existencias antiguas que ya estaban aquí el 21 de octubre, elija el lote de EXISTENCIAS INICIALES':"That supplier lot isn't in stock in FL — if it's old stock already here on Oct 21, choose the OPENING STOCK lot")));
  assert.equal(h.all(node=>node.dataset.stockLot).length,3);
});
test('ambiguous supplier code requires choosing a row and never chooses FIFO automatically',async()=>{
  const draft=plain(stockedDraft);draft.lot_confirmation_group.ingredients[0].candidates[0].supplier_lot_codes.push('BAG 100');
  const h=await harness({draft});typeStock(h,'bag100');await checkStock(h);assert.equal(h.requests.length,1);assert.equal(h.all(node=>node.name==='ingredient_lot_id').some(node=>node.checked),false);assert.ok(h.find(node=>/More than one lot matches/.test(node.textContent)));
  chooseStock(h,12);await checkStock(h);assert.equal(h.requests[1].body.lot_confirmations[0].lot_id,12);assert.equal(h.requests[1].body.lot_confirmations[0].selected,true);
});
for(const language of ['en','es'])test(`an explicitly chosen opening lot carries its dedicated label and evidence in ${language}`,async()=>{
  const h=await harness({draft:stockedDraft,language});assert.ok(h.find(node=>node.tag==='strong'&&node.textContent===(language==='es'?'EXISTENCIAS INICIALES — use para cualquier bolsa que ya estaba aquí el 21 de octubre':'OPENING STOCK — use for any bag already here on Oct 21')));
  chooseStock(h,10);await checkStock(h);assert.deepEqual(h.requests[1].body.lot_confirmations,[{lot_id:10,method:'supplier_lot',value:'OPENING STOCK',selected:true}]);
});
for(const path of ['/make/prepare','/pack/prepare'])test(`${path} accepts two deliberately selected lots only when their lb sum to the ingredient total`,async()=>{
  const h=await harness({path,draft:stockedDraft,payload:path==='/pack/prepare'?{source_product_id:1,lot_allocations:[{lot_id:11,quantity_lb:100}]}:{ingredient_lots:[{ingredient_product_id:1,lot_id:11}]}});
  chooseStock(h,11,40);const second=chooseStock(h,12);assert.equal(second.qty.value,'');await checkStock(h);assert.equal(h.requests.length,1);
  second.qty.value='50';second.qty.oninput();await checkStock(h);assert.equal(h.requests.length,1);assert.ok(h.find(node=>/must total 100 lb; currently 90 lb/.test(node.textContent)));
  second.qty.value='60';second.qty.oninput();await checkStock(h);assert.deepEqual(h.requests[1].body.ingredient_lot_allocations,[{ingredient_product_id:1,lot_id:11,quantity_lb:40},{ingredient_product_id:1,lot_id:12,quantity_lb:60}]);assert.equal(h.requests[1].body.lot_confirmations.length,2);
  if(path==='/pack/prepare')assert.equal('lot_allocations' in h.requests[1].body,false);else assert.deepEqual(h.requests[1].body.ingredient_lots,[]);
});
for(const method of ['last4','full_code','scan','pallet'])test(`stock selector retains typed ${method} evidence for FL validation`,async()=>{
  const h=await harness({draft:stockedDraft});typeStock(h,method==='last4'?'-001':'CNS-A-001',method);await checkStock(h);assert.equal(h.requests[1].body.lot_confirmations[0].method,method);assert.equal(h.requests[1].body.lot_confirmations[0].lot_id,11);
});
test('stock split requires positive amounts and cannot confirm an untouched second ingredient',async()=>{
  const draft=plain(stockedDraft);draft.lot_confirmation_group.ingredients.push({product_id:2,product_name:'Seeds',needed_lb:5,candidates:[{lot_id:21,lot_code:'SEEDS',supplier_lot_code:'SEED BAG',available_lb:10}]});
  const h=await harness({draft});chooseStock(h,11,0);await checkStock(h);assert.equal(h.requests.length,1);assert.ok(h.find(node=>/positive lb for each chosen lot/.test(node.textContent)));
  const qty=descendants(stockRow(h,11)).find(node=>node.name==='ingredient_lot_quantity');qty.value='100';qty.oninput();await checkStock(h);assert.equal(h.requests.length,1);assert.equal(descendants(stockRow(h,21)).find(node=>node.name==='ingredient_lot_id').checked,false);
});
test('language switches translate opening stock and FIFO text while retaining deliberate row choices and split lb',async()=>{
  const h=await harness({draft:stockedDraft});chooseStock(h,11,40);chooseStock(h,12,60);h.switchLanguage('es');assert.ok(h.find(node=>node.textContent==='Sugerencia FIFO'));assert.ok(h.find(node=>node.tag==='strong'&&node.textContent.startsWith('EXISTENCIAS INICIALES')));await checkStock(h);assert.deepEqual(h.requests[1].body.ingredient_lot_allocations.map(value=>value.quantity_lb),[40,60]);
});
test('freshly checked allocation restores chosen quantities and only then enables Record',async()=>{
  const h=await harness({draft:stockedDraft,canCommit:false,onPrepare:body=>({can_commit:!!body.lot_confirmations?.length,draft:body.lot_confirmations?.length?{...plain(stockedDraft),input_plan:body.ingredient_lot_allocations.map(row=>({product_id:row.ingredient_product_id,lot_id:row.lot_id,quantity_lb:row.quantity_lb}))}:plain(stockedDraft)})});
  chooseStock(h,11,40);chooseStock(h,12,60);await checkStock(h);assert.equal(h.button('Record').disabled,false);assert.equal(h.all(node=>node.name==='ingredient_lot_id'&&node.checked).length,2);assert.deepEqual(h.all(node=>node.name==='ingredient_lot_quantity'&&!node.disabled).map(node=>node.value),['40','60']);
  await h.button('Record').onclick();assert.ok(h.requests.at(-1).url.includes('/commit'));
});


test('a selected lot with stock can retain the existing shortage review rather than inventing a new stock cap',async()=>{
  const draft=plain(stockedDraft);draft.lot_confirmation_group.ingredients[0].candidates=draft.lot_confirmation_group.ingredients[0].candidates.filter(lot=>lot.lot_id===11);draft.lot_confirmation_group.ingredients[0].candidates[0].available_lb=50;draft.shortages=[{product_name:'Oats',on_hand_lb:50,short_lb:50}];
  const h=await harness({draft});const chosen=chooseStock(h,11);assert.equal(chosen.qty.value,'100');assert.equal(chosen.qty.max,undefined);await checkStock(h);assert.deepEqual(h.requests[1].body.ingredient_lot_allocations,[{ingredient_product_id:1,lot_id:11,quantity_lb:100}]);assert.ok(h.find(node=>node.dataset.shortageWarning!==undefined));
});


test('received date-only values preserve their calendar date while timestamps use the New York date',async()=>{
  const draft=plain(stockedDraft);draft.lot_confirmation_group.ingredients[0].candidates[0].received_at='2026-10-20T02:00:00Z';
  const h=await harness({draft});assert.ok(h.find(node=>node.textContent==='Received: Oct 1, 2026'));assert.ok(h.find(node=>node.textContent==='Received: Oct 19, 2026'));
});


for(const path of ['/make/prepare','/pack/prepare'])test(`${path} product and needed weights stay visible without a duplicate unchosen FIFO input list`,async()=>{
  const draft=plain(stockedDraft);draft.product={name:'Classic'};draft.batches=1;draft.ingredients=[{ingredient_id:1,ingredient_name:'Oats',needed_lb:100,lots:[{lot_code:'CNS-A-001',available_lb:100}]}];
  const h=await harness({draft,path}),visible=h.renderedDrafts.at(-1);assert.equal(visible.input_plan,undefined);assert.deepEqual(visible.product,{name:'Classic'});assert.equal(visible.batches,1);assert.deepEqual(plain(visible.ingredients),[{ingredient_id:1,ingredient_name:'Oats',needed_lb:100}]);assert.equal(h.all(node=>node.dataset.stockLot).length,3);
});


test('typed unique supplier evidence never claims that the worker explicitly selected its row',async()=>{
  const h=await harness({draft:stockedDraft});typeStock(h,'BAG 100');await checkStock(h);assert.equal(h.requests[1].body.lot_confirmations[0].selected,false);
  const explicit=await harness({draft:stockedDraft});chooseStock(explicit,11);typeStock(explicit,'BAG 100');await checkStock(explicit);assert.equal(explicit.requests[1].body.lot_confirmations[0].selected,true);
});


for(const language of ['en','es'])test(`commingled supplier aliases stay visible beside their primary bag code in ${language}`,async()=>{
  const h=await harness({draft:stockedDraft,language});const row=stockRow(h,12),texts=descendants(row).map(node=>node.textContent);assert.ok(texts.includes('BAG 200'));assert.ok(texts.includes(language==='es'?'También en este lote: Alias B':'Also on this lot: Alias B'));assert.equal(h.all(node=>node.name==='ingredient_lot_id').some(node=>node.checked),false);
});


for(const path of ['/make/prepare','/pack/prepare'])for(const language of ['en','es'])for(const method of ['selected','typed'])test(`${path} refuses a wrong-lot overdraw when total stock covers the ingredient: ${language} ${method}`,async()=>{
  const draft=plain(stockedDraft),group=draft.lot_confirmation_group.ingredients[0];group.candidates.find(lot=>lot.lot_id===11).available_lb=20;group.total_available_lb=240;group.stock_shortage=false;
  const h=await harness({path,language,draft});
  if(method==='typed')typeStock(h,'BAG 100');else chooseStock(h,11);
  await checkStock(h);assert.equal(h.requests.length,1);assert.ok(h.find(node=>node.textContent.includes(language==='es'?'Elige el lote correcto o reparte entre varios lotes':'Pick the right lot or split across lots')));
  assert.equal(descendants(stockRow(h,11)).find(node=>node.name==='ingredient_lot_quantity').max,'20');
  chooseStock(h,11,20);chooseStock(h,12,80);await checkStock(h);assert.equal(h.requests.length,2);assert.deepEqual(h.requests[1].body.ingredient_lot_allocations,[{ingredient_product_id:1,lot_id:11,quantity_lb:20},{ingredient_product_id:1,lot_id:12,quantity_lb:80}]);
});
for(const path of ['/make/prepare','/pack/prepare'])for(const language of ['en','es'])test(`${path} retains stock-positive overage when the ingredient total is short: ${language}`,async()=>{
  const draft=plain(stockedDraft),group=draft.lot_confirmation_group.ingredients[0];group.candidates=group.candidates.filter(lot=>lot.lot_id===11);group.candidates[0].available_lb=20;group.total_available_lb=20;group.stock_shortage=true;
  const h=await harness({path,language,draft}),selected=chooseStock(h,11);assert.equal(selected.qty.max,undefined);assert.equal(selected.qty.value,'100');await checkStock(h);assert.equal(h.requests.length,2);assert.equal(h.requests[1].body.ingredient_lot_allocations[0].quantity_lb,100);
});
