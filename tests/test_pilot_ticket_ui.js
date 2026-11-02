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
async function harness({formula=false,language='en',path='/make/prepare',draft={},canCommit=true,onPrepare,initialAttachment=null,initialPhoto=null,photoKind='correction_photo',payload={},result={receipt_number:'MK-261020-001',success:true}}={}){
  const host=new Element('main'),notice=new Element('div'),requests=[],failures=[],uploads=[],busy=[],attachments=[];let cancelled=0,edited=0;
  const labelled=new Set();
  const label=(node,value)=>{node.textContent=String(value??'');if(value?.en!==undefined){node.labels=[value.en,value.es];labelled.add(node);}else{delete node.labels;labelled.delete(node);}return node;};
  const ui={...ledger,lang:language,t:(en,es)=>Object.assign(new String(ui.lang==='es'?es:en),{en,es}),setText:label,relabel(){for(const node of labelled)if(descendants(host).includes(node))node.textContent=node.labels[ui.lang==='es'?1:0];},el(tag,text,parent){const node=label(new Element(tag),text);parent?.append(node);return node;},makeDraftHtml:()=>'',draftHtml:()=>'',autoSubstitutionsHtml:()=>'',plantTime:()=>'',failure(error){failures.push(error.message);},notice(){},
    async upload(file,kind){uploads.push({file,kind});return {attachment_id:91,kind};},
    async request(url,body){requests.push({url,body:plain(body)});if(url.includes('/commit'))return plain(result);return {ticket:'ticket-1',payload_hash:'hash-1',draft:plain(draft),actor:{name:'Arturo'},can_commit:canCommit,...(onPrepare?onPrepare(body):{})};}};
  const root={FLLedger:ui,addEventListener(){},matchMedia:()=>({matches:true})};if(formula)vm.runInNewContext(fs.readFileSync('dashboard/make-formula.js','utf8'),{window:root});vm.runInNewContext(source,{window:root,document:{getElementById:()=>notice},FLReceipts:{link:receipt=>'receipt.html?number='+receipt,page:name=>name}});
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
test('lot confirmation list only sends typed codes: no lot picker, no lot override (Make uses the bag cards)',async()=>{
  const h=await harness({draft:makeDraft,canCommit:false});fillTags(h);await h.button('Check confirmations → new draft').onclick();await flush();assert.equal(h.find(node=>node.dataset.confirmLot!==undefined),undefined);const body=h.requests[1].body;assert.equal('ingredient_lots' in body,false);assert.deepEqual(body.lot_confirmations,[{lot_id:11,method:'last4',value:'-001'},{lot_id:21,method:'last4',value:'SUN1'}]);
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
// Formula-driven Make (make-formula.js): ingredient cards replace the per-lot rows.
const formulaDraft=confirmed=>({...makeDraft,formula_entry:{pans:2,pan_lb:100,output_lb:200,tolerance_pct:10,ingredients:[{product_id:1,product_name:'Oats',standard_lb:160,bags:[{lot_id:11,lot_code:'OATS-A-001',quantity_lb:160,confirmed}],candidates:[{lot_id:11,lot_code:'OATS-A-001',available_lb:500},{lot_id:12,lot_code:'OATS-B-001',available_lb:90}]}]}});
// Make-entry simplification PR-1: lots are confirmed by a tap (no typed code), the draft
// says plainly whether anything was recorded, and Record cannot post one Make twice.
const allText=node=>descendants(node).map(n=>String(n.textContent)).join('');
const lotButton=h=>h.find(node=>node.dataset.lotConfirm!==undefined);
const choice=(h,id)=>h.find(node=>node.dataset.lotId===String(id));
test('a tapped lot enables Record without a re-check and is sent as the selection',async()=>{
  const h=await harness({formula:true,draft:formulaDraft(false),canCommit:false,payload:{formula_entry:true,ingredient_lines:[]},onPrepare:()=>({blockers:[{code:'LOT_NOT_CONFIRMED',message:'Confirm Oats lot OATS-A-001'}]})});
  assert.equal(h.find(node=>node.name==='bag_code'),undefined,'no typed code');
  assert.equal(h.all(node=>node.dataset.blocker==='LOT_NOT_CONFIRMED').length,0,'untapped lots are counted, not listed as blockers');
  assert.equal(h.button('Record').disabled,true);
  assert.match(String(h.find(node=>node.dataset.pendingLots!==undefined).textContent),/^1 lot still needs a tap/);
  assert.match(String(lotButton(h).textContent),/^Tap to confirm: OATS-A-001/);
  lotButton(h).onclick();
  assert.match(String(lotButton(h).textContent),/^✓ OATS-A-001/);
  assert.equal(h.button('Record').disabled,false);assert.equal(h.requests.length,1,'no re-check for a tap');
  await h.button('Record').onclick();
  assert.deepEqual(h.requests.at(-1).body.lot_confirmations,[{lot_id:11,method:'selected',value:'OATS-A-001'}]);
  assert.match(allText(h.find(node=>node.dataset.entryStatus==='good')),/^RECORDED ✓ MK-261020-001/);
});
test('the draft says NOT RECORDED YET until Record succeeds',async()=>{
  const h=await harness({formula:true,draft:formulaDraft(true),payload:{formula_entry:true}});
  assert.match(allText(h.find(node=>node.dataset.entryStatus==='draft')),/^NOT RECORDED YET/);
});
test('changing pounds or the lot needs a re-check; the re-check sends lines and taps',async()=>{
  const h=await harness({formula:true,draft:formulaDraft(true),payload:{formula_entry:true,ingredient_lines:[]}});
  assert.equal(h.button('Record').disabled,false);
  const lb=h.find(node=>node.name==='bag_lb');lb.value='70';lb.oninput();
  assert.equal(h.button('Record').disabled,true,'pounds changed: FL must check first');
  choice(h,12).onclick();
  assert.match(String(lotButton(h).textContent),/^✓ OATS-B-001/,'a lot tapped in the list is an intentional selection');
  await h.button('Check lots and pounds → new draft').onclick();await flush();
  const body=h.requests[1].body;
  assert.deepEqual(body.ingredient_lines,[{ingredient_product_id:1,bags:[{lot_id:12,quantity_lb:70}],reason_code:null,note:null}]);
  assert.deepEqual(body.lot_confirmations,[{lot_id:12,method:'selected',value:'OATS-B-001'}]);
});
test('a re-check FL refused keeps the cards and Record off',async()=>{
  let calls=0;const h=await harness({formula:true,draft:formulaDraft(true),payload:{formula_entry:true},onPrepare:()=>(++calls>1?{draft:{},can_commit:false,blockers:[{code:'MAKE_REASON_REQUIRED',message:'Choose a reason for Salt.'}]}:{})});
  const lb=h.find(node=>node.name==='bag_lb');lb.value='120';lb.oninput();
  await h.button('Check lots and pounds → new draft').onclick();await flush();
  assert.ok(h.find(node=>node.name==='bag_lb'),'cards are still shown');assert.ok(h.find(node=>String(node.textContent).includes('Choose a reason for Salt')));
  assert.equal(h.button('Record').disabled,true);
});
test('other blockers still show on a formula draft',async()=>{
  const h=await harness({formula:true,draft:formulaDraft(false),canCommit:false,payload:{formula_entry:true},onPrepare:()=>({blockers:[{code:'LOT_NOT_CONFIRMED',message:'x'},{code:'MAKE_REASON_REQUIRED',message:'Choose a reason for Salt.'}]})});
  assert.deepEqual(h.all(node=>node.dataset.blocker!==undefined).map(node=>node.dataset.blocker),['MAKE_REASON_REQUIRED']);
  lotButton(h).onclick();assert.equal(h.button('Record').disabled,true,'a tap never clears another blocker');
});
test('outside the Make lot cards each confirmation blocker is still listed',async()=>{
  const blockers=[{code:'LOT_NOT_CONFIRMED',message:'Confirm Oats lot OATS-A-001'},{code:'LOT_NOT_CONFIRMED',message:'Confirm Oats lot OATS-B-001'}];
  const h=await harness({draft:makeDraft,canCommit:false,onPrepare:()=>({blockers})});
  assert.equal(h.all(node=>node.dataset.blocker==='LOT_NOT_CONFIRMED').length,2);
});
test('Make warnings that need no tick are notes; ticked warnings and FL shortage text are unchanged',async()=>{
  const warnings=[{code:'INGREDIENT_OFF_RECIPE',requires_ack:true,message:'More than 10% off the recipe.'},{code:'INGREDIENT_CHANGED',requires_ack:false,message:'Not used or swapped.'},{code:'WILL_CREATE_SHORTAGE',requires_ack:false,message:'Honey lot H-1 has 30 lb; posting 40 lb leaves it short 10 lb.'}];
  const h=await harness({formula:true,draft:{...formulaDraft(true),shortages:[{product_name:'Honey',on_hand_lb:30,short_lb:10}]},payload:{formula_entry:true},onPrepare:()=>({warnings})});
  assert.deepEqual(h.all(node=>node.dataset.warning!==undefined).map(node=>node.dataset.warning),['INGREDIENT_OFF_RECIPE']);
  assert.deepEqual(h.all(node=>node.dataset.warningInfo!==undefined).map(node=>node.dataset.warningInfo),['INGREDIENT_CHANGED','WILL_CREATE_SHORTAGE']);
  const shortage=h.all(node=>node.dataset.shortageWarning!==undefined);assert.equal(shortage.length,1);assert.match(String(shortage[0].textContent),/short 10 lb/);
  h.find(node=>node.dataset.warning==='INGREDIENT_OFF_RECIPE').checked=true;await h.button('Record').onclick();
  assert.deepEqual(h.requests.at(-1).body.acknowledged_warnings,['INGREDIENT_OFF_RECIPE']);
});
test('a lot with fewer book pounds says how to add a second lot or record a shortage',async()=>{
  const draft=formulaDraft(true);draft.formula_entry.ingredients[0].bags[0].lot_id=12;draft.formula_entry.ingredients[0].bags[0].lot_code='OATS-B-001';
  const h=await harness({formula:true,draft,payload:{formula_entry:true}});
  const short=h.find(node=>node.dataset.bagShort!==undefined);assert.equal(short.hidden,false);assert.match(String(short.textContent),/only 90 lb.*add it with its pounds.*shortage/);
});
test('the re-check button sits in the record bar and appears only after an edit',async()=>{
  const h=await harness({formula:true,draft:formulaDraft(true),payload:{formula_entry:true}});
  const bar=h.find(node=>node.className==='record-bar sticky'),check=h.button('Check lots and pounds → new draft'),hint=h.find(node=>node.dataset.updateHint!==undefined);
  assert.ok(bar.children.includes(check)&&bar.children.includes(h.button('Record')));
  assert.equal(check.hidden,true);assert.equal(hint.hidden,true);assert.equal(h.button('Record').disabled,false);
  h.find(node=>node.name==='bag_lb').value='120';h.find(node=>node.name==='bag_lb').oninput();
  assert.equal(check.hidden,false);assert.equal(check.className,'primary');assert.equal(hint.hidden,false);assert.equal(h.button('Record').disabled,true);
});
test('re-checking keeps the draft on screen, and a failed re-check keeps it with Record off',async()=>{
  let release;const h=await harness({formula:true,draft:formulaDraft(true),payload:{formula_entry:true}});
  const original=h.ui.request;h.ui.request=(url,body)=>url==='/make/prepare'?new Promise((resolve,reject)=>{release=reject;}):original(url,body);
  h.find(node=>node.name==='bag_lb').value='120';h.find(node=>node.name==='bag_lb').oninput();
  h.button('Check lots and pounds → new draft').onclick();await flush();
  assert.equal(h.host.children[0]['aria-busy'],'true');assert.ok(h.find(node=>node.name==='bag_lb'),'cards stay while FL checks');assert.equal(h.find(node=>node.textContent==='Preparing…'),undefined);
  release(Object.assign(Error('Could not reach FL'),{transport:true}));await flush();
  assert.equal(h.host.children[0]['aria-busy'],'false');assert.equal(h.find(node=>node.name==='bag_lb').value,'120');assert.equal(h.button('Record').disabled,true);
  assert.equal(h.button('Try prepare again'),undefined);
  assert.match(allText(h.find(node=>node.dataset.entryStatus==='bad')),/^NOT RECORDED/);
});
function failingCommit(h,errors){const original=h.ui.request;h.ui.request=async(url,body)=>{if(url.includes('/commit')&&errors.length){h.requests.push({url,body:plain(body)});throw errors.shift();}return original(url,body);};}
test('Record with no answer says NOT CONFIRMED, keeps entries, and retries the same ticket',async()=>{
  const h=await harness({formula:true,draft:formulaDraft(true),payload:{formula_entry:true}});
  failingCommit(h,[Object.assign(Error('Could not reach FL'),{transport:true})]);
  await h.button('Record').onclick();
  assert.match(allText(h.find(node=>node.dataset.entryStatus==='bad')),/^NOT CONFIRMED — FL did not answer.*will not record the same Make twice/);
  assert.equal(h.button('Record').disabled,false);assert.equal(h.find(node=>node.name==='bag_lb').value,'160');
  await h.button('Record').onclick();
  const commits=h.requests.filter(r=>r.url.includes('/commit'));assert.equal(commits.length,2);assert.equal(new Set(commits.map(r=>r.url)).size,1,'the retry uses the same ticket');
  assert.ok(h.find(node=>node.dataset.entryStatus==='good'));
});
test('an expired draft keeps entries and "Check again" re-prepares with them',async()=>{
  const h=await harness({formula:true,draft:formulaDraft(true),payload:{formula_entry:true,ingredient_lines:[]}});
  failingCommit(h,[Object.assign(Error('expired'),{status:409,result:{detail:{error_code:'TICKET_EXPIRED',message:'Ticket expired; prepare again.'}}})]);
  await h.button('Record').onclick();
  const box=h.find(node=>node.dataset.entryStatus==='bad');assert.match(allText(box),/^NOT RECORDED — Ticket expired/);
  await h.button('Check again (keeps my entries)').onclick();await flush();
  const again=h.requests.at(-1);assert.equal(again.url,'/make/prepare');
  assert.deepEqual(again.body.ingredient_lines,[{ingredient_product_id:1,bags:[{lot_id:11,quantity_lb:160}],reason_code:null,note:null}]);
  assert.deepEqual(again.body.lot_confirmations,[{lot_id:11,method:'selected',value:'OATS-A-001'}]);
});
test('a Make FL already recorded shows its receipt and never records again',async()=>{
  const h=await harness({formula:true,draft:formulaDraft(true),payload:{formula_entry:true}});
  failingCommit(h,[Object.assign(Error('dup'),{status:409,result:{detail:{error_code:'MAKE_ALREADY_RECORDED',message:'already',receipt_number:'MK-261102-007'}}})]);
  await h.button('Record').onclick();
  assert.match(allText(h.find(node=>node.dataset.entryStatus==='good')),/^Already RECORDED as MK-261102-007/);
  assert.equal(h.button('Record').disabled,true);
});
test('a draft for a Make already recorded says so at the top',async()=>{
  const h=await harness({formula:true,draft:{...formulaDraft(true),already_recorded:{receipt_number:'MK-261102-007',status:'committed'}},canCommit:false,payload:{formula_entry:true}});
  assert.match(allText(h.find(node=>node.dataset.entryStatus==='good')),/^Already RECORDED as MK-261102-007/);
  assert.equal(h.button('Record').disabled,true);
});
test('a double tap on Record sends one commit',async()=>{
  const h=await harness({formula:true,draft:formulaDraft(true),payload:{formula_entry:true}});
  const record=h.button('Record');await Promise.all([record.onclick(),record.onclick()]);
  assert.equal(h.requests.filter(r=>r.url.includes('/commit')).length,1);
});
test('the entries on the draft can be carried back to the form',async()=>{
  const h=await harness({formula:true,draft:formulaDraft(false),canCommit:false,payload:{formula_entry:true}});
  lotButton(h).onclick();const kept=plain(h.controller.entries());
  assert.deepEqual(kept.lines,[{ingredient_product_id:1,bags:[{lot_id:11,quantity_lb:160}],reason_code:null,note:null}]);
  assert.deepEqual(kept.confirmations,[{lot_id:11,method:'selected',value:'OATS-A-001'}]);
});
