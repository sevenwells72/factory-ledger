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
const isVisible=node=>!!node&&!node.hidden&&(!node.parent||isVisible(node.parent));
const plain=value=>JSON.parse(JSON.stringify(value));
async function harness({language='en',path='/make/prepare',draft={},canCommit=true,onPrepare,initialAttachment=null,initialPhoto=null,photoKind='correction_photo',payload={}}={}){
  const host=new Element('main'),notice=new Element('div'),requests=[],failures=[],uploads=[],busy=[],attachments=[];let cancelled=0,edited=0;
  const labelled=new Set();
  const label=(node,value)=>{node.textContent=String(value??'');if(value?.en!==undefined){node.labels=[value.en,value.es];labelled.add(node);}else{delete node.labels;labelled.delete(node);}return node;};
  const ui={...ledger,lang:language,t:(en,es)=>Object.assign(new String(ui.lang==='es'?es:en),{en,es}),setText:label,relabel(){for(const node of labelled)if(descendants(host).includes(node))node.textContent=node.labels[ui.lang==='es'?1:0];},el(tag,text,parent){const node=label(new Element(tag),text);parent?.append(node);return node;},makeDraftHtml:()=>'',draftHtml:()=>'',autoSubstitutionsHtml:()=>'',plantTime:()=>'',failure(error){failures.push(error.message);},notice(){},
    async upload(file,kind){uploads.push({file,kind});return {attachment_id:91,kind};},
    async request(url,body){requests.push({url,body:plain(body)});if(url.includes('/commit'))return {receipt_number:'MK-261020-001',success:true};return {ticket:'ticket-1',payload_hash:'hash-1',draft:plain(draft),actor:{name:'Arturo'},can_commit:canCommit,...(onPrepare?onPrepare(body):{})};}};
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
const makeDraft={ingredients:[{ingredient_id:1,ingredient_name:'Oats',needed_lb:10},{ingredient_id:2,ingredient_name:'Sunflower Seeds',needed_lb:2}],input_plan:[{product_id:1,product_name:'Oats',lot_id:11,lot_code:'OATS-A-001',quantity_lb:10,confirmed:false},{product_id:2,product_name:'Sunflower Seeds',lot_id:21,lot_code:'SUN-SUN1',quantity_lb:2,confirmed:false}],lot_confirmation_group:{ingredients:[{product_id:1,product_name:'Oats',needed_lb:10,candidates:[{lot_id:11,lot_code:'OATS-A-001',available_lb:30,received_at:'2026-10-01T12:00:00Z'},{lot_id:12,lot_code:'OATS-B-001',available_lb:20,received_at:'2026-10-02T12:00:00Z'}]},{product_id:2,product_name:'Sunflower Seeds',needed_lb:2,candidates:[{lot_id:21,lot_code:'SUN-SUN1',available_lb:20,received_at:'2026-10-01T12:00:00Z'}]}]}};
// Simulate FL returning a newly validated plan, including changed lot identity.
function reviewedMake(body,initial=makeDraft){
  const draft=plain(initial),evidence=body.lot_confirmations||[];
  const choices=new Map((body.ingredient_lots||[]).map(item=>[item.ingredient_product_id,item.lot_id]));
  for(const item of body.substitutions||[])choices.set(item.substitute_product_id,item.lot_id);
  for(const [productId,lotId] of choices){
    const lots=draft.input_plan.filter(item=>item.product_id===productId),candidate=draft.lot_confirmation_group.ingredients.find(item=>item.product_id===productId)?.candidates.find(item=>item.lot_id===lotId);
    if(!candidate||!lots.length)continue;
    draft.input_plan=draft.input_plan.filter(item=>item.product_id!==productId);
    draft.input_plan.push({...lots[0],lot_id:lotId,lot_code:candidate.lot_code,quantity_lb:lots.reduce((sum,item)=>sum+item.quantity_lb,0)});
  }
  if(evidence.some(item=>!draft.input_plan.some(lot=>lot.lot_id===item.lot_id)))throw Error('UNUSED_LOT_CONFIRMATION');
  for(const lot of draft.input_plan){const confirmation=evidence.find(item=>item.lot_id===lot.lot_id);lot.confirmed=!!confirmation;if(confirmation)lot.confirmation=confirmation;else delete lot.confirmation;}
  const can_commit=draft.input_plan.every(item=>item.confirmed);
  return {draft,can_commit,blockers:can_commit?[]:[{code:'LOT_NOT_CONFIRMED',message:'Confirm every physical lot.'}]};
}
const byData=(h,key,value)=>h.find(node=>node.dataset[key]===String(value));
const checked=(h,lotId)=>byData(h,'makeLotConfirm',lotId)['aria-pressed']==='true';
const progress=h=>h.find(node=>node.dataset.makeProgress!==undefined).textContent;
const tap=async(h,lotId)=>{await byData(h,'makeLotConfirm',lotId).onclick();await flush();};
const selectLot=async(h,productId,lotId)=>{byData(h,'makeChange',productId).onclick();const choice=byData(h,'makeLotChoice',lotId);assert.equal(isVisible(choice),true);await choice.onclick();await flush();};
async function typeCode(h,lotId,method,value){
  byData(h,'makeTypeCode',lotId).onclick();const select=byData(h,'makeCodeMethod',lotId),input=byData(h,'makeCodeValue',lotId);
  assert.equal(select.parent.parent.hidden,false);select.value=method;select.oninput();input.value=value;input.oninput();await byData(h,'makeCodeCheck',lotId).onclick();await flush();
}
const compactHarness=({draft=makeDraft,onPrepare,...options}={})=>harness({draft,onPrepare:onPrepare||((body)=>reviewedMake(body,draft)),...options});

test('Make starts with one compact row per ingredient, selected lots unchecked and optional controls hidden',async()=>{
  const h=await compactHarness();assert.equal(h.all(node=>node.dataset.makeIngredient!==undefined).length,2);assert.equal(progress(h),'0 of 2 checked');assert.equal(checked(h,11),false);assert.equal(checked(h,21),false);assert.equal(h.button('Record').disabled,true);assert.equal('lot_confirmations' in h.requests[0].body,false);
  assert.ok(h.find(node=>node.className==='make-lot-suffix'&&node.textContent==='-001'));assert.equal(byData(h,'makeCodeMethod',11).parent.parent.hidden,true);assert.equal(isVisible(byData(h,'makeLotChoice',12)),false);
  await h.button('Record').onclick();assert.equal(h.requests.length,1);
});
test('each explicit tap advances the ingredient count and Record waits for every checked tag and a fresh server draft',async()=>{
  const h=await compactHarness();await tap(h,11);assert.equal(progress(h),'1 of 2 checked');assert.equal(checked(h,11),true);assert.equal(checked(h,21),false);assert.equal(h.requests.length,1);assert.equal(h.button('Record').disabled,true);
  await tap(h,21);assert.equal(progress(h),'2 of 2 checked');assert.equal(h.button('Record').disabled,false);assert.deepEqual(h.requests[1].body.lot_confirmations,[{lot_id:11,method:'tap',value:'OATS-A-001'},{lot_id:21,method:'tap',value:'SUN-SUN1'}]);
  await tap(h,11);assert.equal(progress(h),'1 of 2 checked');assert.equal(h.button('Record').disabled,true);await h.button('Record').onclick();assert.equal(h.requests.some(request=>request.url.includes('/commit')),false);
});
test('all local checks still cannot override a fresh server blocker',async()=>{
  const h=await compactHarness({onPrepare:body=>({...reviewedMake(body),can_commit:false,blockers:[{code:'RECIPE_CHANGED',message:'Prepare again after recipe review.'}]})});await tap(h,11);await tap(h,21);assert.equal(progress(h),'2 of 2 checked');assert.equal(h.button('Record').disabled,true);assert.ok(h.find(node=>node.textContent==='Prepare again after recipe review.'));await h.button('Record').onclick();assert.equal(h.requests.some(request=>request.url.includes('/commit')),false);
});
test('every FIFO split lot must be checked before its ingredient counts as checked',async()=>{
  const draft=plain(makeDraft);draft.input_plan=[{...draft.input_plan[0],quantity_lb:6},{...draft.input_plan[0],lot_id:12,lot_code:'OATS-B-001',quantity_lb:4},draft.input_plan[1]];
  const h=await compactHarness({draft});assert.equal(h.all(node=>node.dataset.makeIngredient!==undefined).length,2);await tap(h,11);assert.equal(progress(h),'0 of 2 checked');assert.equal(h.button('Record').disabled,true);await tap(h,12);assert.equal(progress(h),'1 of 2 checked');assert.equal(h.requests.length,1);await tap(h,21);assert.equal(progress(h),'2 of 2 checked');assert.equal(h.button('Record').disabled,false);assert.deepEqual(h.requests[1].body.lot_confirmations.map(item=>item.lot_id),[11,12,21]);
});
test('substitutions using the same ingredient share one row with the summed requirement',async()=>{
  const draft=plain(makeDraft);draft.ingredients=[{ingredient_id:1,ingredient_name:'Oats',needed_lb:4},{ingredient_id:1,ingredient_name:'Oats',needed_lb:6},draft.ingredients[1]];
  const payload={substitutions:[{ingredient_product_id:3,substitute_product_id:1,lot_id:11,reason_code:'physical_substitution'},{ingredient_product_id:4,substitute_product_id:1,lot_id:11,reason_code:'physical_substitution'}]};const h=await compactHarness({draft,payload}),row=byData(h,'makeIngredient',1),heading=descendants(row).find(node=>node.tag==='h3');assert.equal(h.all(node=>node.dataset.makeIngredient!==undefined).length,2);assert.match(heading.children.map(node=>node.textContent).join(''),/Oats · 10 lb needed/);assert.equal(descendants(row).filter(node=>node.dataset.makeLotConfirm!==undefined).length,1);await tap(h,11);assert.equal(progress(h),'1 of 2 checked');await tap(h,21);assert.equal(h.button('Record').disabled,false);assert.equal(h.requests.at(-1).body.lot_confirmations.filter(item=>item.lot_id===11).length,1);
});
for(const language of ['en','es'])test(`Change pages through every on-hand lot without confirming it in ${language}`,async()=>{
  const draft=plain(makeDraft);draft.lot_confirmation_group.ingredients[0].candidates=Array.from({length:14},(_,index)=>({lot_id:11+index,lot_code:index===0?'OATS-A-001':index===1?'OATS-B-001':'OATS-PAGE-'+String(index).padStart(3,'0'),available_lb:20,received_at:'2026-10-01T12:00:00Z'}));const h=await compactHarness({language,draft}),rows=()=>descendants(byData(h,'makeIngredient',1)).filter(node=>node.dataset.makeLotChoice!==undefined);assert.equal(rows().length,6);assert.equal(isVisible(rows()[0]),false);byData(h,'makeChange',1).onclick();assert.equal(isVisible(rows()[0]),true);assert.equal(progress(h),language==='en'?'0 of 2 checked':'0 de 2 verificados');
  const more=byData(h,'makeMoreLots',1);assert.equal(more.textContent,language==='en'?'More lots':'Más lotes');more.onclick();assert.equal(rows().length,12);assert.equal(more.hidden,false);more.onclick();assert.equal(rows().length,14);assert.equal(more.hidden,true);assert.equal(new Set(rows().map(node=>node.dataset.makeLotChoice)).size,14);assert.equal(h.requests.length,1);assert.equal(checked(h,11),false);assert.equal(h.button(language==='en'?'Record':'Registrar').disabled,true);
  await byData(h,'makeLotChoice',24).onclick();await flush();assert.deepEqual(h.requests[1].body.ingredient_lots,[{ingredient_product_id:1,lot_id:24}]);assert.equal('lot_confirmations' in h.requests[1].body,false);assert.equal(checked(h,24),false);await tap(h,24);assert.equal(progress(h),language==='en'?'1 of 2 checked':'1 de 2 verificados');
});
test('an ambiguous typed suffix offers exact lots and choosing the suggestion still requires an explicit tap',async()=>{
  const h=await compactHarness();await tap(h,21);await typeCode(h,11,'last4','-001');assert.equal(h.requests.length,1);assert.ok(h.button('OATS-A-001'));assert.ok(h.button('OATS-B-001'));await h.button('OATS-A-001').onclick();await flush();assert.deepEqual(h.requests[1].body.lot_confirmations,[{lot_id:21,method:'tap',value:'SUN-SUN1'}]);assert.equal(checked(h,11),false);assert.equal(h.button('Record').disabled,true);
  await tap(h,11);assert.deepEqual(h.requests[2].body.lot_confirmations,[{lot_id:21,method:'tap',value:'SUN-SUN1'},{lot_id:11,method:'tap',value:'OATS-A-001'}]);assert.equal(h.button('Record').disabled,false);
});
test('changing an already checked lot clears that ingredient evidence, preserves other checks and waits for the new bag check',async()=>{
  const h=await compactHarness();await tap(h,11);await tap(h,21);assert.equal(h.button('Record').disabled,false);await selectLot(h,1,12);
  assert.deepEqual(h.requests[2].body.ingredient_lots,[{ingredient_product_id:1,lot_id:12}]);assert.deepEqual(h.requests[2].body.lot_confirmations,[{lot_id:21,method:'tap',value:'SUN-SUN1'}]);assert.equal(checked(h,12),false);assert.equal(checked(h,21),true);assert.equal(progress(h),'1 of 2 checked');assert.equal(h.button('Record').disabled,true);
  await tap(h,12);assert.deepEqual(h.requests[3].body.lot_confirmations,[{lot_id:21,method:'tap',value:'SUN-SUN1'},{lot_id:12,method:'tap',value:'OATS-B-001'}]);assert.equal(h.button('Record').disabled,false);
});
test('choosing another lot never invents checks for that lot or an untouched ingredient',async()=>{
  const h=await compactHarness();await selectLot(h,1,12);assert.equal('lot_confirmations' in h.requests[1].body,false);assert.equal(progress(h),'0 of 2 checked');await tap(h,12);assert.equal(progress(h),'1 of 2 checked');assert.equal(checked(h,21),false);assert.equal(h.button('Record').disabled,true);assert.equal(h.requests.some(request=>request.url.includes('/commit')),false);
});
for(const [method,value] of [['last4','SUN1'],['full_code','SUN-SUN1'],['scan','SUN-SUN1']])test(`optional ${method} code confirmation keeps its honest method beside tap evidence`,async()=>{
  const h=await compactHarness();await typeCode(h,21,method,value);assert.deepEqual(h.requests[1].body.lot_confirmations,[{lot_id:21,method,value}]);assert.equal(checked(h,21),true);assert.equal(h.button('Record').disabled,true);await tap(h,11);assert.deepEqual(h.requests[2].body.lot_confirmations,[{lot_id:21,method,value},{lot_id:11,method:'tap',value:'OATS-A-001'}]);assert.equal(h.button('Record').disabled,false);
});
for(const language of ['en','es'])test(`a failed changed-lot prepare blocks stale Record and Retry recovers without stale lot evidence in ${language}`,async()=>{
  const record=language==='en'?'Record':'Registrar';let failNext=false;const h=await compactHarness({language,onPrepare:body=>{if(failNext){failNext=false;throw Error('offline while changing lot');}return reviewedMake(body);}});await tap(h,11);await tap(h,21);assert.equal(h.button(record).disabled,false);failNext=true;await selectLot(h,1,12);assert.match(h.failures.at(-1),/offline/);assert.equal(h.button(record).disabled,true);await h.button(record).onclick();assert.equal(h.requests.some(request=>request.url.includes('/commit')),false);
  await h.find(node=>node.dataset.makeRetry!==undefined).onclick();await flush();assert.deepEqual(h.failures,['offline while changing lot']);assert.equal(checked(h,12),false);assert.equal(checked(h,21),true);assert.equal(h.button(record).disabled,true);await tap(h,12);assert.equal(h.button(record).disabled,false);
});
test('switching language preserves compact checks, typed scanner evidence and the selected draft',async()=>{
  const h=await compactHarness();await typeCode(h,21,'scan','SUN-SUN1');const requestCount=h.requests.length;h.switchLanguage('es');assert.equal(progress(h),'1 de 2 verificados');assert.equal(checked(h,21),true);assert.equal(byData(h,'makeCodeMethod',21).value,'scan');assert.equal(byData(h,'makeCodeValue',21).value,'SUN-SUN1');assert.ok(h.button('Cambiar'));assert.ok(h.button('Escribir código'));assert.equal(h.requests.length,requestCount);
  await tap(h,11);assert.equal(h.button('Registrar').disabled,false);h.switchLanguage('en');assert.equal(progress(h),'2 of 2 checked');assert.equal(byData(h,'makeCodeMethod',21).value,'scan');assert.equal(byData(h,'makeCodeValue',21).value,'SUN-SUN1');assert.equal(h.button('Record').disabled,false);
});
for(const language of ['en','es'])test(`zero-stock warning stays visible while lot choices are hidden and still requires every tag in ${language}`,async()=>{
  const draft=plain(makeDraft);draft.shortages=[{product_id:1,product_name:'Oats',on_hand_lb:0,short_lb:10}];const h=await compactHarness({language,draft});const warning=byData(h,'makeShortage',1);assert.match(warning.textContent,language==='en'?/Only 0 lb on hand.*short 10 lb.*shortage/:/Solo hay 0 lb.*faltan 10 lb.*faltante/);assert.equal(warning.parent.hidden,false);assert.equal(isVisible(byData(h,'makeLotChoice',12)),false);assert.equal(h.button(language==='en'?'Record':'Registrar').disabled,true);await tap(h,11);await tap(h,21);assert.equal(h.button(language==='en'?'Record':'Registrar').disabled,false);
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

test('an alternate physical lot updates every substitution using that product and its override without checking the new tag',async()=>{
  const payload={substitutions:[{ingredient_product_id:3,substitute_product_id:1,lot_id:11,reason_code:'physical_substitution'},{ingredient_product_id:4,substitute_product_id:1,lot_id:11,reason_code:'physical_substitution'},{ingredient_product_id:5,substitute_product_id:2,lot_id:21,reason_code:'physical_substitution'}],ingredient_lots:[{ingredient_product_id:1,lot_id:11},{ingredient_product_id:2,lot_id:21}]};
  const h=await compactHarness({payload});await selectLot(h,1,12);const body=h.requests[1].body;
  assert.deepEqual(body.substitutions,[{...payload.substitutions[0],lot_id:12},{...payload.substitutions[1],lot_id:12},payload.substitutions[2]]);assert.deepEqual(body.ingredient_lots,[{ingredient_product_id:1,lot_id:12},{ingredient_product_id:2,lot_id:21}]);assert.equal('lot_confirmations' in body,false);assert.equal(checked(h,12),false);assert.equal(checked(h,21),false);await tap(h,12);assert.equal(progress(h),'1 of 2 checked');assert.equal(h.button('Record').disabled,true);assert.equal(h.requests.some(request=>request.url.includes('/commit')),false);
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
