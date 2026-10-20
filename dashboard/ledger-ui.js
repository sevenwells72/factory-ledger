/* Presentation and transport only. FL owns permissions, validation and posting. */
(function(root){
  'use strict';
  const esc = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text = (value,lang) => typeof value==='string'?value:(lang==='es'&&value?.message_es)||value?.message||value?.error_code||value?.code||'';
  function errorText(result,lang){const d=result?.detail||result;if(Array.isArray(d))return d.map(e=>text(e,lang)||e.msg).join('\n');return [text(d,lang),...(d?.blockers||[]).map(b=>text(b,lang))].filter(Boolean).join('\n')||'Request failed / La solicitud falló';}
  function recorded(result){return typeof result?.receipt_number==='string' && result.receipt_number.trim().length>0 && result.success!==false && result.posted!==false && !result.held && !['awaiting_approval','rejected','prepared'].includes(result.status||result.ticket_status);}
  // XR identifies the decision. A resolve response's receipt_number is its SOURCE posting.
  // Rejection can record an XR while posted=false / ticket_status=rejected for the held stock.
  function decisionOutcome(action,result){
    const receipt=result?.resolution_receipt_number;
    if(typeof receipt!=='string'||!/^XR-\d{6}-\d{3,}$/.test(receipt)||result.success===false||result.held)return {tone:'bad'};
    let status;
    if(action==='approve'){if(!recorded(result))return {tone:'bad'};status='approved';}
    else if(action==='reject'){if(!result.rejected)return {tone:'bad'};status='declined';}
    else if(action==='resolve')status=['resolved','waived'].includes(result.status)?result.status:['open','escalated'].includes(result.status)?'still_open':null;
    if(!status)return {tone:'bad'};
    const postingReceipt=action==='approve'?result.receipt_number:result.resolution_posting_receipt_number;
    return {tone:'good',receipt,status,...(typeof postingReceipt==='string'&&postingReceipt.trim()?{postingReceipt}:{})};
  }
  // FL's actor-specific allowed_actions is authoritative; absent/empty means no controls.
  function resolutionChoices(item){
    return Array.isArray(item?.allowed_actions)?item.allowed_actions.map(action=>({approve:'approved',reject:'declined'}[action]||action)):[];
  }
  // Owner approve: note optional (blank is omitted). Reject, write-off and the rest need one.
  function decisionRequest(kind,note,extra={}){
    const text=String(note??'').trim();
    if(kind==='approved')return {action:'approve',body:text?{note:text}:{}};
    if(kind==='declined')return {action:'reject',body:{resolution_kind:'declined',note:text}};
    return {action:'resolve',body:{resolution_kind:kind,note:text,...extra}};
  }
  const noteRequired=kind=>!['approved','identified'].includes(kind);
  function voidChoices(receipt,enabled){
    return enabled===true?(receipt?.transactions||[]).filter(t=>Number.isInteger(t.id)&&t.allowed_actions?.includes('void')):[];
  }
  function voidRequest(transactionId,note=''){
    if(!Number.isInteger(transactionId)||transactionId<=0)throw Error('Invalid entry');
    return {path:'/void/'+transactionId+'/prepare',body:{reason:'data_entry_error',client_source:'dashboard',...(note.trim()?{note:note.trim()}:{})}};
  }
  function receiptState(receipt){
    if(receipt?.status!=='committed')return {tone:'bad',state:'not_recorded'};
    const txns=receipt.transactions||[],voided=txns.filter(t=>t.effective_status==='voided').length;
    const voidReceipt=receipt.void_receipt_number||txns.map(t=>t.void_receipt_number).find(Boolean)||null;
    if(receipt.effective_status==='voided'||(txns.length&&voided===txns.length))return {tone:'voided',state:'voided',voidReceipt};
    if(receipt.effective_status==='partially_voided'||voided)return {tone:'voided',state:'partly_voided',voidReceipt};
    return {tone:'good',state:'recorded'};
  }
  const TIMEOUT=['No answer from FL within 20 seconds — this is NOT confirmed. Press Record again to check: it reuses the same ticket and cannot record twice.','FL no respondió en 20 segundos — esto NO está confirmado. Presiona Registrar otra vez para verificar: usa el mismo ticket y no puede registrarse dos veces.'];
  const OFFLINE=['Could not reach FL — this is NOT confirmed. Check the connection, then press Record again (same ticket).','No se pudo contactar a FL — esto NO está confirmado. Revisa la conexión y presiona Registrar otra vez (mismo ticket).'];
  function transportText(error,lang){const pick=pair=>pair[lang==='es'?1:0];if(error?.name==='AbortError')return pick(TIMEOUT);if(error instanceof TypeError)return pick(OFFLINE);return null;}
  function groups(rows){const map=new Map();[...rows].sort((a,b)=>Number(!!b.overdue)-Number(!!a.overdue)||String(a.due_at||'z').localeCompare(String(b.due_at||'z'))||a.id-b.id).forEach(r=>{if(!map.has(r.kind))map.set(r.kind,[]);map.get(r.kind).push(r);});return [...map];}
  const formatNumber=value=>Number.isFinite(Number(value))?new Intl.NumberFormat('en-US',{maximumFractionDigits:2}).format(Number(value)):String(value??'');
  const exceptionNames={UNIDENTIFIED_LOT:['Unidentified lot','Lote sin identificar'],PRE_MAKE_ADJUST:['Adjustment before a make','Ajuste antes de una producción'],SHORTAGE:['Stock shortage','Faltante de inventario'],NEGATIVE_BALANCE:['Negative stock balance','Saldo de inventario negativo'],LARGE_CORRECTION:['Large correction — owner review','Corrección grande — revisión del dueño'],KOSHER_ATTESTATION:['Kosher approval','Aprobación kosher'],LATE_ENTRY:['Late entry','Registro tardío'],SHIPMENT_PROOF_MISSING:['Shipment photo missing','Falta foto del envío'],BOL_PHOTO_MISSING:['BOL photo missing','Falta foto BOL'],ASSISTANT_OFFLINE:['Assistant offline','Asistente sin conexión']};
  function exceptionLabel(kind,lang='en'){return exceptionNames[kind]?.[lang==='es'?1:0]||codeText(kind,lang);}
  function exceptionQueue(rows){
    const unresolved=item=>['open','escalated'].includes(item.status),overdue=item=>unresolved(item)&&!!item.overdue;
    const time=value=>Date.parse(value)||0;
    return [...rows].sort((a,b)=>Number(unresolved(b))-Number(unresolved(a))
      ||Number(overdue(b))-Number(overdue(a))
      ||(overdue(a)&&overdue(b)?time(a.due_at)-time(b.due_at):0)
      ||Number(openingStock(a))-Number(openingStock(b))
      ||time(b.opened_at||b.created_at)-time(a.opened_at||a.created_at)||b.id-a.id);
  }
  function openingStock(item){return item.is_opening_stock===true;}
  function dataHtml(value){
    if(value===null||value===undefined)return '—';
    if(Array.isArray(value))return value.length?'<ul>'+value.map(v=>'<li>'+dataHtml(v)+'</li>').join('')+'</ul>':'—';
    if(typeof value==='object')return '<dl>'+Object.entries(value).filter(([k])=>!['ticket','payload_hash','key_hash','session_token'].includes(k)).map(([k,v])=>'<dt>'+esc(k.replaceAll('_',' '))+'</dt><dd>'+dataHtml(v)+'</dd>').join('')+'</dl>';
    return esc(typeof value==='number'?formatNumber(value):value);
  }
  const labels={
    draft:['Reviewed draft','Borrador revisado'],response:['FL result','Resultado FL'],type:['Entry type','Tipo de entrada'],effective_status:['Effective status','Estado efectivo'],lot_confirmations:['Lot confirmations','Confirmaciones de lote'],method:['Confirmation method','Método de confirmación'],value:['Entered code','Código ingresado'],previous_supplier_lot_code:['Previous supplier lot','Lote de proveedor anterior'],exception_id:['Exception #','Excepción #'],product_name:['Product','Producto'],product:['Product','Producto'],source_product:['Bulk batch','Lote a granel'],target_product:['Packed product','Producto empacado'],supplier:['Supplier','Proveedor'],supplier_name:['Supplier','Proveedor'],shipper_name:['Supplier','Proveedor'],supplier_lot_code:['Supplier lot','Lote del proveedor'],lot_code:['Lot','Lote'],output_lot_code:['Output lot','Lote de salida'],bol_reference:['BOL','BOL'],cases:['Cases','Cajas'],cases_received:['Cases','Cajas'],case_size_lb:['lb per case','lb por caja'],case_weight_lb:['lb per case','lb por caja'],quantity:['Quantity','Cantidad'],quantity_lb:['lb','lb'],total_lb:['Total lb','Total lb'],total_output_lb:['Output lb','Producción lb'],batches:['Batches','Lotes de producción'],delta_lb:['Change (lb)','Cambio (lb)'],current_balance_lb:['FL balance (lb)','Saldo FL (lb)'],new_balance_lb:['Resulting balance (lb)','Saldo resultante (lb)'],reason:['Reason','Motivo'],reason_code:['Reason','Motivo'],note:['Note','Nota'],notes:['Notes','Notas'],happened_at:['Happened','Ocurrió'],entered_at:['Recorded','Registrado'],actor:['Entered by','Ingresado por'],prepared_by:['Prepared by','Preparado por'],name:['Name','Nombre'],role:['Role','Rol'],ingredients:['Ingredients','Ingredientes'],ingredient_name:['Ingredient','Ingrediente'],substitute_name:['Substitute','Sustituto'],substitutions:['Substitutions','Sustituciones'],input_plan:['Lots to consume','Lotes a consumir'],lines:['Lines','Líneas'],needed_lb:['Needed lb','lb necesarios'],available_lb:['Available lb','lb disponibles'],short_lb:['Short lb','lb faltantes'],shortages:['Shortages','Faltantes'],customer:['Customer','Cliente'],order:['Order','Pedido'],photo:['Photo','Foto'],identity_status:['Lot identity','Identidad del lote'],identify_by:['Identify by','Identificar antes de'],message:['FL message','Mensaje FL'],message_es:['Mensaje FL','Mensaje FL'],kosher_tier:['Kosher tier','Nivel kosher'],kosher_label:['Kosher label','Etiqueta kosher'],production_warning:['Production warning','Advertencia de producción'],balance_warning:['Balance warning','Advertencia de saldo'],correction_review:['Correction review','Revisión de corrección'],photo_required:['Photo required','Foto requerida'],action:['Action','Acción'],counted_lb:['Counted lb','lb contadas'],resolution_kind:['Resolution','Resolución'],receipt_number:['Receipt #','Recibo #'],resolution_receipt_number:['Decision receipt #','Recibo de decisión #'],resolution_posting_receipt_number:['Correction receipt #','Recibo de corrección #'],source_receipt_number:['Source receipt #','Recibo de origen #'],void_receipt_number:['Void receipt #','Recibo de anulación #'],status:['Status','Estado'],transactions:['Ledger entries','Entradas'],lots:['Lots','Lotes'],shipment:['Shipment','Envío'],proof_status:['Shipment proof','Prueba de envío'],proof_due_at:['Photo due','Fecha límite de foto']
  };
  Object.assign(labels,{contact_name:['Contact name','Contacto'],email:['Email','Correo'],phone:['Phone','Teléfono'],address:['Address','Dirección'],transaction_id:["Entry #","Entrada #"],voided_transaction_id:["Voided entry #","Entrada anulada #"],"summary": ["Summary", "Resumen"], "summary_es": ["Resumen", "Resumen"], "customer_po": ["Customer PO", "PO del cliente"], "customer_po_status": ["PO status", "Estado PO"], "order_number": ["Order number", "Número de pedido"], "order_number_provisional": ["Provisional order number", "Número provisional del pedido"], "requested_ship_date": ["Ship by", "Enviar antes de"], "order_date": ["Order date", "Fecha del pedido"], "unit": ["Unit", "Unidad"], "unit_price": ["Price per unit", "Precio por unidad"], "amount": ["Line total", "Total de línea"], "total": ["Order total", "Total del pedido"], "lines_added": ["Lines added", "Líneas agregadas"], "expected_receipt": ["Expected delivery", "Entrega esperada"], "expected_receipts": ["Expected deliveries", "Entregas esperadas"], "created": ["Expected deliveries", "Entregas esperadas"], "expected_qty": ["Expected lb", "lb esperadas"], "expected_qty_lb": ["Expected lb", "lb esperadas"], "expected_date": ["Expected date", "Fecha esperada"], "reference_number": ["Reference", "Referencia"], "customer_description": ["Customer description", "Descripción del cliente"], "vendor_description": ["Vendor description", "Descripción del proveedor"], "is_service": ["Service line", "Línea de servicio"], "state": ["Order state", "Estado del pedido"], "warnings": ["FL warnings", "Advertencias FL"]});
  function plantTime(value,lang='en'){return value?new Date(value).toLocaleString(lang==='es'?'es-US':'en-US',{timeZone:'America/New_York',year:'numeric',month:'short',day:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'}):'—';}
  function autoSubstitutionsHtml(items){
    return (items||[]).filter(item=>item.message_en&&item.message_es).map(item=>'<p class="auto-substitution" role="status" data-en="'+esc(item.message_en)+'" data-es="'+esc(item.message_es)+'">'+esc(api.lang==='es'?item.message_es:item.message_en)+'</p>').join('');
  }
  const codeFields=new Set(['reason','reason_code','identity_status','status','effective_status','proof_status','role','method','action','resolution_kind','type','photo','state','customer_po_status']);
  const codeLabels={physical_count:['Physical count','Conteo físico'],missing_receipt:['Missing receipt','Falta recibo'],missing_production:['Missing production','Falta producción'],wrong_lot_used:['Wrong lot used','Se usó el lote equivocado'],damage_disposal:['Damage or disposal','Daño o desecho'],unrecorded_usage:['Unrecorded usage','Uso sin registrar'],data_entry_error:['Data-entry error','Error de captura'],unidentified:['Lot identity needs to be checked','Hay que verificar la identidad del lote'],unidentified_lot:['Lot identity needs to be checked','Hay que verificar la identidad del lote'],unknown:['Unknown','Desconocido'],photo_pending:['Photo pending','Foto pendiente'],awaiting_approval:['Waiting for owner approval','En espera de aprobación del dueño'],ship_order:['Ship order','Enviar pedido'],ship_standalone:['Ship','Enviar'],ship_bulk:['Ship multiple products','Enviar varios productos']};
  Object.assign(codeLabels,{committed:['Recorded','Registrado'],prepared:['Draft','Borrador'],posted:['Counts in stock','Cuenta en el inventario'],voided:['Voided','Anulado'],partially_voided:['Partly voided','Anulado en parte'],active:['Active','Activo'],identified:['Lot identity checked','Identidad del lote verificada'],receive:['Receive','Recibir'],make:['Make','Producir'],pack:['Pack','Empacar'],adjust:['Adjust','Ajustar'],found:['Found inventory','Inventario encontrado']});
  function codeText(value,lang='en'){
    const raw=String(value??''),pair=codeLabels[raw.toLowerCase()];if(pair)return pair[lang==='es'?1:0];
    if(/^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$/i.test(raw)){const words=raw.replaceAll('_',' ').toLowerCase();return words.charAt(0).toUpperCase()+words.slice(1);}
    return raw;
  }
  Object.assign(labels,{notes_es:['Note — Spanish','Nota — español'],found_location:['Found location','Lugar del hallazgo'],target_product_name:['Packed product','Producto empacado'],suggested_lots:['Suggested lots','Lotes sugeridos'],more_lots:['Other available lots','Otros lotes disponibles'],lots_used:['Shipped lots','Lotes enviados'],lines_shipped:['Shipped lines','Líneas enviadas'],shipped_lb:['Shipped lb','lb enviadas'],consumed_lb:['Used lb','lb usadas']});
  // A receipt combines an immutable draft with live shipment proof. Display the
  // live state throughout, without rewriting the stored draft or its evidence.
  function receiptData(receipt){
    const complete=receipt.shipment?.proof_status==='complete';
    const metadata=['create_customer','create_order','add_order_lines','update_order_line','cancel_order_line','update_order_header','update_order_status','mark_order_ready','cancel_order','close_order','reopen_order'].includes(receipt.action);
    function current(value){
      if(Array.isArray(value))return value.map(current);
      if(!value||typeof value!=='object')return value;
      return Object.fromEntries(Object.entries(value).filter(([key])=>!(complete&&key==='proof_due_at')).map(([key,item])=>[key,metadata&&key==='happened_at'&&receipt.happened_at?receipt.happened_at:complete&&key==='photo'?'attached':complete&&key==='proof_status'?'complete':current(item)]));
    }
    return current(receipt);
  }
  function makeDraftHtml(draft,compact=true){
    const ingredients=(draft.ingredients||[]).map(ingredient=>{
      const plan=(draft.input_plan||[]).filter(item=>item.product_id===ingredient.ingredient_id);
      const available=ingredient.lots||[];
      const suggested=plan.map(item=>({...available.find(lot=>lot.lot_code===item.lot_code),lot_code:item.lot_code,quantity_lb:item.quantity_lb}));
      const codes=new Set(plan.map(item=>item.lot_code));
      const {lots,...rest}=ingredient;
      return {...rest,suggested_lots:suggested,more_lots:available.filter(lot=>!codes.has(lot.lot_code))};
    });
    return draftHtml({...draft,ingredients},undefined,{compactLots:compact});
  }
  function draftHtml(value,field,options={}){
    if(field==='photo'&&value==='attached'||field==='proof_status'&&value==='complete')return '<span data-en="Photo attached ✓" data-es="Foto adjunta ✓">'+(api.lang==='es'?'Foto adjunta ✓':'Photo attached ✓')+'</span>';
    if(field==='more_lots'&&Array.isArray(value))return '<details class="more-lots"'+(options.compactLots?'':' open')+'><summary data-en="Show more lots" data-es="Mostrar más lotes">'+(api.lang==='es'?'Mostrar más lotes':'Show more lots')+'</summary>'+draftHtml(value,undefined,options)+'</details>';
    if((field==='reason'||field==='reason_code')&&value&&typeof value==='object'&&!Array.isArray(value)){
      const label=value[api.lang==='es'?'label_es':'label_en']||value.label_en||value.label||codeText(value.code,api.lang);
      return esc(label)+(value.note?'<p>'+esc(value.note)+'</p>':'');
    }
    if(Array.isArray(value))return '<ul'+(field==='suggested_lots'?' class="suggested-lots"':'')+'>'+value.map(v=>'<li>'+draftHtml(v,undefined,options)+'</li>').join('')+'</ul>';
    if(value&&typeof value==='object')return autoSubstitutionsHtml(value.auto_substitutions)+'<dl>'+Object.entries(value).filter(([k,v])=>labels[k]&&k!=='summary_es'&&v!==null&&v!==undefined&&!(Array.isArray(v)&&!v.length)).map(([k,v])=>'<dt data-en="'+esc(labels[k][0])+'" data-es="'+esc(labels[k][1])+'">'+esc(labels[k][api.lang==='es'?1:0])+'</dt><dd>'+(k==='summary'&&value.summary_es?'<span data-en="'+esc(v)+'" data-es="'+esc(value.summary_es)+'">'+esc(api.lang==='es'?value.summary_es:v)+'</span>':['happened_at','entered_at','proof_due_at'].includes(k)?esc(plantTime(v,api.lang)):draftHtml(v,k,options))+'</dd>').join('')+'</dl>';
    return esc(codeFields.has(field)?codeText(value,api.lang):(typeof value==='number'||/^(?:quantity|quantity_lb|cases|cases_received|batches|counted_lb|.*_lb|unit_price|amount|total)$/.test(field||'')&&value!==null&&value!==''&&Number.isFinite(Number(value)))?formatNumber(value):value);
  }
  const api={formatNumber,exceptionLabel,exceptionQueue,openingStock,esc,text,errorText,recorded,decisionOutcome,resolutionChoices,decisionRequest,noteRequired,voidChoices,voidRequest,receiptState,transportText,groups,dataHtml,draftHtml,makeDraftHtml,receiptData,autoSubstitutionsHtml,plantTime,codeText};
  if(typeof module!=='undefined')module.exports=api;
  root.FLLedger=api;
  if(!root.document)return;
  api.lang=sessionStorage.getItem('fl-ledger-language')||'en';
  // t() returns a String that remembers both languages; el() keeps that text node so a
  // language switch can relabel in place (relabel) without rebuilding forms or drafts.
  const labelled=new Set();
  api.t=(en,es)=>Object.assign(new String(api.lang==='es'?es:en),{en,es});
  api.el=(tag,value,parent)=>{const e=document.createElement(tag);if(value!==undefined){const node=document.createTextNode(String(value));if(value instanceof String&&value.en!==undefined){node.labels=[value.en,value.es];labelled.add(node);}e.append(node);}if(parent)parent.append(e);return e;};
  api.relabel=()=>{for(const node of labelled){if(!node.isConnected){labelled.delete(node);continue;}node.nodeValue=node.labels[api.lang==='es'?1:0];}};
  api.translate=()=>{document.documentElement.lang=api.lang;document.querySelectorAll('[data-en]').forEach(e=>e.textContent=e.dataset[api.lang]||e.dataset.en);const b=document.getElementById('language');if(b)b.textContent=api.lang==='en'?'ES':'EN';};
  api.notice=(message,kind='bad')=>{const box=document.getElementById('notice');box.hidden=false;box.className=kind;box.textContent=message;};
  api.failure=error=>api.notice(error?.accessUnavailable?root.FLDashboard.accessMessage(api.lang):error?.transport?error.message:api.t('NOT recorded — ','NO registrado — ')+(error?.result?errorText(error.result,api.lang):error.message||String(error)));
  api.base=root.FLDashboard.base;
  let actor=null;
  api.session=FLKeySessions.create({base:api.base,fetch:root.fetch.bind(root),storage:root.sessionStorage,config:root.FLDashboard.config,changed:()=>paintIdentity()});
  function paintIdentity(){actor=api.session.actor||actor;const identity=document.getElementById('identity');if(!identity)return;const current=api.session.actor;
    identity.hidden=!root.FLDashboard.writesEnabled;if(identity.hidden){identity.textContent='';document.getElementById('sign-in-form').hidden=true;document.getElementById('sign-out').hidden=true;return;}
    identity.textContent=current?api.t('Signed in as ','Sesión de ')+current.name:api.t('Sign in before recording','Inicia sesión antes de registrar');
    document.getElementById('sign-in-form').hidden=!!current;document.getElementById('sign-out').hidden=!current;
  }
  api.ready=root.FLDashboard.ready.then(async on=>{if(!on){api.session.clear();return false;}await api.session.init();return true;}).catch(error=>{api.failure(error);throw error;});api.ready.catch(()=>{});
  api.request=async(path,body,{read=false,form=false,background=false,binary=false}={})=>{
    await api.ready;
    if(!read&&!root.FLDashboard.writesEnabled)throw Error('Recording is unavailable / El registro no está disponible');
    const key=api.session.token;
    if(!key&&!read)throw Error(api.t('Sign in before recording.','Inicia sesión antes de registrar.'));
    const headers={'X-API-Key':key||await root.FLDashboard.key(api.base,api.lang),...(background?{'X-FL-Background':'1'}:{})};if(body!==undefined&&!form)headers['Content-Type']='application/json';
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);let response,result;
    try{response=await fetch(api.base+path,{method:body===undefined?'GET':'POST',headers,body:body===undefined?undefined:form?body:JSON.stringify(body),signal:controller.signal,cache:'no-store'});result=await Promise.resolve().then(()=>binary&&response.ok?response.blob():response.json()).catch(error=>{if([401,403].includes(response.status))return {};throw error;});}
    catch(error){const plain=transportText(error,api.lang);if(!plain)throw error;const wrapped=Error(plain);wrapped.transport=true;throw wrapped;}finally{clearTimeout(timer);}
    if(response.status===401&&result.detail?.error_code==='SESSION_EXPIRED'){api.session.clear();actor=null;paintIdentity();}
    root.FLDashboard.checkReadAccess(response,{method:body===undefined?'GET':'POST',headers},api.lang,api.base);
    if(!response.ok){const error=Error(errorText(result,api.lang));error.result=result;error.status=response.status;throw error;}return result;
  };
  api.upload=async(file,kind)=>{const body=new FormData();body.append('file',file);body.append('kind',kind);return api.request('/attachments',body,{form:true});};
  api.bind=()=>{
    api.translate();document.getElementById('language').onclick=()=>{api.lang=api.lang==='en'?'es':'en';sessionStorage.setItem('fl-ledger-language',api.lang);api.translate();paintIdentity();root.dispatchEvent(new Event('ledger-language'));};
    document.getElementById('sign-in-form').onsubmit=async e=>{e.preventDefault();const input=document.getElementById('actor-key'),value=input.value;input.value='';const submit=e.submitter;submit.disabled=true;
      try{await api.ready;if(!root.FLDashboard.writesEnabled)throw Error('Recording is unavailable / El registro no está disponible');await api.session.signIn(value);paintIdentity();root.dispatchEvent(new Event('ledger-auth'));}catch(error){actor=null;paintIdentity();api.failure(error);}finally{submit.disabled=false;}};
    document.getElementById('sign-out').onclick=async()=>{try{await api.session.signOut();actor=null;paintIdentity();root.dispatchEvent(new Event('ledger-signout'));}catch(error){api.failure(error);}};
    api.ready.then(paintIdentity).catch(()=>{});
  };
})(typeof window==='undefined'?globalThis:window);
