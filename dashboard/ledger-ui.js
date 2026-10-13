/* Presentation and transport only. FL owns permissions, validation and posting. */
(function(root){
  'use strict';
  const esc = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text = (value,lang) => typeof value==='string'?value:(lang==='es'&&value?.message_es)||value?.message||value?.error_code||value?.code||'';
  function errorText(result,lang){const d=result?.detail||result;if(Array.isArray(d))return d.map(e=>text(e,lang)||e.msg).join('\n');return [text(d,lang),...(d?.blockers||[]).map(b=>text(b,lang))].filter(Boolean).join('\n')||'Request failed / La solicitud falló';}
  function recorded(result){return typeof result?.receipt_number==='string' && result.receipt_number.trim().length>0 && result.success!==false && result.posted!==false && !result.held && !['awaiting_approval','rejected','prepared'].includes(result.status||result.ticket_status);}
  function groups(rows){const map=new Map();[...rows].sort((a,b)=>Number(!!b.overdue)-Number(!!a.overdue)||String(a.due_at||'z').localeCompare(String(b.due_at||'z'))||a.id-b.id).forEach(r=>{if(!map.has(r.kind))map.set(r.kind,[]);map.get(r.kind).push(r);});return [...map];}
  function dataHtml(value){
    if(value===null||value===undefined)return '—';
    if(Array.isArray(value))return value.length?'<ul>'+value.map(v=>'<li>'+dataHtml(v)+'</li>').join('')+'</ul>':'—';
    if(typeof value==='object')return '<dl>'+Object.entries(value).filter(([k])=>!['ticket','payload_hash','key_hash','session_token'].includes(k)).map(([k,v])=>'<dt>'+esc(k.replaceAll('_',' '))+'</dt><dd>'+dataHtml(v)+'</dd>').join('')+'</dl>';
    return esc(value);
  }
  const labels={
    draft:['Reviewed draft','Borrador revisado'],response:['FL result','Resultado FL'],type:['Entry type','Tipo de entrada'],effective_status:['Effective status','Estado efectivo'],lot_confirmations:['Lot confirmations','Confirmaciones de lote'],method:['Confirmation method','Método de confirmación'],value:['Entered code','Código ingresado'],previous_supplier_lot_code:['Previous supplier lot','Lote de proveedor anterior'],exception_id:['Exception #','Excepción #'],product_name:['Product','Producto'],product:['Product','Producto'],source_product:['Source product','Producto de origen'],target_product:['Packed product','Producto empacado'],supplier:['Supplier','Proveedor'],supplier_name:['Supplier','Proveedor'],shipper_name:['Supplier','Proveedor'],supplier_lot_code:['Supplier lot','Lote del proveedor'],lot_code:['Lot','Lote'],output_lot_code:['Output lot','Lote de salida'],bol_reference:['BOL','BOL'],cases:['Cases','Cajas'],cases_received:['Cases','Cajas'],case_size_lb:['lb per case','lb por caja'],case_weight_lb:['lb per case','lb por caja'],quantity:['Quantity','Cantidad'],quantity_lb:['lb','lb'],total_lb:['Total lb','Total lb'],total_output_lb:['Output lb','Producción lb'],batches:['Batches','Lotes de producción'],delta_lb:['Change (lb)','Cambio (lb)'],current_balance_lb:['FL balance (lb)','Saldo FL (lb)'],new_balance_lb:['Resulting balance (lb)','Saldo resultante (lb)'],reason:['Reason','Motivo'],reason_code:['Reason','Motivo'],note:['Note','Nota'],notes:['Notes','Notas'],happened_at:['Happened','Ocurrió'],entered_at:['Recorded','Registrado'],actor:['Entered by','Ingresado por'],prepared_by:['Prepared by','Preparado por'],name:['Name','Nombre'],role:['Role','Rol'],ingredients:['Ingredients','Ingredientes'],ingredient_name:['Ingredient','Ingrediente'],substitute_name:['Substitute','Sustituto'],substitutions:['Substitutions','Sustituciones'],input_plan:['Lots to consume','Lotes a consumir'],lines:['Lines','Líneas'],needed_lb:['Needed lb','lb necesarios'],available_lb:['Available lb','lb disponibles'],short_lb:['Short lb','lb faltantes'],shortages:['Shortages','Faltantes'],customer:['Customer','Cliente'],order:['Order','Pedido'],photo:['Photo','Foto'],identity_status:['Lot identity','Identidad del lote'],identify_by:['Identify by','Identificar antes de'],message:['FL message','Mensaje FL'],message_es:['Mensaje FL','Mensaje FL'],kosher_tier:['Kosher tier','Nivel kosher'],kosher_label:['Kosher label','Etiqueta kosher'],production_warning:['Production warning','Advertencia de producción'],balance_warning:['Balance warning','Advertencia de saldo'],correction_review:['Correction review','Revisión de corrección'],photo_required:['Photo required','Foto requerida'],action:['Action','Acción'],counted_lb:['Counted lb','lb contadas'],resolution_kind:['Resolution','Resolución'],receipt_number:['Receipt #','Recibo #'],status:['Status','Estado'],transactions:['Ledger entries','Entradas'],lots:['Lots','Lotes'],shipment:['Shipment','Envío'],proof_status:['Shipment proof','Prueba de envío'],proof_due_at:['Photo due','Fecha límite de foto']
  };
  function plantTime(value,lang='en'){return value?new Date(value).toLocaleString(lang==='es'?'es-US':'en-US',{timeZone:'America/New_York',year:'numeric',month:'short',day:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'}):'—';}
  function draftHtml(value){
    if(Array.isArray(value))return '<ul>'+value.map(v=>'<li>'+draftHtml(v)+'</li>').join('')+'</ul>';
    if(value&&typeof value==='object')return '<dl>'+Object.entries(value).filter(([k,v])=>labels[k]&&v!==null&&v!==undefined&&!(Array.isArray(v)&&!v.length)).map(([k,v])=>'<dt>'+esc(labels[k][api.lang==='es'?1:0])+'</dt><dd>'+(['happened_at','entered_at','proof_due_at'].includes(k)?esc(plantTime(v,api.lang)):draftHtml(v))+'</dd>').join('')+'</dl>';
    return esc(value);
  }
  const api={esc,text,errorText,recorded,groups,dataHtml,draftHtml,plantTime};
  if(typeof module!=='undefined')module.exports=api;
  root.FLLedger=api;
  if(!root.document)return;
  api.lang=sessionStorage.getItem('fl-ledger-language')||'en';
  api.t=(en,es)=>api.lang==='es'?es:en;
  api.el=(tag,value,parent)=>{const e=document.createElement(tag);if(value!==undefined)e.textContent=value;if(parent)parent.append(e);return e;};
  api.translate=()=>{document.documentElement.lang=api.lang;document.querySelectorAll('[data-en]').forEach(e=>e.textContent=e.dataset[api.lang]||e.dataset.en);const b=document.getElementById('language');if(b)b.textContent=api.lang==='en'?'ES':'EN';};
  api.notice=(message,kind='bad')=>{const box=document.getElementById('notice');box.hidden=false;box.className=kind;box.textContent=message;};
  api.failure=error=>api.notice(api.t('NOT recorded — ','NO registrado — ')+(error?.result?errorText(error.result,api.lang):error.message||String(error)));
  api.base=root.FL_SESSION_API_BASE||(location.hostname.endsWith('netlify.app')?'https://fastapi-production-b73a.up.railway.app':location.origin);
  api.readKey='dashboard-key-2026';
  let legacy='', actor=null;
  api.session=FLKeySessions.create({base:api.base,fetch:root.fetch.bind(root),storage:root.sessionStorage,changed:()=>paintIdentity()});
  function paintIdentity(){actor=api.session.actor||actor;const identity=document.getElementById('identity');if(!identity)return;const current=api.session.enabled?api.session.actor:actor;
    identity.textContent=current?api.t('Signed in as ','Sesión de ')+current.name:api.t('Sign in before recording','Inicia sesión antes de registrar');
    document.getElementById('sign-in-form').hidden=!!current;document.getElementById('sign-out').hidden=!current;
  }
  api.ready=api.session.init().catch(error=>{api.failure(error);throw error;});api.ready.catch(()=>{});
  api.request=async(path,body,{read=false,form=false}={})=>{
    await api.ready;
    const key=api.session.enabled?api.session.token:legacy;
    if(!key&&!read)throw Error(api.t('Sign in before recording.','Inicia sesión antes de registrar.'));
    const headers={'X-API-Key':key||api.readKey};if(body!==undefined&&!form)headers['Content-Type']='application/json';
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);let response,result;
    try{response=await fetch(api.base+path,{method:body===undefined?'GET':'POST',headers,body:body===undefined?undefined:form?body:JSON.stringify(body),signal:controller.signal,cache:'no-store'});result=await response.json();}finally{clearTimeout(timer);}
    if(!response.ok){if(response.status===401&&result.detail?.error_code==='SESSION_EXPIRED'){api.session.clear();legacy='';actor=null;paintIdentity();}const error=Error(errorText(result,api.lang));error.result=result;error.status=response.status;throw error;}return result;
  };
  api.upload=async(file,kind)=>{const body=new FormData();body.append('file',file);body.append('kind',kind);return api.request('/attachments',body,{form:true});};
  api.bind=()=>{
    api.translate();document.getElementById('language').onclick=()=>{api.lang=api.lang==='en'?'es':'en';sessionStorage.setItem('fl-ledger-language',api.lang);api.translate();paintIdentity();root.dispatchEvent(new Event('ledger-language'));};
    document.getElementById('sign-in-form').onsubmit=async e=>{e.preventDefault();const input=document.getElementById('actor-key'),value=input.value;input.value='';const submit=e.submitter;submit.disabled=true;
      try{await api.ready;if(api.session.enabled)await api.session.signIn(value);else{legacy=value;const who=await api.request('/auth/whoami');actor=who.actor;if(!actor)throw Error('Use your personal actor key / Usa tu clave personal');}paintIdentity();root.dispatchEvent(new Event('ledger-auth'));}catch(error){legacy='';actor=null;paintIdentity();api.failure(error);}finally{submit.disabled=false;}};
    document.getElementById('sign-out').onclick=async()=>{try{await api.session.signOut();legacy='';actor=null;paintIdentity();root.dispatchEvent(new Event('ledger-signout'));}catch(error){api.failure(error);}};
    api.ready.then(paintIdentity).catch(()=>{});
  };
})(typeof window==='undefined'?globalThis:window);
