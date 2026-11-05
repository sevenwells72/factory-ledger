(() => {
  const ui=FLLedger,G=FLExceptionsGuide;ui.bind();let rows=[],closedUnverified=[],selected,helpOpen=false;
  const content=document.getElementById('content'),resolution=document.getElementById('resolution');
  // Render only the actor-specific allowed_actions returned by FL.
  const labels={counted:['Counted adjustment','Ajuste contado'],missing_movement:['Link missing movement receipt','Vincular recibo de movimiento faltante'],voided:['Posting was voided','Registro anulado'],unverified:['Resolve without count','Resolver sin conteo'],identified:['Identify supplier lot','Identificar lote de proveedor'],written_off:['Owner write-off','Baja por propietario'],approved:['Owner approve','Aprobar como propietario'],declined:['Owner reject','Rechazar como propietario'],photo_attached:['Attach BOL photo','Adjuntar foto BOL'],waived:['Owner waive','Eximir como propietario'],acknowledged:['Acknowledge','Reconocer'],resolved:['Resolve','Resolver']};
  const both=(fn,...args)=>ui.t(fn(...args,'en'),fn(...args,'es'));
  // Hover shows the title; on touch screens a "?" next to the control shows the same text.
  function tipped(host,tag,label,name){const wrap=ui.el('span',undefined,host);wrap.className='tip-wrap';const control=ui.el(tag,label,wrap);if(tag==='button')control.type='button';
    const text=both(G.tip,name);control.title=String(text);const toggle=ui.el('button','?',wrap);toggle.type='button';toggle.className='tip-toggle';toggle.setAttribute('aria-label',String(ui.t('What does this do?','¿Qué hace esto?')));toggle.setAttribute('aria-expanded','false');
    const hint=ui.el('span',text,wrap);hint.className='tip-text';hint.hidden=true;toggle.onclick=()=>{hint.hidden=!hint.hidden;toggle.setAttribute('aria-expanded',String(!hint.hidden));};return control;}
  function unverifiedBadge(host){const b=ui.el('span',ui.t('Unverified','Sin verificar'),host);b.className='badge-unverified';b.title=String(ui.t('Closed without a count. Stock was not checked.','Cerrada sin conteo. No se revisó el inventario.'));return b;}
  function renderHelp(host){const panel=ui.el('section',undefined,host);panel.className='panel guide-panel';panel.id='exceptions-help';ui.el('h2',ui.t('How this works','Cómo funciona'),panel);
    const en=G.help('en'),es=G.help('es');en.forEach((s,i)=>{ui.el('h3',ui.t(s.title,es[i].title),panel);s.lines.forEach((line,j)=>ui.el('p',ui.t(line,es[i].lines[j]),panel));});}
  function render(){content.replaceChildren();const bar=ui.el('div',undefined,content);bar.className='exceptions-toolbar';
    const refresh=tipped(bar,'button',ui.t('Refresh exceptions','Actualizar excepciones'),'refresh');refresh.onclick=load;
    const howto=tipped(bar,'button',ui.t('How this works','Cómo funciona'),'help');howto.setAttribute('aria-controls','exceptions-help');howto.setAttribute('aria-expanded',String(helpOpen));howto.onclick=()=>{helpOpen=!helpOpen;render();};
    if(helpOpen)renderHelp(content);
    if(!rows.length)ui.el('p',ui.t('No open exceptions.','No hay excepciones abiertas.'),content);
    // An overdue opening-stock item needs attention now; never hide it in the collapsed group.
    const ordered=ui.exceptionQueue(rows),recent=ordered.filter(item=>item.overdue||!ui.openingStock(item)),opening=ordered.filter(item=>!item.overdue&&ui.openingStock(item));
    function renderRows(items,host){for(const item of items){const row=ui.el('article',undefined,host);row.dataset.exceptionId=item.id;row.className='exception-row'+(item.overdue?' overdue':'');
      const desc=ui.el('div',undefined,row),number=item.receipt_number||item.ticket_receipt_number;
      ui.el('h3',ui.exceptionLabel(item.kind,ui.lang),desc);
      if(number){const a=ui.el('a',number,desc);a.href=FLReceipts.link(number);a.className='record-id';}
      ui.el('p',[item.product?.name,item.lot?.lot_code].filter(Boolean).join(' · '),desc);
      const note=ui.lang==='es'?item.detail?.message_es:item.detail?.message;if(note)ui.el('p',note,desc);
      const explanation=ui.el('p',both(G.explain,item),desc);explanation.className='neutral exception-explain';explanation.hidden=true;explanation.id='explain-'+item.id;
      ui.el('div',ui.t('Owner: ','Responsable: ')+(item.owner?.name||ui.t('Unassigned','Sin asignar')),row);
      ui.el('div',(item.overdue?ui.t('Overdue · ','Vencido · '):'')+(item.due_at?new Date(item.due_at).toLocaleString(ui.lang==='es'?'es-US':'en-US',{timeZone:'America/New_York'}):ui.t('No deadline','Sin fecha límite')),row);
      const actions=ui.el('div',undefined,row);actions.className='row-actions';
      const explain=tipped(actions,'button',ui.t('Explain','Explicar'),'explain');explain.dataset.explain=item.id;explain.setAttribute('aria-controls',explanation.id);explain.setAttribute('aria-expanded','false');
      explain.onclick=()=>{explanation.hidden=!explanation.hidden;explain.setAttribute('aria-expanded',String(!explanation.hidden));};
      if(ui.resolutionChoices(item).length){const button=tipped(actions,'button',ui.t('Resolve','Resolver'),'resolve');button.dataset.resolve=item.id;button.onclick=()=>open(item.id);}
    }}
    if(recent.length){const section=ui.el('section',undefined,content);section.className='panel';ui.el('h2',ui.t('Needs follow-up','Necesita seguimiento')+' · '+recent.length,section);renderRows(recent,section);}
    if(opening.length){const group=ui.el('details',undefined,content);group.className='panel opening-stock';ui.el('summary',ui.t('Opening-stock items','Elementos del inventario inicial')+' · '+opening.length,group);renderRows(opening,group);}
    if(closedUnverified.length){const group=ui.el('details',undefined,content);group.className='panel closed-unverified';ui.el('summary',ui.t('Closed without a count','Cerradas sin conteo')+' · '+closedUnverified.length,group);
      ui.el('p',ui.t('These shortages were closed on a note only. Stock was not checked. The owner reviews them.','Estos faltantes se cerraron solo con una nota. No se revisó el inventario. El dueño los revisa.'),group).className='muted';
      for(const item of closedUnverified){const row=ui.el('article',undefined,group);row.className='exception-row';row.dataset.exceptionId=item.id;const desc=ui.el('div',undefined,row);
        const h=ui.el('h3',ui.exceptionLabel(item.kind,ui.lang)+' #'+item.id+' ',desc);unverifiedBadge(h);
        ui.el('p',[item.product?.name,item.lot?.lot_code].filter(Boolean).join(' · '),desc);if(item.resolution_note)ui.el('p',item.resolution_note,desc);
        ui.el('div',ui.t('Closed by: ','Cerrada por: ')+(item.resolved_by?.name||'—'),row);
        ui.el('div',item.resolved_at?new Date(item.resolved_at).toLocaleString(ui.lang==='es'?'es-US':'en-US',{timeZone:'America/New_York'}):'',row);
        const cell=ui.el('div',undefined,row);if(item.resolution_receipt_number){const a=ui.el('a',item.resolution_receipt_number,cell);a.href=FLReceipts.link(item.resolution_receipt_number);}
      }}
  }
  const byReceipt=list=>{const number=new URLSearchParams(location.search).get('receipt');return number?list.filter(r=>(r.receipt_number||r.ticket_receipt_number)===number):list;};
  async function load({background=false}={}){try{const result=await ui.request('/exceptions?status=open&limit=1000',undefined,{background});rows=byReceipt(result.exceptions||[]);
    // Read-only list of shortages closed without a count; a failure here never hides the open queue.
    try{const closed=await ui.request('/exceptions?status=resolved&resolution_kind=unverified&limit=50',undefined,{background:true});closedUnverified=byReceipt((closed.exceptions||[]).filter(r=>r.resolution_kind==='unverified'));}catch{closedUnverified=[];}
    render();if(result.count>=1000)ui.notice(ui.t('First 1,000 items shown.','Se muestran los primeros 1.000 elementos.'),'notice');}catch(error){ui.failure(error);}}
  async function open(id){try{selected=await ui.request('/exceptions/'+id);renderForm();resolution.scrollIntoView({block:'start'});}catch(error){ui.failure(error);}}
  function renderForm(){resolution.replaceChildren();const item=selected;if(!item)return;
    const panel=ui.el('section',undefined,resolution);panel.className='panel';ui.el('h2',ui.t('Resolve ','Resolver ')+ui.exceptionLabel(item.kind,ui.lang)+' #'+item.id,panel);
    const why=ui.el('p',both(G.explain,item),panel);why.className='neutral exception-explain';
    const detail=ui.el('div',undefined,panel);detail.innerHTML=ui.draftHtml(item.detail);
    const form=ui.el('form',undefined,panel);form.className='fields';
    const choices=ui.resolutionChoices(item);
    if(!choices.length){const p=ui.el('p',ui.t('You cannot resolve this one. The owner decides it.','No puedes resolver esta. La decide el propietario.'),panel);p.className='neutral';form.remove();return;}
    const label=ui.el('label',ui.t('Resolution','Resolución'),form),kind=ui.el('select',undefined,label);label.title=G.field('resolution',ui.lang);choices.forEach(k=>{const o=ui.el('option',labels[k]?ui.t(...labels[k]):k,kind);o.value=k;});
    ui.el('small',both(G.field,'resolution'),label).className='field-help';
    const guide=ui.el('div',undefined,form);guide.className='wide option-guide';guide.setAttribute('aria-live','polite');
    const fields=ui.el('div',undefined,form);fields.className='wide fields';
    function field(name,en,es,type='text',help=name){const l=ui.el('label',ui.t(en,es),fields),i=ui.el(type==='textarea'?'textarea':'input',undefined,l);i.name=name;if(type!=='textarea')i.type=type;
      const text=both(G.field,help);l.title=String(text);const small=ui.el('small',text,l);small.className='field-help';return i;}
    let note,count,receipt,photo,code,reason;
    function renderGuide(){guide.replaceChildren();const help=['en','es'].map(lang=>G.option(kind.value,lang));if(!help[0])return;
      for(const [key,en,es] of [['use','Use when: ','Úsalo cuando: '],['example','Example: ','Ejemplo: '],['effect','This will: ','Esto hará: ']]){const p=ui.el('p',undefined,guide);p.dataset.guide=key;ui.el('strong',ui.t(en,es),p);ui.el('span',ui.t(help[0][key],help[1][key]),p);}}
    function update(){fields.replaceChildren();renderGuide();const required=ui.noteRequired(kind.value);note=required?field('note','Note (required)','Nota (obligatoria)','textarea',kind.value==='unverified'?'note_unverified':'note'):field('note','Note (optional)','Nota (opcional)','textarea');note.required=required;count=receipt=photo=code=reason=null;
      if(kind.value==='counted'){count=field('counted_lb','Physically counted lb','Libras contadas físicamente','number');count.step='any';count.min='0';count.required=true;}
      if(kind.value==='missing_movement'){receipt=field('receipt_number','Evidence Receipt #','Recibo de evidencia #');receipt.required=true;}
      if(kind.value==='identified'){code=field('supplier_lot_code','Supplier lot code','Código de lote del proveedor');code.required=true;const l=ui.el('label',ui.t('Reason','Motivo'),fields);reason=ui.el('select',undefined,l);[['data_entry_error','Data-entry error / Error de captura'],['wrong_lot','Wrong lot used / Lote equivocado']].forEach(([v,t])=>{const o=ui.el('option',t,reason);o.value=v;});const text=both(G.field,'reason');l.title=String(text);ui.el('small',text,l).className='field-help';}
      if(['counted','photo_attached'].includes(kind.value)){photo=field('photo','Photo','Foto','file',kind.value==='photo_attached'?'photo_bol':'photo');photo.accept='image/jpeg,image/png,image/webp';if(kind.value==='photo_attached')photo.required=true;}
      preview.replaceChildren();
    }
    const button=tipped(form,'button',ui.t('Review resolution','Revisar resolución'),'review');button.type='submit';button.className='primary';const preview=ui.el('div',undefined,panel);kind.onchange=update;update();form.oninput=()=>preview.replaceChildren();
    const confirm=values=>{const p=ui.el('p',undefined,preview);p.className='notice confirm-line';p.dataset.confirm=kind.value;ui.el('strong',both(G.confirmLine,item,kind.value,values),p);};
    form.onsubmit=async event=>{event.preventDefault();button.disabled=true;preview.replaceChildren();try{
      if(kind.value==='identified'){confirm({supplier_lot_code:code.value.trim()});const host=ui.el('div',undefined,preview);ui.ticketCard(host,'/lots/'+item.lot.id+'/supplier-lot/prepare',{supplier_lot_code:code.value,reason_code:reason.value,...(note.value.trim()?{note:note.value}:{}),client_source:'dashboard'},{onRecorded:result=>{form.remove();const ids=result.resolved_exception_ids||[];const p=ui.el('p',ids.length?ui.t('Resolved exception ','Excepción resuelta ')+ids.map(id=>'#'+id).join(', '):ui.t('No open exception was closed by this correction.','Esta corrección no cerró ninguna excepción abierta.'),preview);p.className='neutral';p.dataset.resolved=ids.join(',');load();}});return;}
      const extra={};if(count)extra.counted_lb=Number(count.value);if(receipt)extra.receipt_number=receipt.value;
      if(photo?.files[0])extra.attachment_id=(await ui.upload(photo.files[0],kind.value==='photo_attached'?'bol':'correction_photo')).attachment_id;
      const {action,body}=ui.decisionRequest(kind.value,note.value,extra);
      // Use FL's compatible explicit decision routes; its response identifies the XR action receipt.
      // Show the latest FL snapshot and exact submitted evidence before the decision.
      const current=await ui.request('/exceptions/'+item.id);if(!ui.resolutionChoices(current).includes(kind.value)){selected=current;renderForm();await load();return;}
      confirm({counted_lb:extra.counted_lb,receipt_number:extra.receipt_number});
      ui.el('h3',ui.t('Review — no decision recorded yet','Revisa — aún no se ha registrado la decisión'),preview);const data=ui.el('div',undefined,preview);data.innerHTML=ui.draftHtml(current.detail)+ui.draftHtml(body.resolution_kind&&labels[body.resolution_kind]?{...body,resolution_kind:String(ui.t(...labels[body.resolution_kind]))}:body);
      const record=tipped(preview,'button',ui.t('Confirm resolution','Confirmar resolución'),'confirm');record.className='primary';const chosen=kind.value;record.onclick=async()=>{record.disabled=true;
        try{const result=await ui.request('/exceptions/'+item.id+'/'+action,body),outcome=ui.decisionOutcome(action,result);
          if(outcome.tone==='good'){showOutcome(outcome,result.resolution_kind||chosen);selected=null;resolution.replaceChildren();}
          else ui.failure(Error(ui.errorText(result,ui.lang)));await load();preview.replaceChildren();
        }catch(error){ui.failure(error);record.disabled=false;}};
    }catch(error){ui.failure(error);}finally{button.disabled=false;}};
  }
  function showOutcome(outcome,kind){
    ui.notice(ui.t('Recorded decision — receipt ','Decisión registrada — recibo '),'good');const box=document.getElementById('notice');
    const link=ui.el('a',outcome.receipt,box);link.href=FLReceipts.link(outcome.receipt);
    if(outcome.postingReceipt){const p=ui.el('p',ui.t('Stock posting receipt: ','Recibo del movimiento de inventario: '),box);const a=ui.el('a',outcome.postingReceipt,p);a.href=FLReceipts.link(outcome.postingReceipt);}
    if(outcome.status==='still_open')ui.el('p',ui.t('Evidence recorded; this exception is still open.','Evidencia registrada; esta excepción sigue abierta.'),box);
    if(outcome.status==='declined')ui.el('p',ui.t('Rejected; no stock was posted.','Rechazada; no se registró movimiento de inventario.'),box);
    if(kind==='unverified'&&outcome.status==='resolved'){const p=ui.el('p',undefined,box);unverifiedBadge(p);p.append(' ');ui.el('span',ui.t('Closed without a count. No stock was changed.','Cerrada sin conteo. No se cambió el inventario.'),p);}
  }
  window.addEventListener('ledger-auth',load);window.addEventListener('ledger-language',()=>{render();renderForm();});window.addEventListener('ledger-signout',()=>{rows=[];closedUnverified=[];selected=null;content.replaceChildren();resolution.replaceChildren();});ui.ready.then(()=>{if(ui.session.actor)load({background:true});}).catch(()=>{});
})();
