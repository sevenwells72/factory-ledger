/* The draft is FL's response; the browser never calculates permission or stock rules. */
(function(root){
  const ui=root.FLLedger;
  ui.ticketCard=function(host,path,payload,{onRecorded=()=>{},onPrepared=()=>{},onFailure=()=>{},onBusy=()=>{},onEdit=null,photoKind='correction_photo',receiptPrefix=null,allowPhoto=true}={}){
    let prepared=null,busy=false,finished=false,attachment=null,confirmations=[],lastLots=[];
    const card=ui.el('section',undefined,host);card.className='panel';
    function editButton(){if(onEdit){const edit=ui.el('button',ui.t('Edit entry','Editar entrada'),card);edit.type='button';edit.onclick=()=>{if(!busy)onEdit();};}}
    function failure(error){ui.failure(error);onFailure(error);}
    function invalidate(){finished=true;card.replaceChildren();ui.el('p',ui.t('Draft cleared. Prepare again after signing in.','Borrador borrado. Prepara de nuevo tras iniciar sesión.'),card);}
    window.addEventListener('ledger-signout',invalidate,{once:true});
    async function prepare(){if(busy||finished)return;busy=true;card.replaceChildren();ui.el('p',ui.t('Preparing…','Preparando…'),card);
      try{prepared=await ui.request(path,{...payload,...(confirmations.length?{lot_confirmations:confirmations}:{})});if(finished)return;if(!prepared.ticket||!prepared.payload_hash||!prepared.draft)throw Error('FL did not return a draft / FL no devolvió un borrador');document.getElementById('notice').hidden=true;render();onPrepared(prepared);}
      catch(error){failure(error);card.replaceChildren();const retry=ui.el('button',ui.t('Try prepare again','Preparar de nuevo'),card);retry.type='button';retry.onclick=prepare;editButton();}finally{busy=false;}}
    function render(){card.replaceChildren();editButton();ui.el('h2',ui.t('Draft — review before recording','Borrador — revisa antes de registrar'),card);ui.el('p',ui.t('Recording as ','Registrando como ')+(prepared.actor?.name||prepared.draft.actor?.name||'—'),card);ui.el('p',ui.t('Expires: ','Vence: ')+ui.plantTime(prepared.expires_at,ui.lang),card);
      const view=ui.el('div',undefined,card);view.className='draft-data';view.innerHTML=path==='/make/prepare'?ui.makeDraftHtml(prepared.draft,root.matchMedia?.('(max-width: 520px)').matches!==false):ui.draftHtml(prepared.draft);
      const ack=[];let evidenceDirty=false;
      (prepared.warnings||[]).forEach(w=>{const label=ui.el('label',undefined,card);label.className='check notice';const input=ui.el('input',undefined,label);input.type='checkbox';input.dataset.warning=w.code;ui.el('span',ui.t(ui.text(w,'en'),ui.text(w,'es')),label);ack.push([input,w.code]);});
      if(prepared.can_commit!==true){const state=ui.el('p',ui.t('NOT recorded — resolve the blockers below.','NO registrado — resuelve los bloqueos abajo.'),card);state.className='bad';}
      (prepared.blockers||[]).forEach(b=>{const p=ui.el('p',ui.t(ui.text(b,'en'),ui.text(b,'es')),card);p.className='bad';});
      if(prepared.draft.input_plan?.length)lastLots=prepared.draft.input_plan;
      const lots=new Map(lastLots.map(l=>[l.lot_id,l]));const inputs=[];
      lots.forEach(lot=>{const field=ui.el('fieldset',undefined,card);ui.el('legend',ui.t('Confirm physical lot: ','Confirma lote físico: ')+lot.lot_code,field);const label=ui.el('label',ui.t('Confirmation method','Método de confirmación'),field),method=ui.el('select',undefined,label);
        [['last4',ui.t('Last 4 characters (include hyphen)','Últimos 4 caracteres (incluye guion)')],['full_code',ui.t('Full code','Código completo')],['pallet',ui.t('Pallet — full code, moved to production','Palé — código completo, movido a producción')]].forEach(([v,t])=>{const o=ui.el('option',t,method);o.value=v;});
        const label2=ui.el('label',ui.t('Type the code from the lot label','Escribe el código de la etiqueta'),field),value=ui.el('input',undefined,label2);value.autocomplete='off';const prev=confirmations.find(c=>c.lot_id===lot.lot_id);if(prev){method.value=prev.method;value.value=prev.value;}
        inputs.push(()=>({lot_id:lot.lot_id,method:method.value,value:value.value}));
        for(const control of [method,value])control.oninput=()=>{evidenceDirty=true;record.disabled=true;};
      });
      if(lots.size){const check=ui.el('button',ui.t('Check confirmations → new draft','Verificar confirmaciones → nuevo borrador'),card);check.type='button';check.onclick=()=>{confirmations=inputs.map(fn=>fn());prepare();};}
      const uploadLabel=ui.el('label',ui.t('Attach photo','Adjuntar foto'),card),photo=ui.el('input',undefined,uploadLabel);photo.type='file';photo.accept='image/jpeg,image/png,image/webp';photo.setAttribute('capture','environment');
      const uploadStatus=ui.el('p',attachment?ui.t('Photo attached','Foto adjunta'):'',card);
      uploadLabel.hidden=!allowPhoto;uploadStatus.hidden=!allowPhoto;
      photo.onchange=async()=>{if(!photo.files[0])return;record.disabled=true;try{attachment=await ui.upload(photo.files[0],photoKind);uploadStatus.textContent=ui.t('Photo attached','Foto adjunta');}catch(error){failure(error);}finally{record.disabled=busy||finished||evidenceDirty||prepared.can_commit!==true;}};
      const actions=ui.el('div',undefined,card);actions.className='record-bar';const record=ui.el('button',ui.t('Record','Registrar'),actions);record.type='button';record.className='primary';record.disabled=prepared.can_commit!==true;
      record.onclick=async()=>{if(busy||finished)return;busy=true;onBusy(true);record.disabled=true;const body={payload_hash:prepared.payload_hash,acknowledged_warnings:ack.filter(([input])=>input.checked).map(([,code])=>code)};
        if(attachment){if(photoKind==='bol')body.attachment_ids=[attachment.attachment_id];else body.attachment_id=attachment.attachment_id;}
        try{const result=await ui.request('/tickets/'+encodeURIComponent(prepared.ticket)+'/commit',body);if(finished)return;
          if(!ui.recorded(result)||(receiptPrefix&&!new RegExp('^'+receiptPrefix+'-\\d{6}-\\d{3,}$').test(result.receipt_number))){ui.notice(ui.t('NOT recorded — ','NO registrado — ')+ui.errorText(result,ui.lang));if(result.held||result.status==='awaiting_approval'){finished=true;const a=ui.el('a',ui.t('Open owner approval queue','Ver cola de aprobación'),card);a.href=FLReceipts.page('exceptions.html');}return;}
          finished=true;document.getElementById('notice').hidden=true;card.replaceChildren();const p=ui.el('p',ui.t('Recorded — receipt ','Registrado — recibo '),card);p.className='good';const a=ui.el('a',result.receipt_number,p);a.href=FLReceipts.link(result.receipt_number);const substitutions=ui.el('div',undefined,card);substitutions.innerHTML=ui.autoSubstitutionsHtml(result.auto_substitutions);onRecorded(result);
        }catch(error){failure(error);}finally{busy=false;onBusy(false);record.disabled=finished||evidenceDirty||prepared.can_commit!==true;}};
    }
    prepare();return {invalidate};
  };
})(window);
