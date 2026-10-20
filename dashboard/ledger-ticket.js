/* The draft is FL's response; the browser never calculates permission or stock rules. */
(function(root){
  const ui=root.FLLedger;
  ui.ticketCard=function(host,path,payload,{onRecorded=()=>{},onPrepared=()=>{},onFailure=()=>{},onBusy=()=>{},onEdit=null,onCancel=null,onAttachment=()=>{},initialAttachment=null,photoKind='correction_photo',receiptPrefix=null,allowPhoto=true}={}){
    if(path==='/receive/prepare')photoKind=['bol','lot_tag'].includes(initialAttachment?.kind)?initialAttachment.kind:['bol','lot_tag'].includes(photoKind)?photoKind:'lot_tag';
    let prepared=null,busy=false,finished=false,attachment=initialAttachment,uploading=false,confirmations=[],lastLots=[];
    const card=ui.el('section',undefined,host);card.className='panel';
    function editButton(){const cancel=ui.el('button',ui.t('Cancel','Cancelar'),card);cancel.type='button';cancel.className='secondary';cancel.onclick=()=>{if(busy||uploading||finished)return;finished=true;card.remove();document.getElementById('notice').hidden=true;if(onCancel)onCancel();else if(onEdit)onEdit();};if(onEdit){const edit=ui.el('button',ui.t('Edit entry','Editar entrada'),card);edit.type='button';edit.onclick=()=>{if(!busy&&!uploading&&!finished)onEdit();};}}
    function failure(error){ui.failure(error);onFailure(error);}
    function invalidate(){finished=true;card.replaceChildren();ui.el('p',ui.t('Draft cleared. Prepare again after signing in.','Borrador borrado. Prepara de nuevo tras iniciar sesión.'),card);}
    window.addEventListener('ledger-signout',invalidate,{once:true});
    async function prepare(){if(busy||finished)return;busy=true;card.replaceChildren();ui.el('p',ui.t('Preparing…','Preparando…'),card);
      try{prepared=await ui.request(path,{...payload,...(confirmations.length?{lot_confirmations:confirmations}:{})});if(finished)return;if(!prepared.ticket||!prepared.payload_hash||!prepared.draft)throw Error('FL did not return a draft / FL no devolvió un borrador');document.getElementById('notice').hidden=true;render();onPrepared(prepared);}
      catch(error){if(finished)return;failure(error);card.replaceChildren();const retry=ui.el('button',ui.t('Try prepare again','Preparar de nuevo'),card);retry.type='button';retry.onclick=prepare;editButton();}finally{busy=false;}}
    function render(){card.replaceChildren();editButton();ui.el('h2',ui.t('Draft — review before recording','Borrador — revisa antes de registrar'),card);ui.el('p',ui.t('Recording as ','Registrando como ')+(prepared.actor?.name||prepared.draft.actor?.name||'—'),card);ui.el('p',ui.t('Expires: ','Vence: ')+ui.plantTime(prepared.expires_at,ui.lang),card);
      const view=ui.el('div',undefined,card);view.className='draft-data';view.innerHTML=path==='/make/prepare'?ui.makeDraftHtml(prepared.draft,root.matchMedia?.('(max-width: 520px)').matches!==false):ui.draftHtml(prepared.draft);
      if(path==='/make/prepare')for(const shortage of prepared.draft.shortages||[]){
        const name=shortage.product_name||ui.t('this ingredient','este ingrediente'),amount=ui.formatNumber(shortage.on_hand_lb??shortage.available_lb??0);
        const warning=ui.el('p',ui.t('Only '+amount+' lb of '+name+' on hand — this will be recorded as a shortage.','Solo hay '+amount+' lb de '+name+' — se registrará como un faltante.'),card);warning.className='notice';warning.dataset.shortageWarning='';
      }
      const ack=[];let evidenceDirty=false;
      (prepared.warnings||[]).forEach(w=>{const label=ui.el('label',undefined,card);label.className='check notice';const input=ui.el('input',undefined,label);input.type='checkbox';input.dataset.warning=w.code;ui.el('span',ui.t(ui.text(w,'en'),ui.text(w,'es')),label);ack.push([input,w.code]);});
      if(prepared.can_commit!==true){const state=ui.el('p',ui.t('NOT recorded — resolve the blockers below.','NO registrado — resuelve los bloqueos abajo.'),card);state.className='bad';}
      (prepared.blockers||[]).forEach(b=>{const p=ui.el('p',ui.t(ui.text(b,'en'),ui.text(b,'es')),card);p.className='bad';});
      if(prepared.draft.input_plan?.length)lastLots=prepared.draft.input_plan;
      const lots=new Map(lastLots.map(l=>[l.lot_id,l]));const inputs=[];
      const ingredientGroups=prepared.draft.lot_confirmation_group?.ingredients||[];
      lots.forEach(lot=>{const field=ui.el('fieldset',undefined,card);field.className='lot-confirmation-row';field.dataset.productId=lot.product_id;
        ui.el('legend',ui.t('Confirm physical lot: ','Confirma lote físico: ')+(lot.product_name?lot.product_name+' · ':'')+lot.lot_code,field);
        const controls=ui.el('div',undefined,field);controls.className='lot-confirmation-controls';
        const label=ui.el('label',ui.t('Confirmation method','Método de confirmación'),controls),method=ui.el('select',undefined,label);
        [['last4',ui.t('Last 4 characters (include hyphen)','Últimos 4 caracteres (incluye guion)')],['full_code',ui.t('Full code','Código completo')],['pallet',ui.t('Pallet — full code, moved to production','Palé — código completo, movido a producción')]].forEach(([v,t])=>{const o=ui.el('option',t,method);o.value=v;});
        const label2=ui.el('label',ui.t('Type the code from the lot label','Escribe el código de la etiqueta'),controls),value=ui.el('input',undefined,label2);value.autocomplete='off';const prev=confirmations.find(c=>c.lot_id===lot.lot_id);if(prev){method.value=prev.method;value.value=prev.value;}
        const choices=ui.el('div',undefined,field);choices.className='lot-confirmation-choices';
        const input={lot,method,value,choices,read:()=>({lot_id:lot.lot_id,method:method.value,value:value.value})};inputs.push(input);
        for(const control of [method,value])control.oninput=()=>{choices.replaceChildren();evidenceDirty=true;record.disabled=true;};
      });
      if(lots.size){const check=ui.el('button',ui.t('Check confirmations → new draft','Verificar confirmaciones → nuevo borrador'),card);check.type='button';check.onclick=async()=>{
        if(busy||finished||uploading)return;let ambiguous=false;
        for(const input of inputs){
          if(path!=='/make/prepare'||input.method.value!=='last4'||input.value.value.trim().length!==4)continue;
          const group=ingredientGroups.find(g=>g.product_id===input.lot.product_id);
          const matches=(group?.candidates||[]).filter(l=>l.lot_code.toUpperCase().endsWith(input.value.value.trim().toUpperCase()));
          if(matches.length<2)continue;ambiguous=true;input.choices.replaceChildren();
          ui.el('p',ui.t('More than one lot matches. Choose the full code on the bag:','Coincide más de un lote. Elige el código completo de la bolsa:'),input.choices);
          for(const candidate of matches){const choose=ui.el('button',candidate.lot_code,input.choices);choose.type='button';choose.dataset.confirmLot=candidate.lot_id;
            choose.onclick=()=>{if(busy||finished||uploading)return;
              if(candidate.lot_id===input.lot.lot_id){input.method.value='full_code';input.value.value=candidate.lot_code;input.choices.replaceChildren();evidenceDirty=true;record.disabled=true;return;}
              // Selecting a different physical lot changes the reviewed plan; FL validates it again.
              const substitutes=(payload.substitutions||[]).filter(s=>s.substitute_product_id===input.lot.product_id);
              substitutes.forEach(substitute=>substitute.lot_id=candidate.lot_id);
              const override=(payload.ingredient_lots||[]).find(l=>l.ingredient_product_id===input.lot.product_id);
              if(override)override.lot_id=candidate.lot_id;
              else if(!substitutes.length)payload.ingredient_lots=[...(payload.ingredient_lots||[]),{ingredient_product_id:input.lot.product_id,lot_id:candidate.lot_id}];
              confirmations=inputs.filter(i=>i.lot.product_id!==input.lot.product_id).map(i=>i.read()).filter(c=>c.value.trim());
              confirmations.push({lot_id:candidate.lot_id,method:'full_code',value:candidate.lot_code});prepare();
            };
          }
        }
        if(ambiguous)return;confirmations=inputs.map(input=>input.read());prepare();
      };}
      let photoType=null;if(path==='/receive/prepare'){const kindLabel=ui.el('label',ui.t('Photo type','Tipo de foto'),card);photoType=ui.el('select',undefined,kindLabel);photoType.name='draft_photo_kind';for(const [value,en,es] of [['lot_tag','Label / lot tag','Etiqueta / etiqueta de lote'],['bol','BOL','BOL']]){const option=ui.el('option',ui.t(en,es),photoType);option.value=value;}photoType.value=photoKind;photoType.onchange=()=>{photoKind=photoType.value;};kindLabel.hidden=!allowPhoto;}
      const uploadLabel=ui.el('label',ui.t('Attach photo','Adjuntar foto'),card),photo=ui.el('input',undefined,uploadLabel);photo.type='file';photo.accept='image/jpeg,image/png,image/webp';photo.setAttribute('capture','environment');
      const uploadStatus=ui.el('p',attachment?ui.t('Photo attached','Foto adjunta'):'',card);
      uploadLabel.hidden=!allowPhoto;uploadStatus.hidden=!allowPhoto;
      photo.onchange=async()=>{if(!photo.files[0]||busy||finished||uploading)return;uploading=true;if(photoType)photoType.disabled=true;onBusy(true);record.disabled=true;try{const uploaded=await ui.upload(photo.files[0],photoKind);if(finished)return;attachment=uploaded;onAttachment(attachment);uploadStatus.textContent=ui.t('Photo attached','Foto adjunta');}catch(error){if(!finished)failure(error);}finally{uploading=false;if(photoType)photoType.disabled=false;if(!finished)onBusy(false);record.disabled=busy||finished||evidenceDirty||prepared.can_commit!==true;}};
      const actions=ui.el('div',undefined,card);actions.className='record-bar';const record=ui.el('button',ui.t('Record','Registrar'),actions);record.type='button';record.className='primary';record.disabled=prepared.can_commit!==true;
      record.onclick=async()=>{if(busy||finished||uploading)return;busy=true;onBusy(true);record.disabled=true;const body={payload_hash:prepared.payload_hash,acknowledged_warnings:ack.filter(([input])=>input.checked).map(([,code])=>code)};
        if(attachment){if(path==='/receive/prepare'||photoKind==='bol')body.attachment_ids=[attachment.attachment_id];else body.attachment_id=attachment.attachment_id;}
        try{const result=await ui.request('/tickets/'+encodeURIComponent(prepared.ticket)+'/commit',body);if(finished)return;
          if(!ui.recorded(result)||(receiptPrefix&&!new RegExp('^'+receiptPrefix+'-\\d{6}-\\d{3,}$').test(result.receipt_number))){ui.notice(ui.t('NOT recorded — ','NO registrado — ')+ui.errorText(result,ui.lang));if(result.held||result.status==='awaiting_approval'){finished=true;const a=ui.el('a',ui.t('Open owner approval queue','Ver cola de aprobación'),card);a.href=FLReceipts.page('exceptions.html');}return;}
          finished=true;document.getElementById('notice').hidden=true;card.replaceChildren();const p=ui.el('p',ui.t('Recorded — receipt ','Registrado — recibo '),card);p.className='good';const a=ui.el('a',result.receipt_number,p);a.href=FLReceipts.link(result.receipt_number);const substitutions=ui.el('div',undefined,card);substitutions.innerHTML=ui.autoSubstitutionsHtml(result.auto_substitutions);onRecorded(result);
        }catch(error){failure(error);}finally{busy=false;onBusy(false);record.disabled=finished||evidenceDirty||prepared.can_commit!==true;}};
    }
    prepare();return {invalidate};
  };
})(window);
