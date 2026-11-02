/* The draft is FL's response; the browser never calculates permission or stock rules. */
(function(root){
  const ui=root.FLLedger;
  ui.ticketCard=function(host,path,payload,{onRecorded=()=>{},onPrepared=()=>{},onFailure=()=>{},onBusy=()=>{},onEdit=null,onCancel=null,onAttachment=()=>{},initialAttachment=null,initialPhoto=null,photoKind='correction_photo',receiptPrefix=null,allowPhoto=true}={}){
    allowPhoto=allowPhoto&&!['/make/prepare','/pack/prepare'].includes(path);
    if(!allowPhoto)initialAttachment=null;
    if(path==='/receive/prepare')photoKind=['bol','lot_tag'].includes(initialAttachment?.kind)?initialAttachment.kind:['bol','lot_tag'].includes(photoKind)?photoKind:'lot_tag';
    let prepared=null,busy=false,finished=false,attachment=initialAttachment,photoFile=initialPhoto,uploading=false,confirmations=[],lastLots=[],lastFormula=null,anchor=null;
    const card=ui.el('section',undefined,host);card.className='panel';
    function editButton(){const cancel=ui.el('button',ui.t('Cancel','Cancelar'),card);cancel.type='button';cancel.className='secondary';cancel.onclick=()=>{if(busy||uploading||finished)return;finished=true;card.remove();document.getElementById('notice').hidden=true;if(onCancel)onCancel();else if(onEdit)onEdit();};if(onEdit){const edit=ui.el('button',ui.t('Edit entry','Editar entrada'),card);edit.type='button';edit.onclick=()=>{if(!busy&&!uploading&&!finished)onEdit();};}}
    function failure(error){ui.failure(error);onFailure(error);}
    function invalidate(){finished=true;card.replaceChildren();ui.el('p',ui.t('Draft cleared. Prepare again after signing in.','Borrador borrado. Prepara de nuevo tras iniciar sesión.'),card);}
    window.addEventListener('ledger-signout',invalidate,{once:true});
    async function prepare({refresh=false}={}){if(busy||finished)return;busy=true;
      // refresh: an edited Make draft stays on screen (dimmed) while FL re-checks it.
      if(refresh){card.setAttribute('aria-busy','true');card.classList?.add('updating');}else{card.replaceChildren();ui.el('p',ui.t('Preparing…','Preparando…'),card);}
      try{const next=await ui.request(path,{...payload,...(confirmations.length?{lot_confirmations:confirmations}:{})});if(finished)return;if(!next.ticket||!next.payload_hash||!next.draft)throw Error('FL did not return a draft / FL no devolvió un borrador');prepared=next;document.getElementById('notice').hidden=true;render();keepInView();onPrepared(prepared);}
      catch(error){if(finished)return;failure(error);if(refresh){anchor=null;return;}card.replaceChildren();const retry=ui.el('button',ui.t('Try prepare again','Preparar de nuevo'),card);retry.type='button';retry.onclick=()=>prepare();editButton();}finally{busy=false;if(refresh){card.setAttribute('aria-busy','false');card.classList?.remove('updating');}}}
    function keepInView(){if(!anchor)return;const box=[...(card.querySelectorAll?.('.formula-ingredient')||[])].find(b=>b.dataset.productId===anchor.id),top=box?.getBoundingClientRect?.().top;if(top!=null&&anchor.top!=null)root.scrollBy?.(0,top-anchor.top);anchor=null;}
    function render(){card.replaceChildren();editButton();ui.el('h2',ui.t('Draft — review before recording','Borrador — revisa antes de registrar'),card);ui.el('p',ui.t('Recording as '+(prepared.actor?.name||prepared.draft.actor?.name||'—'),'Registrando como '+(prepared.actor?.name||prepared.draft.actor?.name||'—')),card);ui.el('p',ui.t('Expires: '+ui.plantTime(prepared.expires_at,'en'),'Vence: '+ui.plantTime(prepared.expires_at,'es')),card);
      const view=ui.el('div',undefined,card);view.className='draft-data';view.innerHTML=path==='/make/prepare'?ui.makeDraftHtml(prepared.draft,root.matchMedia?.('(max-width: 520px)').matches!==false):ui.draftHtml(prepared.draft);
      // Make (phone): FL's own warnings already name each short lot; a warning that needs
      // no tick is shown as a note, never as a checkbox (commit checks only requires_ack).
      const make=path==='/make/prepare',serverShortage=(prepared.warnings||[]).some(w=>w.code==='WILL_CREATE_SHORTAGE');
      if(make&&!serverShortage)for(const shortage of prepared.draft.shortages||[]){
        const name=shortage.product_name||ui.t('this ingredient','este ingrediente'),amount=ui.formatNumber(shortage.on_hand_lb??shortage.available_lb??0);
        const warning=ui.el('p',ui.t('Only '+amount+' lb of '+name+' on hand — this will be recorded as a shortage.','Solo hay '+amount+' lb de '+name+' — se registrará como un faltante.'),card);warning.className='notice';warning.dataset.shortageWarning='';
      }
      const ack=[];let evidenceDirty=false;
      (prepared.warnings||[]).forEach(w=>{if(make&&w.requires_ack===false){const note=ui.el('p',ui.t(ui.text(w,'en'),ui.text(w,'es')),card);note.className='notice';note.dataset.warningInfo=w.code;if(w.code==='WILL_CREATE_SHORTAGE')note.dataset.shortageWarning='';return;}const label=ui.el('label',undefined,card);label.className='check notice';const input=ui.el('input',undefined,label);input.type='checkbox';input.dataset.warning=w.code;ui.el('span',ui.t(ui.text(w,'en'),ui.text(w,'es')),label);ack.push([input,w.code]);});
      if(prepared.can_commit!==true){const state=ui.el('p',ui.t('NOT recorded — resolve the blockers below.','NO registrado — resuelve los bloqueos abajo.'),card);state.className='bad';}
      // Each Make bag shows its own "Not confirmed" state, so unconfirmed bags are one line here.
      let blockers=prepared.blockers||[];const unconfirmed=blockers.filter(b=>b.code==='LOT_NOT_CONFIRMED');
      if(make&&prepared.draft.formula_entry&&unconfirmed.length>1)blockers=[...blockers.filter(b=>b.code!=='LOT_NOT_CONFIRMED'),{code:'LOT_NOT_CONFIRMED',message:unconfirmed.length+' bags still need the code from their label — type it on each bag marked "Not confirmed" below.',message_es:'A '+unconfirmed.length+' bolsas les falta el código de la etiqueta — escríbelo en cada bolsa marcada "Sin confirmar" abajo.'}];
      blockers.forEach(b=>{const p=ui.el('p',ui.t(ui.text(b,'en'),ui.text(b,'es')),card);p.className='bad';p.dataset.blocker=b.code||'';});
      if(prepared.draft.input_plan?.length)lastLots=prepared.draft.input_plan;
      // Formula-driven Make: ingredient cards replace the per-lot rows (make-formula.js).
      if(prepared.draft.formula_entry)lastFormula=prepared.draft.formula_entry;const formula=path==='/make/prepare'&&ui.formulaEntry?lastFormula:null;
      // The re-check button lives in the record bar (sticky on a phone) and shows once something changed.
      const actions=ui.el('div');actions.className=formula?'record-bar sticky':'record-bar';let updateHint=null;
      const formulaApi=formula?ui.formulaEntry(card,formula,{confirmations,checkHost:actions,dirty:()=>{evidenceDirty=true;record.disabled=true;syncUpdate();},update:(lines,next)=>{if(busy||finished||uploading)return;payload.ingredient_lines=lines;confirmations=next;const at=formulaApi.anchor();anchor=at?{id:at.dataset.productId,top:at.getBoundingClientRect?.().top}:null;prepare({refresh:true});}}):null;
      function syncUpdate(){if(!formulaApi)return;formulaApi.check.hidden=!evidenceDirty;formulaApi.check.className=evidenceDirty?'primary':'secondary';if(updateHint)updateHint.hidden=!evidenceDirty;}
      const lots=new Map(formula?[]:lastLots.map(l=>[l.lot_id,l]));const inputs=[];
      lots.forEach(lot=>{const field=ui.el('fieldset',undefined,card);field.className='lot-confirmation-row';field.dataset.productId=lot.product_id;
        ui.el('legend',ui.t('Confirm physical lot: '+(lot.product_name?lot.product_name+' · ':'')+lot.lot_code,'Confirma lote físico: '+(lot.product_name?lot.product_name+' · ':'')+lot.lot_code),field);
        const controls=ui.el('div',undefined,field);controls.className='lot-confirmation-controls';
        const label=ui.el('label',ui.t('Confirmation method','Método de confirmación'),controls),method=ui.el('select',undefined,label);
        [['last4',ui.t('Last 4 characters (include hyphen)','Últimos 4 caracteres (incluye guion)')],['full_code',ui.t('Full code','Código completo')],['pallet',ui.t('Pallet — full code, moved to production','Palé — código completo, movido a producción')]].forEach(([v,t])=>{const o=ui.el('option',t,method);o.value=v;});
        const label2=ui.el('label',ui.t('Type the code from the lot label','Escribe el código de la etiqueta'),controls),value=ui.el('input',undefined,label2);value.autocomplete='off';const prev=confirmations.find(c=>c.lot_id===lot.lot_id);if(prev){method.value=prev.method;value.value=prev.value;}
        const choices=ui.el('div',undefined,field);choices.className='lot-confirmation-choices';
        const input={lot,method,value,choices,read:()=>({lot_id:lot.lot_id,method:method.value,value:value.value})};inputs.push(input);
        for(const control of [method,value])control.oninput=()=>{choices.replaceChildren();evidenceDirty=true;record.disabled=true;};
      });
      if(lots.size){const check=ui.el('button',ui.t('Check confirmations → new draft','Verificar confirmaciones → nuevo borrador'),card);check.type='button';check.onclick=async()=>{
        // Make bags (and their ambiguous last-4 check) live in make-formula.js; this list only sends typed codes.
        if(busy||finished||uploading)return;
        // Every listed lot needs its typed code; a blank row would reach FL as "" (422).
        const blank=inputs.filter(input=>!input.value.value.trim());
        if(blank.length){for(const input of blank){input.choices.replaceChildren();const name=(input.lot.product_name?input.lot.product_name+' · ':'')+input.lot.lot_code;const p=ui.el('p',ui.t('Type the code from the lot label for '+name+'.','Escribe el código de la etiqueta para '+name+'.'),input.choices);p.className='bad';p.dataset.blankConfirmation='';}blank[0].value.focus?.();return;}
        confirmations=inputs.map(input=>input.read());prepare();
      };}
      if(allowPhoto){
      let photoType=null;if(path==='/receive/prepare'){const kindLabel=ui.el('label',ui.t('Photo type','Tipo de foto'),card);photoType=ui.el('select',undefined,kindLabel);photoType.name='draft_photo_kind';for(const [value,en,es] of [['lot_tag','Label / lot tag','Etiqueta / etiqueta de lote'],['bol','BOL','BOL']]){const option=ui.el('option',ui.t(en,es),photoType);option.value=value;}photoType.value=photoKind;}
      const uploadLabel=ui.el('label',ui.t('Attach photo','Adjuntar foto'),card),photo=ui.el('input',undefined,uploadLabel);photo.type='file';photo.accept='image/jpeg,image/png,image/webp';photo.setAttribute('capture','environment');
      const uploadStatus=ui.el('p',attachment?ui.t('Photo attached','Foto adjunta'):'',card);
      async function uploadPhoto(file,kind,{retype=false}={}){if(!file||busy||finished||uploading)return;const previousKind=photoKind;uploading=true;photo.disabled=true;if(photoType)photoType.disabled=true;onBusy(true);record.disabled=true;ui.setText(uploadStatus,ui.t('Uploading photo…','Subiendo foto…'));try{const uploaded=await ui.upload(file,kind);if(finished)return;attachment=uploaded;photoFile=file;photoKind=kind;onAttachment(attachment,photoFile);ui.setText(uploadStatus,ui.t('Photo attached','Foto adjunta'));}catch(error){if(!finished){if(retype){photoKind=previousKind;photoType.value=previousKind;ui.setText(uploadStatus,ui.t('Photo type was not changed. The attached photo keeps its previous type.','No se cambió el tipo de foto. La foto adjunta conserva su tipo anterior.'));}else ui.setText(uploadStatus,attachment?ui.t('Photo attached','Foto adjunta'):ui.t('Photo was not attached. Try again.','La foto no se adjuntó. Intenta de nuevo.'));failure(error);}}finally{uploading=false;photo.disabled=false;if(photoType)photoType.disabled=false;if(!finished)onBusy(false);record.disabled=busy||finished||evidenceDirty||prepared.can_commit!==true;}}
      if(photoType)photoType.onchange=()=>{if(busy||finished||uploading)return;if(attachment){if(photoFile)return uploadPhoto(photoFile,photoType.value,{retype:true});photoType.value=photoKind;ui.setText(uploadStatus,ui.t('Choose the photo again to change its type.','Elige la foto de nuevo para cambiar su tipo.'));return;}photoKind=photoType.value;};
      photo.onchange=()=>uploadPhoto(photo.files?.[0],photoType?.value||photoKind);
      }
      card.append(actions);if(formulaApi){updateHint=ui.el('p',ui.t('You changed bags, pounds or codes. Check them first, then Record.','Cambiaste bolsas, libras o códigos. Verifícalos primero y luego Registra.'));updateHint.className='muted';updateHint.dataset.updateHint='';actions.append(updateHint);}
      const record=ui.el('button',ui.t('Record','Registrar'),actions);record.type='button';record.className='primary';record.disabled=prepared.can_commit!==true;syncUpdate();
      record.onclick=async()=>{if(busy||finished||uploading)return;busy=true;onBusy(true);record.disabled=true;const body={payload_hash:prepared.payload_hash,acknowledged_warnings:ack.filter(([input])=>input.checked).map(([,code])=>code)};
        if(attachment){if(path==='/receive/prepare'||photoKind==='bol')body.attachment_ids=[attachment.attachment_id];else body.attachment_id=attachment.attachment_id;}
        try{const result=await ui.request('/tickets/'+encodeURIComponent(prepared.ticket)+'/commit',body);if(finished)return;
          if(!ui.recorded(result)||(receiptPrefix&&!new RegExp('^'+receiptPrefix+'-\\d{6}-\\d{3,}$').test(result.receipt_number))){ui.notice(ui.t('NOT recorded — '+ui.errorText(result,'en'),'NO registrado — '+ui.errorText(result,'es')));if(result.held||result.status==='awaiting_approval'){finished=true;const a=ui.el('a',ui.t('Open owner approval queue','Ver cola de aprobación'),card);a.href=FLReceipts.page('exceptions.html');}return;}
          finished=true;document.getElementById('notice').hidden=true;card.replaceChildren();const tag=ui.el('div',undefined,card);tag.innerHTML=ui.receiveLotHtml(result);const p=ui.el('p',ui.t('Recorded — receipt ','Registrado — recibo '),card);p.className='good';const a=ui.el('a',result.receipt_number,p);a.href=FLReceipts.link(result.receipt_number);const substitutions=ui.el('div',undefined,card);substitutions.innerHTML=ui.autoSubstitutionsHtml(result.auto_substitutions);onRecorded(result);
        }catch(error){failure(error);}finally{busy=false;onBusy(false);record.disabled=finished||evidenceDirty||prepared.can_commit!==true;}};
    }
    prepare();return {invalidate};
  };
})(window);
