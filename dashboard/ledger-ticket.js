/* The draft is FL's response; the browser never calculates permission or stock rules. */
(function(root){
  const ui=root.FLLedger;
  // Commit failures where FL wants a fresh draft; the Make screen re-checks with the entries kept.
  const STALE=['TICKET_EXPIRED','TICKET_STALE','TICKET_NOT_COMMITTABLE','INPUT_PLAN_CHANGED','LOT_IDENTITY_CHANGED','LOT_CODE_TAKEN','DRAFT_BLOCKED'];
  ui.ticketCard=function(host,path,payload,{onRecorded=()=>{},onPrepared=()=>{},onFailure=()=>{},onBusy=()=>{},onEdit=null,onCancel=null,onAttachment=()=>{},initialAttachment=null,initialPhoto=null,photoKind='correction_photo',receiptPrefix=null,allowPhoto=true,initialConfirmations=[]}={}){
    allowPhoto=allowPhoto&&!['/make/prepare','/pack/prepare'].includes(path);
    if(!allowPhoto)initialAttachment=null;
    if(path==='/receive/prepare')photoKind=['bol','lot_tag'].includes(initialAttachment?.kind)?initialAttachment.kind:['bol','lot_tag'].includes(photoKind)?photoKind:'lot_tag';
    let prepared=null,busy=false,finished=false,locked=false,attachment=initialAttachment,photoFile=initialPhoto,uploading=false,confirmations=[...initialConfirmations],lastLots=[],lastFormula=null,anchor=null,formulaApi=null,statusBox=null;
    const make=path==='/make/prepare';
    // Make: one status line at the top of the draft that never leaves doubt whether it was recorded.
    function status(kind,en,es,extra){if(!statusBox||(statusBox.parentNode??statusBox.parent)!==card){statusBox=ui.el('div');statusBox.setAttribute('role','status');statusBox.dataset.entryStatus='';card.prepend?card.prepend(statusBox):card.append(statusBox);}
      statusBox.className='entry-status '+kind;statusBox.dataset.entryStatus=kind;statusBox.replaceChildren();ui.el('strong',ui.t(en,es),statusBox);if(extra)extra(statusBox);statusBox.scrollIntoView?.({block:'nearest'});}
    function codeOf(error){const d=error?.result?.detail||error?.result||{};return d.error_code||d.code||'';}
    function receiptLink(number){return box=>{const a=ui.el('a',number,box);a.href=FLReceipts.link(number);};}
    const card=ui.el('section',undefined,host);card.className='panel';
    function editButton(){const cancel=ui.el('button',ui.t('Cancel','Cancelar'),card);cancel.type='button';cancel.className='secondary';cancel.onclick=()=>{if(busy||uploading||finished)return;finished=true;card.remove();document.getElementById('notice').hidden=true;if(onCancel)onCancel();else if(onEdit)onEdit();};if(onEdit){const edit=ui.el('button',ui.t('Edit entry','Editar entrada'),card);edit.type='button';edit.onclick=()=>{if(!busy&&!uploading&&!finished)onEdit();};}}
    function failure(error){ui.failure(error);onFailure(error);}
    function invalidate(){finished=true;card.replaceChildren();ui.el('p',ui.t('Draft cleared. Prepare again after signing in.','Borrador borrado. Prepara de nuevo tras iniciar sesión.'),card);}
    window.addEventListener('ledger-signout',invalidate,{once:true});
    async function prepare({refresh=false}={}){if(busy||finished)return;busy=true;
      // refresh: an edited Make draft stays on screen (dimmed) while FL re-checks it.
      if(refresh){card.setAttribute('aria-busy','true');card.classList?.add('updating');}else{card.replaceChildren();ui.el('p',ui.t('Preparing…','Preparando…'),card);}
      try{const next=await ui.request(path,{...payload,...(confirmations.length?{lot_confirmations:confirmations}:{})});if(finished)return;if(!next.ticket||!next.payload_hash||!next.draft)throw Error('FL did not return a draft / FL no devolvió un borrador');prepared=next;document.getElementById('notice').hidden=true;render();keepInView();onPrepared(prepared);}
      catch(error){if(finished)return;if(make)onFailure(error);else failure(error);if(refresh){anchor=null;if(make)status('bad','NOT RECORDED — the check did not finish: '+ui.errorText(error.result||{detail:{message:error.message}},'en')+' Your lots and pounds are kept; try again.','NO REGISTRADO — la verificación no terminó: '+ui.errorText(error.result||{detail:{message:error.message}},'es')+' Tus lotes y libras se conservan; intenta de nuevo.');return;}card.replaceChildren();
        if(make)status('bad','NOT RECORDED — FL could not prepare the draft: '+(error.transport?error.message:ui.errorText(error.result||{detail:{message:error.message}},'en'))+' Your entries are kept.','NO REGISTRADO — FL no pudo preparar el borrador: '+(error.transport?error.message:ui.errorText(error.result||{detail:{message:error.message}},'es'))+' Tus datos se conservan.');
        const retry=ui.el('button',ui.t('Try prepare again','Preparar de nuevo'),card);retry.type='button';retry.onclick=()=>prepare();editButton();}finally{busy=false;if(refresh){card.setAttribute('aria-busy','false');card.classList?.remove('updating');}}}
    function keepInView(){if(!anchor)return;const box=[...(card.querySelectorAll?.('.formula-ingredient')||[])].find(b=>b.dataset.productId===anchor.id),top=box?.getBoundingClientRect?.().top;if(top!=null&&anchor.top!=null)root.scrollBy?.(0,top-anchor.top);anchor=null;}
    function render(){card.replaceChildren();statusBox=null;
      if(make){const done=prepared.draft.already_recorded;if(done?.receipt_number&&done.status==='committed')status('good','Already RECORDED as '+done.receipt_number+'. Nothing new will be recorded.','Ya REGISTRADO como '+done.receipt_number+'. No se registrará nada nuevo.',receiptLink(done.receipt_number));else if(done?.status==='awaiting_approval')status('held','NOT RECORDED YET — this Make is already waiting for owner approval. Nothing new will be recorded.','AÚN NO REGISTRADO — esta producción ya espera la aprobación del dueño. No se registrará nada nuevo.');else status('draft','NOT RECORDED YET — this is a draft. Inventory changes only when you tap Record.','AÚN NO REGISTRADO — esto es un borrador. El inventario cambia solo cuando tocas Registrar.');}
      editButton();ui.el('h2',ui.t('Draft — review before recording','Borrador — revisa antes de registrar'),card);ui.el('p',ui.t('Recording as '+(prepared.actor?.name||prepared.draft.actor?.name||'—'),'Registrando como '+(prepared.actor?.name||prepared.draft.actor?.name||'—')),card);ui.el('p',ui.t('Expires: '+ui.plantTime(prepared.expires_at,'en'),'Vence: '+ui.plantTime(prepared.expires_at,'es')),card);
      const view=ui.el('div',undefined,card);view.className='draft-data';view.innerHTML=path==='/make/prepare'?ui.makeDraftHtml(prepared.draft,root.matchMedia?.('(max-width: 520px)').matches!==false):ui.draftHtml(prepared.draft);
      // Make (phone): FL's own warnings already name each short lot; a warning that needs
      // no tick is shown as a note, never as a checkbox (commit checks only requires_ack).
      const serverShortage=(prepared.warnings||[]).some(w=>w.code==='WILL_CREATE_SHORTAGE');
      if(make&&!serverShortage)for(const shortage of prepared.draft.shortages||[]){
        const name=shortage.product_name||ui.t('this ingredient','este ingrediente'),amount=ui.formatNumber(shortage.on_hand_lb??shortage.available_lb??0);
        const warning=ui.el('p',ui.t('Only '+amount+' lb of '+name+' on hand — this will be recorded as a shortage.','Solo hay '+amount+' lb de '+name+' — se registrará como un faltante.'),card);warning.className='notice';warning.dataset.shortageWarning='';
      }
      const ack=[];let evidenceDirty=false;
      (prepared.warnings||[]).forEach(w=>{if(make&&w.requires_ack===false){const note=ui.el('p',ui.t(ui.text(w,'en'),ui.text(w,'es')),card);note.className='notice';note.dataset.warningInfo=w.code;if(w.code==='WILL_CREATE_SHORTAGE')note.dataset.shortageWarning='';return;}const label=ui.el('label',undefined,card);label.className='check notice';const input=ui.el('input',undefined,label);input.type='checkbox';input.dataset.warning=w.code;ui.el('span',ui.t(ui.text(w,'en'),ui.text(w,'es')),label);ack.push([input,w.code]);});
      // Formula Make: untapped lots are shown on each lot and counted next to Record, not as blockers.
      const tapLater=make&&prepared.draft.formula_entry,otherBlockers=(prepared.blockers||[]).filter(b=>!(tapLater&&b.code==='LOT_NOT_CONFIRMED'));
      if(prepared.can_commit!==true&&(!tapLater||otherBlockers.length)){const state=ui.el('p',ui.t('NOT recorded — resolve the blockers below.','NO registrado — resuelve los bloqueos abajo.'),card);state.className='bad';}
      const blockers=otherBlockers;
      blockers.forEach(b=>{const p=ui.el('p',ui.t(ui.text(b,'en'),ui.text(b,'es')),card);p.className='bad';p.dataset.blocker=b.code||'';});
      if(prepared.draft.input_plan?.length)lastLots=prepared.draft.input_plan;
      // Formula-driven Make: ingredient cards replace the per-lot rows (make-formula.js).
      if(prepared.draft.formula_entry)lastFormula=prepared.draft.formula_entry;const formula=path==='/make/prepare'&&ui.formulaEntry?lastFormula:null;
      // The re-check button lives in the record bar (sticky on a phone) and shows once something changed.
      const actions=ui.el('div');actions.className=formula?'record-bar sticky':'record-bar';let updateHint=null;
      let record=null,pendingNote=null;
      formulaApi=formula?ui.formulaEntry(card,formula,{confirmations,checkHost:actions,changed:()=>syncRecord(),dirty:()=>{evidenceDirty=true;syncRecord();syncUpdate();},update:(lines,next)=>{if(busy||finished||uploading)return;payload.ingredient_lines=lines;confirmations=next;const at=formulaApi.anchor();anchor=at?{id:at.dataset.productId,top:at.getBoundingClientRect?.().top}:null;prepare({refresh:true});}}):null;
      function syncUpdate(){if(!formulaApi)return;formulaApi.check.hidden=!evidenceDirty;formulaApi.check.className=evidenceDirty?'primary':'secondary';if(updateHint)updateHint.hidden=!evidenceDirty;}
      // Formula Make: unconfirmed lots are tapped on this screen and sent with Record, so
      // "lots not confirmed" is the only draft blocker the browser may clear (FL re-checks).
      function committable(){if(locked||(formulaApi&&formulaApi.invalid()>0))return false;if(prepared.can_commit===true)return true;if(!formulaApi||evidenceDirty||prepared.draft.already_recorded)return false;return (prepared.blockers||[]).every(b=>b.code==='LOT_NOT_CONFIRMED')&&formulaApi.pending()===0;}
      function syncRecord(){if(!record)return;record.disabled=finished||uploading||evidenceDirty||!committable();
        if(pendingNote){const bad=formulaApi?formulaApi.invalid():0,n=formulaApi?formulaApi.pending():0;pendingNote.hidden=!bad&&(!n||evidenceDirty);pendingNote.className=bad?'bad':'muted';
          if(bad)ui.setText(pendingNote,ui.t('Record is off: '+bad+(bad===1?' amount is':' amounts are')+' negative. Fix '+(bad===1?'it':'them')+' first.','Registrar está apagado: '+bad+(bad===1?' cantidad es negativa':' cantidades son negativas')+'. Corrígelas primero.'));
          else ui.setText(pendingNote,ui.t(n+(n===1?' lot still needs a tap':' lots still need a tap')+' before Record.',(n===1?'Falta tocar 1 lote':'Falta tocar '+n+' lotes')+' antes de Registrar.'));}}
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
      card.append(actions);if(formulaApi){pendingNote=ui.el('p','');pendingNote.className='muted';pendingNote.dataset.pendingLots='';actions.append(pendingNote);updateHint=ui.el('p',ui.t('You changed a lot or pounds. Check them first, then Record.','Cambiaste un lote o libras. Verifícalos primero y luego Registra.'));updateHint.className='muted';updateHint.dataset.updateHint='';actions.append(updateHint);}
      record=ui.el('button',ui.t('Record','Registrar'),actions);record.type='button';record.className='primary';syncRecord();syncUpdate();
      record.onclick=async()=>{if(busy||finished||locked||uploading||record.disabled)return;busy=true;onBusy(true);record.disabled=true;const body={payload_hash:prepared.payload_hash,acknowledged_warnings:ack.filter(([input])=>input.checked).map(([,code])=>code)};
        if(formulaApi){body.lot_confirmations=formulaApi.selections();if(make)status('draft','Recording… do not close this page.','Registrando… no cierres esta página.');}
        if(attachment){if(path==='/receive/prepare'||photoKind==='bol')body.attachment_ids=[attachment.attachment_id];else body.attachment_id=attachment.attachment_id;}
        try{const result=await ui.request('/tickets/'+encodeURIComponent(prepared.ticket)+'/commit',body);if(finished)return;
          if(!ui.recorded(result)||(receiptPrefix&&!new RegExp('^'+receiptPrefix+'-\\d{6}-\\d{3,}$').test(result.receipt_number))){ui.notice(ui.t('NOT recorded — '+ui.errorText(result,'en'),'NO registrado — '+ui.errorText(result,'es')));const held=result.held||result.status==='awaiting_approval';if(make)status(held?'held':'bad',held?'NOT RECORDED YET — waiting for owner approval. Do not record it again.':'NOT RECORDED — '+ui.errorText(result,'en'),held?'AÚN NO REGISTRADO — espera la aprobación del dueño. No lo registres otra vez.':'NO REGISTRADO — '+ui.errorText(result,'es'));if(held){if(make){locked=true;record.disabled=true;}else finished=true;const a=ui.el('a',ui.t('Open owner approval queue','Ver cola de aprobación'),card);a.href=FLReceipts.page('exceptions.html');}return;}
          finished=true;formulaApi=null;document.getElementById('notice').hidden=true;card.replaceChildren();statusBox=null;if(make)status('good','RECORDED ✓ '+result.receipt_number+(result.replayed?' (FL had already saved it; nothing was recorded twice)':''),'REGISTRADO ✓ '+result.receipt_number+(result.replayed?' (FL ya lo tenía guardado; no se registró dos veces)':''));const tag=ui.el('div',undefined,card);tag.innerHTML=ui.receiveLotHtml(result);const p=ui.el('p',ui.t('Recorded — receipt ','Registrado — recibo '),card);p.className='good';const a=ui.el('a',result.receipt_number,p);a.href=FLReceipts.link(result.receipt_number);const substitutions=ui.el('div',undefined,card);substitutions.innerHTML=ui.autoSubstitutionsHtml(result.auto_substitutions);onRecorded(result);
        }catch(error){if(!make){failure(error);}else{onFailure(error);const code=codeOf(error),detail=error.result?.detail||{};
            if(error.transport)status('bad','NOT CONFIRMED — FL did not answer, so this Make may or may not be recorded. Tap Record again: FL will not record the same Make twice.','SIN CONFIRMAR — FL no respondió; esta producción puede o no estar registrada. Toca Registrar otra vez: FL no registrará la misma producción dos veces.');
            else if(code==='MAKE_ALREADY_RECORDED'){finished=true;formulaApi=null;status('good','Already RECORDED as '+(detail.receipt_number||'—')+'. Nothing new was recorded.','Ya REGISTRADO como '+(detail.receipt_number||'—')+'. No se registró nada nuevo.',detail.receipt_number?receiptLink(detail.receipt_number):null);onRecorded({receipt_number:detail.receipt_number,replayed:true});}
            else{const stale=STALE.includes(code)&&formulaApi;status('bad','NOT RECORDED — '+ui.errorText(error.result,'en')+' Your lots and pounds are kept.'+(stale?' Tap "Check again" for a fresh draft.':''),'NO REGISTRADO — '+ui.errorText(error.result,'es')+' Tus lotes y libras se conservan.'+(stale?' Toca "Verificar otra vez" para un borrador nuevo.':''),stale?box=>{const again=ui.el('button',ui.t('Check again (keeps my entries)','Verificar otra vez (conserva mis datos)'),box);again.type='button';again.className='primary';again.onclick=()=>{if(busy||finished)return;const {lines}=formulaApi.read();payload.ingredient_lines=lines;confirmations=formulaApi.selections();prepare({refresh:true});};}:null);}}
        }finally{busy=false;onBusy(false);if(!finished)syncRecord();else record.disabled=true;}};
    }
    // Make: what the worker entered on the draft, so "Edit entry" + Prepare keeps it (entry.js).
    function entries(){return formulaApi?{...formulaApi.read(),confirmations:formulaApi.selections()}:null;}
    prepare();return {invalidate,entries};
  };
})(window);
