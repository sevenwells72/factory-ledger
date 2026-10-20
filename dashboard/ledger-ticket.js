/* The draft is FL's response; the browser never calculates permission or stock rules. */
(function(root){
  const ui=root.FLLedger;
  ui.ticketCard=function(host,path,payload,{onRecorded=()=>{},onPrepared=()=>{},onFailure=()=>{},onBusy=()=>{},onEdit=null,onCancel=null,onAttachment=()=>{},initialAttachment=null,initialPhoto=null,photoKind='correction_photo',receiptPrefix=null,allowPhoto=true}={}){
    allowPhoto=allowPhoto&&!['/make/prepare','/pack/prepare'].includes(path);
    if(!allowPhoto)initialAttachment=null;
    if(path==='/receive/prepare')photoKind=['bol','lot_tag'].includes(initialAttachment?.kind)?initialAttachment.kind:['bol','lot_tag'].includes(photoKind)?photoKind:'lot_tag';
    let prepared=null,busy=false,finished=false,attachment=initialAttachment,photoFile=initialPhoto,uploading=false,confirmations=[],lastLots=[];
    const card=ui.el('section',undefined,host);card.className='panel';
    function editButton(){const cancel=ui.el('button',ui.t('Cancel','Cancelar'),card);cancel.type='button';cancel.className='secondary';cancel.onclick=()=>{if(busy||uploading||finished)return;finished=true;card.remove();document.getElementById('notice').hidden=true;if(onCancel)onCancel();else if(onEdit)onEdit();};if(onEdit){const edit=ui.el('button',ui.t('Edit entry','Editar entrada'),card);edit.type='button';edit.onclick=()=>{if(!busy&&!uploading&&!finished)onEdit();};}}
    function failure(error){ui.failure(error);onFailure(error);}
    function invalidate(){finished=true;card.replaceChildren();ui.el('p',ui.t('Draft cleared. Prepare again after signing in.','Borrador borrado. Prepara de nuevo tras iniciar sesión.'),card);}
    window.addEventListener('ledger-signout',invalidate,{once:true});
    // Only explicit actions create evidence. A selected FIFO lot starts unchecked.
    const isMake=path==='/make/prepare';
    let acceptedConfirmations=[],prepareFailed=false;
    async function prepare(){if(busy||finished)return;busy=true;card.replaceChildren();ui.el('p',ui.t('Preparing…','Preparando…'),card);
      try{const next=await ui.request(path,{...payload,...(confirmations.length?{lot_confirmations:confirmations}:{})});if(finished)return;if(!next.ticket||!next.payload_hash||!next.draft)throw Error('FL did not return a draft / FL no devolvió un borrador');prepared=next;prepareFailed=false;acceptedConfirmations=confirmations.map(item=>({...item}));document.getElementById('notice').hidden=true;render();onPrepared(prepared);}
      catch(error){if(finished)return;failure(error);if(isMake&&prepared){prepareFailed=true;confirmations=acceptedConfirmations.map(item=>({...item}));render();}else{card.replaceChildren();const retry=ui.el('button',ui.t('Try prepare again','Preparar de nuevo'),card);retry.type='button';retry.onclick=prepare;editButton();}}finally{busy=false;}}
    function compactMake(host,changed){
      const plan=prepared.draft.input_plan||[],groups=prepared.draft.lot_confirmation_group?.ingredients||[];
      // Group by numeric product identity, including every lot in a split FIFO plan.
      const ingredients=new Map();
      for(const ingredient of prepared.draft.ingredients||[]){const existing=ingredients.get(ingredient.ingredient_id);ingredients.set(ingredient.ingredient_id,{...ingredient,product_id:ingredient.ingredient_id,product_name:ingredient.ingredient_name||ingredient.product_name||ingredient.name,needed_lb:Number(existing?.needed_lb||0)+Number(ingredient.needed_lb||0)});}
      for(const lot of plan)if(!ingredients.has(lot.product_id))ingredients.set(lot.product_id,{product_id:lot.product_id,product_name:lot.product_name,needed_lb:plan.filter(item=>item.product_id===lot.product_id).reduce((sum,item)=>sum+Number(item.quantity_lb||0),0)});
      const progress=ui.el('p',undefined,host);progress.className='make-progress';progress.dataset.makeProgress='';progress.setAttribute('role','status');
      const instruction=ui.el('p',ui.t('Check the bag tag, then tap ✓ for each lot you used. Suggested lots are oldest first.','Revisa la etiqueta de la bolsa y toca ✓ para cada lote usado. Se sugieren primero los más antiguos.'),host);instruction.className='make-instruction';
      const rows=[],checks=[];
      const hasEvidence=lot=>confirmations.some(item=>item.lot_id===lot.lot_id);
      const allChecked=()=>[...ingredients.values()].every(ingredient=>{const lots=plan.filter(lot=>lot.product_id===ingredient.product_id);return lots.length>0&&lots.every(hasEvidence);});
      function paint(){
        const count=rows.filter(row=>row.lots.length>0&&row.lots.every(hasEvidence)).length;
        ui.setText(progress,ui.t(count+' of '+rows.length+' checked',count+' de '+rows.length+' verificados'));
        for(const {lot,button} of checks){const checked=hasEvidence(lot);button.disabled=prepareFailed;button.setAttribute('aria-pressed',String(checked));button.className='make-check'+(checked?' is-checked':'');}
      }
      function chooseLot(productId,candidate){
        if(busy||finished||uploading||prepareFailed)return;
        // Changing identity discards that ingredient's checks, never checks the new tag.
        const substitutions=(payload.substitutions||[]).filter(item=>item.substitute_product_id===productId);
        substitutions.forEach(item=>item.lot_id=candidate.lot_id);
        const override=(payload.ingredient_lots||[]).find(item=>item.ingredient_product_id===productId);
        if(override)override.lot_id=candidate.lot_id;
        else if(!substitutions.length)payload.ingredient_lots=[...(payload.ingredient_lots||[]),{ingredient_product_id:productId,lot_id:candidate.lot_id}];
        confirmations=confirmations.filter(item=>!plan.some(lot=>lot.product_id===productId&&lot.lot_id===item.lot_id));
        acceptedConfirmations=acceptedConfirmations.filter(item=>!plan.some(lot=>lot.product_id===productId&&lot.lot_id===item.lot_id));
        changed();prepare();
      }
      function codeLabel(host,code){const full=ui.el('span',code,host);full.className='make-lot-code';const suffix=ui.el('strong',String(code||'').slice(-4),host);suffix.className='make-lot-suffix';}
      for(const ingredient of ingredients.values()){
        const productId=ingredient.product_id,lots=plan.filter(lot=>lot.product_id===productId),group=groups.find(item=>item.product_id===productId);
        const row=ui.el('section',undefined,host);row.className='make-ingredient';row.dataset.makeIngredient=String(productId);rows.push({lots,row});
        const heading=ui.el('h3',undefined,row);ui.el('span',ingredient.product_name||group?.product_name||ui.t('Ingredient','Ingrediente'),heading);ui.el('span',ui.t(' · '+ui.formatNumber(ingredient.needed_lb??group?.needed_lb??0)+' lb needed',' · '+ui.formatNumber(ingredient.needed_lb??group?.needed_lb??0)+' lb necesarios'),heading);
        const summary=ui.el('div',undefined,row);summary.className='make-lot-summary';const selected=ui.el('div',undefined,summary);selected.className='make-selected-lots';
        for(const lot of lots){
          const line=ui.el('div',undefined,selected);line.className='make-selected-lot';const identity=ui.el('div',undefined,line);identity.className='make-lot-identity';codeLabel(identity,lot.lot_code);
          if(lots.length>1)ui.el('small',ui.formatNumber(lot.quantity_lb)+' lb',identity);
          const button=ui.el('button',undefined,line);button.type='button';button.dataset.makeLotConfirm=String(lot.lot_id);ui.el('span','✓',button);const explanation=ui.el('span',ui.t('I checked the bag tag and it matches this lot: '+lot.lot_code,'Revisé la etiqueta de la bolsa y coincide con este lote: '+lot.lot_code),button);explanation.className='sr-only';checks.push({lot,button});
          button.onclick=()=>{if(busy||finished||uploading||prepareFailed)return;const wasChecked=hasEvidence(lot);acceptedConfirmations=acceptedConfirmations.filter(item=>item.lot_id!==lot.lot_id);confirmations=confirmations.filter(item=>item.lot_id!==lot.lot_id);if(!wasChecked)confirmations.push({lot_id:lot.lot_id,method:'tap',value:lot.lot_code});paint();changed();if(allChecked())prepare();};
          const type=ui.el('button',ui.t('Type code instead','Escribir código'),identity);type.type='button';type.disabled=prepareFailed;type.className='text-button';type.dataset.makeTypeCode=String(lot.lot_id);type.setAttribute('aria-expanded','false');
          const manual=ui.el('div',undefined,row);manual.className='make-manual-code';manual.hidden=true;
          type.onclick=()=>{if(busy||finished||uploading||prepareFailed)return;manual.hidden=!manual.hidden;type.setAttribute('aria-expanded',String(!manual.hidden));};
          const methodLabel=ui.el('label',ui.t('Confirmation method','Método de confirmación'),manual),method=ui.el('select',undefined,methodLabel);method.dataset.makeCodeMethod=String(lot.lot_id);
          [['last4','Type last 4 characters','Escribir últimos 4 caracteres'],['full_code','Type full code','Escribir código completo'],['scan','Scan with a barcode reader','Escanear con lector de códigos'],['pallet','Pallet moved to production — full code','Palé movido a producción — código completo']].forEach(([value,en,es])=>{ui.el('option',ui.t(en,es),method).value=value;});method.value='last4';
          const valueLabel=ui.el('label',ui.t('Code from the bag tag','Código de la etiqueta de la bolsa'),manual),value=ui.el('input',undefined,valueLabel);value.autocomplete='off';value.dataset.makeCodeValue=String(lot.lot_id);
          const prior=confirmations.find(item=>item.lot_id===lot.lot_id&&item.method!=='tap');if(prior){method.value=prior.method;value.value=prior.value;}
          const matches=ui.el('div',undefined,manual);matches.className='make-lot-choices';
          const check=ui.el('button',ui.t('Check code','Verificar código'),manual);check.type='button';check.disabled=prepareFailed;check.dataset.makeCodeCheck=String(lot.lot_id);
          for(const control of [method,value])control.oninput=()=>{acceptedConfirmations=acceptedConfirmations.filter(item=>item.lot_id!==lot.lot_id);confirmations=confirmations.filter(item=>item.lot_id!==lot.lot_id);matches.replaceChildren();paint();changed();};
          check.onclick=()=>{if(busy||finished||uploading||prepareFailed)return;const entered=value.value.trim();if(!entered){ui.failure(ui.problem(ui.t('Enter the code from the bag tag.','Escribe el código de la etiqueta de la bolsa.')));return;}
            const candidates=(group?.candidates||[]).filter(candidate=>candidate.lot_code.toUpperCase().endsWith(entered.toUpperCase()));
            if(method.value==='last4'&&entered.length===4&&candidates.length>1){matches.replaceChildren();ui.el('p',ui.t('More than one lot matches. Choose the full code on the bag, then check its tag:','Coincide más de un lote. Elige el código completo de la bolsa y confirma su etiqueta:'),matches);for(const candidate of candidates){const option=ui.el('button',candidate.lot_code,matches);option.type='button';option.dataset.confirmLot=String(candidate.lot_id);option.onclick=()=>chooseLot(productId,candidate);}return;}
            confirmations=confirmations.filter(item=>item.lot_id!==lot.lot_id);confirmations.push({lot_id:lot.lot_id,method:method.value,value:entered});changed();prepare();};
        }
        const change=ui.el('button',ui.t('Change','Cambiar'),summary);change.type='button';change.disabled=prepareFailed;change.className='make-change';change.dataset.makeChange=String(productId);change.setAttribute('aria-expanded','false');
        const choices=ui.el('div',undefined,row);choices.className='make-lot-choices';choices.hidden=true;change.onclick=()=>{if(busy||finished||uploading||prepareFailed)return;choices.hidden=!choices.hidden;change.setAttribute('aria-expanded',String(!choices.hidden));};
        ui.el('p',ui.t('Tap the lot you used, then check its bag tag.','Toca el lote usado y luego confirma su etiqueta.'),choices);
        const candidates=group?.candidates||[];
        if(!candidates.length)ui.el('p',ui.t('No other lots on hand. Edit the entry to choose a different ingredient or receive its lot first.','No hay otros lotes disponibles. Edita la entrada para elegir otro ingrediente o registra primero la recepción del lote.'),choices);
        const lotList=ui.el('div',undefined,choices);lotList.className='make-lot-choices';
        let shown=0;const moreLots=ui.el('button',ui.t('More lots','Más lotes'),choices);moreLots.type='button';moreLots.disabled=prepareFailed;moreLots.dataset.makeMoreLots=String(productId);
        function showLots(){for(const candidate of candidates.slice(shown,shown+6)){
          const choice=ui.el('button',undefined,lotList);choice.type='button';choice.disabled=prepareFailed;choice.dataset.makeLotChoice=String(candidate.lot_id);codeLabel(choice,candidate.lot_code);
          const received=lang=>candidate.received_at?new Date(candidate.received_at).toLocaleDateString(lang==='es'?'es-US':'en-US',{timeZone:'America/New_York',year:'numeric',month:'short',day:'numeric'}):(lang==='es'?'Fecha no disponible':'Date unavailable');
          ui.el('span',ui.t('Received '+received('en')+' · '+ui.formatNumber(candidate.available_lb)+' lb available','Recibido '+received('es')+' · '+ui.formatNumber(candidate.available_lb)+' lb disponibles'),choice);choice.onclick=()=>chooseLot(productId,candidate);
        }
        shown+=6;moreLots.hidden=shown>=candidates.length;}moreLots.onclick=()=>{if(!busy&&!finished&&!prepareFailed)showLots();};showLots();
        const shortages=(prepared.draft.shortages||[]).filter(item=>item.product_id===productId||!item.product_id&&item.product_name===ingredient.product_name);
        for(const shortage of shortages){const warning=ui.el('p',ui.t('Only '+ui.formatNumber(shortage.on_hand_lb??shortage.available_lb??0)+' lb on hand — short '+ui.formatNumber(shortage.short_lb)+' lb. This will be recorded as a shortage.','Solo hay '+ui.formatNumber(shortage.on_hand_lb??shortage.available_lb??0)+' lb — faltan '+ui.formatNumber(shortage.short_lb)+' lb. Se registrará como un faltante.'),row);warning.className='make-shortage';warning.dataset.makeShortage=String(productId);warning.dataset.shortageWarning='';}
      }
      paint();return {allChecked,paint};
    }
    function render(){card.replaceChildren();editButton();ui.el('h2',ui.t('Draft — review before recording','Borrador — revisa antes de registrar'),card);if(isMake){ui.el('p',ui.t('Recording as '+(prepared.actor?.name||prepared.draft.actor?.name||'—')+' · Expires '+ui.plantTime(prepared.expires_at,'en'),'Registrando como '+(prepared.actor?.name||prepared.draft.actor?.name||'—')+' · Vence '+ui.plantTime(prepared.expires_at,'es')),card).className='make-meta';}else{ui.el('p',ui.t('Recording as '+(prepared.actor?.name||prepared.draft.actor?.name||'—'),'Registrando como '+(prepared.actor?.name||prepared.draft.actor?.name||'—')),card);ui.el('p',ui.t('Expires: '+ui.plantTime(prepared.expires_at,'en'),'Vence: '+ui.plantTime(prepared.expires_at,'es')),card);}
      if(isMake&&prepareFailed){ui.el('p',ui.t('The draft could not be refreshed. Retry before recording, or edit the entry.','No se pudo actualizar el borrador. Reintenta antes de registrar o edita la entrada.'),card).className='bad';const retry=ui.el('button',ui.t('Retry draft','Reintentar borrador'),card);retry.type='button';retry.dataset.makeRetry='';retry.onclick=prepare;}
      const view=ui.el('div',undefined,card);view.className='draft-data';
      if(isMake){
        view.className+=' make-summary';
        const {ingredients,input_plan,lot_confirmation_group,shortages,product_name,product,batches,total_output_lb,lot_code,kosher_tier,kosher_label,note,substitutions,auto_substitutions,excluded_ingredients,...details}=prepared.draft;
        ui.el('h3',product_name||product||ui.t('Make','Producir'),view);
        const amount=ui.el('p',undefined,view);if(batches!==undefined)ui.el('span',ui.t(ui.formatNumber(batches)+(Number(batches)===1?' batch':' batches'),ui.formatNumber(batches)+(Number(batches)===1?' lote de producción':' lotes de producción')),amount);if(total_output_lb!==undefined)ui.el('span',' · '+ui.formatNumber(total_output_lb)+' lb',amount);
        if(kosher_label)ui.el('span',' · '+kosher_label,amount);
        if(lot_code)ui.el('p',ui.t('Output lot: '+lot_code,'Lote de salida: '+lot_code),view);
        if(note)ui.el('p',ui.t('Note: '+note,'Nota: '+note),view);
        const changes=ui.el('div',undefined,view);changes.innerHTML=ui.autoSubstitutionsHtml(auto_substitutions);for(const item of substitutions||[])ui.el('p',ui.t('Substitution: '+item.ingredient_name+' → '+item.substitute_name+' · '+item.reason_code,'Sustitución: '+item.ingredient_name+' → '+item.substitute_name+' · '+item.reason_code),changes);for(const item of excluded_ingredients||[])ui.el('p',ui.t('Excluded ingredient: '+item.ingredient_name,'Ingrediente excluido: '+item.ingredient_name),changes);
        const more=ui.el('details',undefined,view);ui.el('summary',ui.t('Entry details','Detalles de la entrada'),more);const body=ui.el('div',undefined,more);body.innerHTML=ui.draftHtml(details);
      }else view.innerHTML=ui.draftHtml(prepared.draft);
      const ack=[];let evidenceDirty=false,makeState=null;
      const changed=()=>{evidenceDirty=true;record.disabled=true;};
      if(isMake)makeState=compactMake(card,changed);
      (prepared.warnings||[]).forEach(w=>{const label=ui.el('label',undefined,card);label.className='check notice';const input=ui.el('input',undefined,label);input.type='checkbox';input.dataset.warning=w.code;ui.el('span',ui.t(ui.text(w,'en'),ui.text(w,'es')),label);ack.push([input,w.code]);});
      const blockers=(prepared.blockers||[]).filter(item=>!isMake||(item.code||item.error_code)!=='LOT_NOT_CONFIRMED');
      if(prepared.can_commit!==true&&(!isMake||blockers.length)){const state=ui.el('p',ui.t('NOT recorded — resolve the blockers below.','NO registrado — resuelve los bloqueos abajo.'),card);state.className='bad';}
      blockers.forEach(b=>{const p=ui.el('p',ui.t(ui.text(b,'en'),ui.text(b,'es')),card);p.className='bad';});
      if(!isMake){
      if(prepared.draft.input_plan?.length)lastLots=prepared.draft.input_plan;
      const lots=new Map(lastLots.map(l=>[l.lot_id,l]));const inputs=[];
      lots.forEach(lot=>{const field=ui.el('fieldset',undefined,card);field.className='lot-confirmation-row';field.dataset.productId=lot.product_id;
        ui.el('legend',ui.t('Confirm physical lot: '+(lot.product_name?lot.product_name+' · ':'')+lot.lot_code,'Confirma lote físico: '+(lot.product_name?lot.product_name+' · ':'')+lot.lot_code),field);
        const controls=ui.el('div',undefined,field);controls.className='lot-confirmation-controls';
        const label=ui.el('label',ui.t('Confirmation method','Método de confirmación'),controls),method=ui.el('select',undefined,label);
        [['last4',ui.t('Last 4 characters (include hyphen)','Últimos 4 caracteres (incluye guion)')],['full_code',ui.t('Full code','Código completo')],['pallet',ui.t('Pallet — full code, moved to production','Palé — código completo, movido a producción')]].forEach(([v,t])=>{const o=ui.el('option',t,method);o.value=v;});
        const label2=ui.el('label',ui.t('Type the code from the lot label','Escribe el código de la etiqueta'),controls),value=ui.el('input',undefined,label2);value.autocomplete='off';const prev=confirmations.find(c=>c.lot_id===lot.lot_id);if(prev){method.value=prev.method;value.value=prev.value;}
        const input={lot,method,value,read:()=>({lot_id:lot.lot_id,method:method.value,value:value.value})};inputs.push(input);
        for(const control of [method,value])control.oninput=changed;
      });
      if(lots.size){const check=ui.el('button',ui.t('Check confirmations → new draft','Verificar confirmaciones → nuevo borrador'),card);check.type='button';check.onclick=()=>{if(busy||finished||uploading)return;confirmations=inputs.map(input=>input.read());prepare();};}
      }
      if(allowPhoto){
      let photoType=null;if(path==='/receive/prepare'){const kindLabel=ui.el('label',ui.t('Photo type','Tipo de foto'),card);photoType=ui.el('select',undefined,kindLabel);photoType.name='draft_photo_kind';for(const [value,en,es] of [['lot_tag','Label / lot tag','Etiqueta / etiqueta de lote'],['bol','BOL','BOL']]){const option=ui.el('option',ui.t(en,es),photoType);option.value=value;}photoType.value=photoKind;}
      const uploadLabel=ui.el('label',ui.t('Attach photo','Adjuntar foto'),card),photo=ui.el('input',undefined,uploadLabel);photo.type='file';photo.accept='image/jpeg,image/png,image/webp';photo.setAttribute('capture','environment');
      const uploadStatus=ui.el('p',attachment?ui.t('Photo attached','Foto adjunta'):'',card);
      async function uploadPhoto(file,kind,{retype=false}={}){if(!file||busy||finished||uploading)return;const previousKind=photoKind;uploading=true;photo.disabled=true;if(photoType)photoType.disabled=true;onBusy(true);record.disabled=true;ui.setText(uploadStatus,ui.t('Uploading photo…','Subiendo foto…'));try{const uploaded=await ui.upload(file,kind);if(finished)return;attachment=uploaded;photoFile=file;photoKind=kind;onAttachment(attachment,photoFile);ui.setText(uploadStatus,ui.t('Photo attached','Foto adjunta'));}catch(error){if(!finished){if(retype){photoKind=previousKind;photoType.value=previousKind;ui.setText(uploadStatus,ui.t('Photo type was not changed. The attached photo keeps its previous type.','No se cambió el tipo de foto. La foto adjunta conserva su tipo anterior.'));}else ui.setText(uploadStatus,attachment?ui.t('Photo attached','Foto adjunta'):ui.t('Photo was not attached. Try again.','La foto no se adjuntó. Intenta de nuevo.'));failure(error);}}finally{uploading=false;photo.disabled=false;if(photoType)photoType.disabled=false;if(!finished)onBusy(false);record.disabled=busy||finished||evidenceDirty||prepared.can_commit!==true;}}
      if(photoType)photoType.onchange=()=>{if(busy||finished||uploading)return;if(attachment){if(photoFile)return uploadPhoto(photoFile,photoType.value,{retype:true});photoType.value=photoKind;ui.setText(uploadStatus,ui.t('Choose the photo again to change its type.','Elige la foto de nuevo para cambiar su tipo.'));return;}photoKind=photoType.value;};
      photo.onchange=()=>uploadPhoto(photo.files?.[0],photoType?.value||photoKind);
      }
      const actions=ui.el('div',undefined,card);actions.className='record-bar';const record=ui.el('button',ui.t('Record','Registrar'),actions);record.type='button';record.className='primary';record.disabled=prepareFailed||prepared.can_commit!==true||!!makeState&&!makeState.allChecked();
      record.onclick=async()=>{if(busy||finished||uploading||record.disabled||prepareFailed||evidenceDirty||makeState&&!makeState.allChecked())return;busy=true;onBusy(true);record.disabled=true;const body={payload_hash:prepared.payload_hash,acknowledged_warnings:ack.filter(([input])=>input.checked).map(([,code])=>code)};
        if(attachment){if(path==='/receive/prepare'||photoKind==='bol')body.attachment_ids=[attachment.attachment_id];else body.attachment_id=attachment.attachment_id;}
        try{const result=await ui.request('/tickets/'+encodeURIComponent(prepared.ticket)+'/commit',body);if(finished)return;
          if(!ui.recorded(result)||(receiptPrefix&&!new RegExp('^'+receiptPrefix+'-\\d{6}-\\d{3,}$').test(result.receipt_number))){ui.notice(ui.t('NOT recorded — '+ui.errorText(result,'en'),'NO registrado — '+ui.errorText(result,'es')));if(result.held||result.status==='awaiting_approval'){finished=true;const a=ui.el('a',ui.t('Open owner approval queue','Ver cola de aprobación'),card);a.href=FLReceipts.page('exceptions.html');}return;}
          finished=true;document.getElementById('notice').hidden=true;card.replaceChildren();const p=ui.el('p',ui.t('Recorded — receipt ','Registrado — recibo '),card);p.className='good';const a=ui.el('a',result.receipt_number,p);a.href=FLReceipts.link(result.receipt_number);const substitutions=ui.el('div',undefined,card);substitutions.innerHTML=ui.autoSubstitutionsHtml(result.auto_substitutions);onRecorded(result);
        }catch(error){failure(error);}finally{busy=false;onBusy(false);record.disabled=finished||prepareFailed||evidenceDirty||prepared.can_commit!==true||!!makeState&&!makeState.allChecked();}};
    }
    prepare();return {invalidate};
  };
})(window);
