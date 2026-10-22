/* The draft is FL's response; the browser never calculates permission or stock rules. */
(function(root){
  const ui=root.FLLedger;
  ui.ticketCard=function(host,path,payload,{onRecorded=()=>{},onPrepared=()=>{},onFailure=()=>{},onBusy=()=>{},onEdit=null,onCancel=null,onAttachment=()=>{},initialAttachment=null,initialPhoto=null,photoKind='correction_photo',receiptPrefix=null,allowPhoto=true}={}){
    allowPhoto=allowPhoto&&!['/make/prepare','/pack/prepare'].includes(path);
    if(!allowPhoto)initialAttachment=null;
    if(path==='/receive/prepare')photoKind=['bol','lot_tag'].includes(initialAttachment?.kind)?initialAttachment.kind:['bol','lot_tag'].includes(photoKind)?photoKind:'lot_tag';
    let prepared=null,busy=false,finished=false,attachment=initialAttachment,photoFile=initialPhoto,uploading=false,confirmations=[...(payload.lot_confirmations||[])],lastLots=[];
    const card=ui.el('section',undefined,host);card.className='panel';
    function editButton(){const cancel=ui.el('button',ui.t('Cancel','Cancelar'),card);cancel.type='button';cancel.className='secondary';cancel.onclick=()=>{if(busy||uploading||finished)return;finished=true;card.remove();document.getElementById('notice').hidden=true;if(onCancel)onCancel();else if(onEdit)onEdit();};if(onEdit){const edit=ui.el('button',ui.t('Edit entry','Editar entrada'),card);edit.type='button';edit.onclick=()=>{if(!busy&&!uploading&&!finished)onEdit();};}}
    function failure(error){ui.failure(error);onFailure(error);}
    function invalidate(){finished=true;card.replaceChildren();ui.el('p',ui.t('Draft cleared. Prepare again after signing in.','Borrador borrado. Prepara de nuevo tras iniciar sesión.'),card);}
    window.addEventListener('ledger-signout',invalidate,{once:true});
    async function prepare(){if(busy||finished)return;busy=true;card.replaceChildren();ui.el('p',ui.t('Preparing…','Preparando…'),card);
      try{prepared=await ui.request(path,{...payload,...(confirmations.length?{lot_confirmations:confirmations}:{})});if(finished)return;if(!prepared.ticket||!prepared.payload_hash||!prepared.draft)throw Error('FL did not return a draft / FL no devolvió un borrador');document.getElementById('notice').hidden=true;render();onPrepared(prepared);}
      catch(error){if(finished)return;failure(error);card.replaceChildren();const retry=ui.el('button',ui.t('Try prepare again','Preparar de nuevo'),card);retry.type='button';retry.onclick=prepare;editButton();}finally{busy=false;}}
    function render(){card.replaceChildren();editButton();ui.el('h2',ui.t('Draft — review before recording','Borrador — revisa antes de registrar'),card);ui.el('p',ui.t('Recording as '+(prepared.actor?.name||prepared.draft.actor?.name||'—'),'Registrando como '+(prepared.actor?.name||prepared.draft.actor?.name||'—')),card);ui.el('p',ui.t('Expires: '+ui.plantTime(prepared.expires_at,'en'),'Vence: '+ui.plantTime(prepared.expires_at,'es')),card);
      const ingredientGroups=prepared.draft.lot_confirmation_group?.ingredients||[];
      const stockGroups=['/make/prepare','/pack/prepare'].includes(path)&&ingredientGroups.some(group=>Number.isFinite(group.needed_lb));
      // The selector below is the reviewed physical input plan; avoid presenting
      // unchosen FIFO codes as though they were lots the worker already used.
      const draftView=stockGroups?{...prepared.draft,input_plan:undefined,ingredients:prepared.draft.ingredients?.map(({lots,...ingredient})=>ingredient)}:prepared.draft;
      const view=ui.el('div',undefined,card);view.className='draft-data';view.innerHTML=stockGroups?ui.draftHtml(draftView):path==='/make/prepare'?ui.makeDraftHtml(draftView,root.matchMedia?.('(max-width: 520px)').matches!==false):ui.draftHtml(draftView);
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
      if(stockGroups){
        const receivedDate=(value,lang)=>{if(!value)return '—';const raw=String(value),date=new Date(raw);return Number.isNaN(date.getTime())?raw:new Intl.DateTimeFormat(lang==='es'?'es-US':'en-US',{year:'numeric',month:'short',day:'numeric',timeZone:/^\d{4}-\d{2}-\d{2}$/.test(raw)?'UTC':'America/New_York'}).format(date);};
        const normalize=value=>String(value??'').replace(/\s/g,'').toUpperCase();
        const groups=ingredientGroups.map(group=>{
          const field=ui.el('fieldset',undefined,card);field.className='ingredient-lot-card';field.dataset.productId=String(group.product_id);
          ui.el('legend',ui.t('Confirm physical lots: '+group.product_name+' · '+ui.formatNumber(group.needed_lb)+' lb needed','Confirma lotes físicos: '+group.product_name+' · '+ui.formatNumber(group.needed_lb)+' lb necesarios'),field);
          ui.el('p',ui.t('Select every lot on the bags you are using. If you use more than one, split the needed lb between them. FIFO is a suggestion; you must choose.','Selecciona cada lote de las bolsas que usas. Si usas más de uno, reparte las lb necesarias entre ellos. FIFO es una sugerencia; debes elegir.'),field);
          const status=ui.el('p','',field);status.setAttribute('role','status');status.className='lot-selection-status';
          const list=ui.el('div',undefined,field);list.className='ingredient-lot-list';
          const candidates=[...(group.candidates||[])].filter(lot=>Number(lot.available_lb)>0).sort((a,b)=>Number(!!b.opening_stock)-Number(!!a.opening_stock)||String(a.received_at||'9999').localeCompare(String(b.received_at||'9999'))||a.lot_id-b.lot_id);
          const state={group,rows:[],status,typedDirty:false};
          function dirty(){evidenceDirty=true;record.disabled=true;ui.setText(status,'');}
          candidates.forEach(lot=>{
            const row=ui.el('div',undefined,list);row.className='ingredient-lot-option'+(lot.fifo_suggested?' fifo-suggested':'');row.dataset.stockLot=String(lot.lot_id);
            const label=ui.el('label',undefined,row);label.className='ingredient-lot-choice';const selected=ui.el('input',undefined,label);selected.type='checkbox';selected.name='ingredient_lot_id';selected.value=String(lot.lot_id);
            const detail=ui.el('span',undefined,label);detail.className='ingredient-lot-detail';
            const supplier=ui.el('strong',lot.opening_stock?ui.t(lot.label_en||'OPENING STOCK — use for any bag already here on Oct 21',lot.label_es||'EXISTENCIAS INICIALES — use para cualquier bolsa que ya estaba aquí el 21 de octubre'):(lot.supplier_lot_code||ui.t('Supplier lot not recorded','Lote del proveedor no registrado')),detail);supplier.className='supplier-lot-code';
            const aliases=[...new Set(lot.supplier_lot_codes||[])].filter(code=>code&&normalize(code)!==normalize(lot.supplier_lot_code));
            if(aliases.length)ui.el('span',ui.t('Also on this lot: '+aliases.join(' / '),'También en este lote: '+aliases.join(' / ')),detail);
            ui.el('span',ui.t('CNS lot: '+lot.lot_code,'Lote CNS: '+lot.lot_code),detail);
            ui.el('span',ui.t('Supplier: '+(lot.supplier_name||'—'),'Proveedor: '+(lot.supplier_name||'—')),detail);
            ui.el('span',ui.t('Received: '+(receivedDate(lot.received_at,'en')),'Recibido: '+(receivedDate(lot.received_at,'es'))),detail);
            ui.el('span',ui.t(ui.formatNumber(lot.available_lb)+' lb available',ui.formatNumber(lot.available_lb)+' lb disponibles'),detail);
            if(lot.fifo_suggested){const badge=ui.el('span',ui.t('FIFO suggestion','Sugerencia FIFO'),detail);badge.className='fifo-badge';}
            const amountLabel=ui.el('label',ui.t('lb from this lot','lb de este lote'),row),amount=ui.el('input',undefined,amountLabel);amount.type='number';amount.name='ingredient_lot_quantity';amount.step='any';amount.min='0';amount.disabled=true;
            const previous=confirmations.find(value=>value.lot_id===lot.lot_id),plan=(prepared.draft.input_plan||[]).find(value=>value.lot_id===lot.lot_id&&value.product_id===group.product_id);
            if(previous&&plan){selected.checked=true;amount.disabled=false;amount.value=String(plan.quantity_lb);}
            const item={lot,selected,amount,previous,explicit:previous?.selected===true};state.rows.push(item);
            selected.onchange=()=>{amount.disabled=!selected.checked;if(selected.checked&&!amount.value&&state.rows.filter(value=>value.selected.checked).length===1)amount.value=String(Number(group.needed_lb));state.typedDirty=false;state.value.value='';item.previous=null;item.explicit=selected.checked;dirty();};
            amount.oninput=dirty;
          });
          if(!candidates.length)ui.el('p',ui.t('No lots are in stock for this ingredient.','No hay lotes con inventario para este ingrediente.'),list);
          const controls=ui.el('div',undefined,field);controls.className='lot-confirmation-controls';
          const methodLabel=ui.el('label',ui.t('Or type / scan the bag code','O escribe / escanea el código de la bolsa'),controls),method=ui.el('select',undefined,methodLabel);method.name='ingredient_confirmation_method';
          for(const [value,en,es] of [['supplier_lot','Supplier lot','Lote del proveedor'],['last4','Last 4 CNS characters (include hyphen)','Últimos 4 caracteres CNS (incluye guion)'],['full_code','Full CNS code','Código CNS completo'],['scan','Scan','Escanear'],['pallet','Pallet — full code, moved to production','Palé — código completo, movido a producción']]){const option=ui.el('option',ui.t(en,es),method);option.value=value;}method.value='supplier_lot';
          const codeLabel=ui.el('label',ui.t('Code from the bag','Código de la bolsa'),controls),value=ui.el('input',undefined,codeLabel);value.autocomplete='off';value.name='ingredient_confirmation_value';state.method=method;state.value=value;
          value.oninput=()=>{state.typedDirty=true;dirty();};method.onchange=()=>{state.typedDirty=true;dirty();};
          state.read=()=>{
            let chosen=state.rows.filter(row=>row.selected.checked);
            const typed=value.value.trim();
            if(state.typedDirty&&typed){
              const matches=state.rows.filter(row=>method.value==='supplier_lot'?[row.lot.supplier_lot_code,...(row.lot.supplier_lot_codes||[])].some(code=>code&&normalize(code)===normalize(typed)):method.value==='last4'?normalize(row.lot.lot_code).endsWith(normalize(typed)):normalize(row.lot.lot_code)===normalize(typed));
              if(matches.length===1&&chosen.length<=1){const row=matches[0],explicit=chosen.includes(row)&&row.explicit;state.rows.forEach(value=>{value.selected.checked=false;value.amount.disabled=true;value.explicit=false;});row.explicit=explicit;row.selected.checked=true;row.amount.disabled=false;row.amount.value=String(Number(group.needed_lb));chosen=[row];}
              if(!chosen.length||method.value==='supplier_lot'&&!matches.length){ui.setText(status,matches.length>1?ui.t('More than one lot matches. Choose the lot from the list.','Coincide más de un lote. Elige el lote de la lista.'):method.value==='supplier_lot'?ui.t("That supplier lot isn't in stock in FL — if it's old stock already here on Oct 21, choose the OPENING STOCK lot",'Ese lote del proveedor no tiene existencias en FL — si son existencias antiguas que ya estaban aquí el 21 de octubre, elija el lote de EXISTENCIAS INICIALES'):ui.t('Choose the physical lot from the list, then check its code.','Elige el lote físico de la lista y luego verifica su código.'));return null;}
            }
            if(!chosen.length){ui.setText(status,ui.t('Select the physical lot(s) you are using. FIFO has not confirmed a lot.','Selecciona los lotes físicos que usas. FIFO no ha confirmado ningún lote.'));return null;}
            if(chosen.some(row=>!Number.isFinite(Number(row.amount.value))||Number(row.amount.value)<=0)){ui.setText(status,ui.t('Enter positive lb for each chosen lot.','Escribe lb positivas para cada lote elegido.'));return null;}
            const total=chosen.reduce((sum,row)=>sum+Number(row.amount.value),0);
            if(Math.abs(total-Number(group.needed_lb))>0.0001){ui.setText(status,ui.t('Chosen lots must total '+ui.formatNumber(group.needed_lb)+' lb; currently '+ui.formatNumber(total)+' lb.','Los lotes elegidos deben sumar '+ui.formatNumber(group.needed_lb)+' lb; ahora suman '+ui.formatNumber(total)+' lb.'));return null;}
            return {allocations:chosen.map(row=>({ingredient_product_id:group.product_id,lot_id:row.lot.lot_id,quantity_lb:Number(row.amount.value)})),confirmations:chosen.map(row=>{
              if(state.typedDirty&&typed&&chosen.length===1)return {lot_id:row.lot.lot_id,method:method.value,value:typed,selected:row.explicit===true};
              if(row.previous)return row.previous;
              return {lot_id:row.lot.lot_id,method:row.lot.opening_stock||row.lot.supplier_lot_code?'supplier_lot':'full_code',value:row.lot.opening_stock?'OPENING STOCK':row.lot.supplier_lot_code||row.lot.lot_code,selected:true};
            })};
          };
          return state;
        });
        // An API FIFO plan is never evidence that the worker chose a physical lot.
        if(groups.some(state=>!state.rows.some(row=>row.selected.checked)))evidenceDirty=true;
        const check=ui.el('button',ui.t('Check confirmations → new draft','Verificar confirmaciones → nuevo borrador'),card);check.type='button';check.onclick=()=>{
          if(busy||finished||uploading)return;const choices=groups.map(group=>group.read());if(choices.some(choice=>!choice))return;
          payload.ingredient_lot_allocations=choices.flatMap(choice=>choice.allocations);
          const productIds=new Set(ingredientGroups.map(group=>group.product_id));
          if(payload.ingredient_lots)payload.ingredient_lots=payload.ingredient_lots.filter(value=>!productIds.has(value.ingredient_product_id));
          if(path==='/pack/prepare'&&productIds.has(payload.source_product_id))delete payload.lot_allocations;
          confirmations=choices.flatMap(choice=>choice.confirmations);prepare();
        };
      }else lots.forEach(lot=>{const field=ui.el('fieldset',undefined,card);field.className='lot-confirmation-row';field.dataset.productId=lot.product_id;
        ui.el('legend',ui.t('Confirm physical lot: '+(lot.product_name?lot.product_name+' · ':'')+lot.lot_code,'Confirma lote físico: '+(lot.product_name?lot.product_name+' · ':'')+lot.lot_code),field);
        const controls=ui.el('div',undefined,field);controls.className='lot-confirmation-controls';
        const label=ui.el('label',ui.t('Confirmation method','Método de confirmación'),controls),method=ui.el('select',undefined,label);
        [['last4',ui.t('Last 4 characters (include hyphen)','Últimos 4 caracteres (incluye guion)')],['full_code',ui.t('Full code','Código completo')],['pallet',ui.t('Pallet — full code, moved to production','Palé — código completo, movido a producción')]].forEach(([v,t])=>{const o=ui.el('option',t,method);o.value=v;});
        const label2=ui.el('label',ui.t('Type the code from the lot label','Escribe el código de la etiqueta'),controls),value=ui.el('input',undefined,label2);value.autocomplete='off';const prev=confirmations.find(c=>c.lot_id===lot.lot_id);if(prev){method.value=prev.method;value.value=prev.value;}
        const choices=ui.el('div',undefined,field);choices.className='lot-confirmation-choices';
        const input={lot,method,value,choices,read:()=>({lot_id:lot.lot_id,method:method.value,value:value.value})};inputs.push(input);
        for(const control of [method,value])control.oninput=()=>{choices.replaceChildren();evidenceDirty=true;record.disabled=true;};
      });
      if(!stockGroups&&lots.size){const check=ui.el('button',ui.t('Check confirmations → new draft','Verificar confirmaciones → nuevo borrador'),card);check.type='button';check.onclick=async()=>{
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
      if(allowPhoto){
      let photoType=null;if(path==='/receive/prepare'){const kindLabel=ui.el('label',ui.t('Photo type','Tipo de foto'),card);photoType=ui.el('select',undefined,kindLabel);photoType.name='draft_photo_kind';for(const [value,en,es] of [['lot_tag','Label / lot tag','Etiqueta / etiqueta de lote'],['bol','BOL','BOL']]){const option=ui.el('option',ui.t(en,es),photoType);option.value=value;}photoType.value=photoKind;}
      const uploadLabel=ui.el('label',ui.t('Attach photo','Adjuntar foto'),card),photo=ui.el('input',undefined,uploadLabel);photo.type='file';photo.accept='image/jpeg,image/png,image/webp';photo.setAttribute('capture','environment');
      const uploadStatus=ui.el('p',attachment?ui.t('Photo attached','Foto adjunta'):'',card);
      async function uploadPhoto(file,kind,{retype=false}={}){if(!file||busy||finished||uploading)return;const previousKind=photoKind;uploading=true;photo.disabled=true;if(photoType)photoType.disabled=true;onBusy(true);record.disabled=true;ui.setText(uploadStatus,ui.t('Uploading photo…','Subiendo foto…'));try{const uploaded=await ui.upload(file,kind);if(finished)return;attachment=uploaded;photoFile=file;photoKind=kind;onAttachment(attachment,photoFile);ui.setText(uploadStatus,ui.t('Photo attached','Foto adjunta'));}catch(error){if(!finished){if(retype){photoKind=previousKind;photoType.value=previousKind;ui.setText(uploadStatus,ui.t('Photo type was not changed. The attached photo keeps its previous type.','No se cambió el tipo de foto. La foto adjunta conserva su tipo anterior.'));}else ui.setText(uploadStatus,attachment?ui.t('Photo attached','Foto adjunta'):ui.t('Photo was not attached. Try again.','La foto no se adjuntó. Intenta de nuevo.'));failure(error);}}finally{uploading=false;photo.disabled=false;if(photoType)photoType.disabled=false;if(!finished)onBusy(false);record.disabled=busy||finished||evidenceDirty||prepared.can_commit!==true;}}
      if(photoType)photoType.onchange=()=>{if(busy||finished||uploading)return;if(attachment){if(photoFile)return uploadPhoto(photoFile,photoType.value,{retype:true});photoType.value=photoKind;ui.setText(uploadStatus,ui.t('Choose the photo again to change its type.','Elige la foto de nuevo para cambiar su tipo.'));return;}photoKind=photoType.value;};
      photo.onchange=()=>uploadPhoto(photo.files?.[0],photoType?.value||photoKind);
      }
      const actions=ui.el('div',undefined,card);actions.className='record-bar';const record=ui.el('button',ui.t('Record','Registrar'),actions);record.type='button';record.className='primary';record.disabled=evidenceDirty||prepared.can_commit!==true;
      record.onclick=async()=>{if(busy||finished||uploading||record.disabled)return;busy=true;onBusy(true);record.disabled=true;const body={payload_hash:prepared.payload_hash,acknowledged_warnings:ack.filter(([input])=>input.checked).map(([,code])=>code)};
        if(attachment){if(path==='/receive/prepare'||photoKind==='bol')body.attachment_ids=[attachment.attachment_id];else body.attachment_id=attachment.attachment_id;}
        try{const result=await ui.request('/tickets/'+encodeURIComponent(prepared.ticket)+'/commit',body);if(finished)return;
          if(!ui.recorded(result)||(receiptPrefix&&!new RegExp('^'+receiptPrefix+'-\\d{6}-\\d{3,}$').test(result.receipt_number))){ui.notice(ui.t('NOT recorded — '+ui.errorText(result,'en'),'NO registrado — '+ui.errorText(result,'es')));if(result.held||result.status==='awaiting_approval'){finished=true;const a=ui.el('a',ui.t('Open owner approval queue','Ver cola de aprobación'),card);a.href=FLReceipts.page('exceptions.html');}return;}
          finished=true;document.getElementById('notice').hidden=true;card.replaceChildren();const p=ui.el('p',ui.t('Recorded — receipt ','Registrado — recibo '),card);p.className='good';const a=ui.el('a',result.receipt_number,p);a.href=FLReceipts.link(result.receipt_number);const substitutions=ui.el('div',undefined,card);substitutions.innerHTML=ui.autoSubstitutionsHtml(result.auto_substitutions);onRecorded(result);
        }catch(error){failure(error);}finally{busy=false;onBusy(false);record.disabled=finished||evidenceDirty||prepared.can_commit!==true;}};
    }
    prepare();return {invalidate};
  };
})(window);
