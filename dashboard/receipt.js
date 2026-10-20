(() => {
  const ui=FLLedger;ui.bind();let receipt,generation=0,ticket=null,busy=false,missing=false;const voidNotes=new Map();
  const content=document.getElementById('content');
  const resolution=document.getElementById('resolution');
  function render(){if(!receipt)return;content.replaceChildren();const card=ui.el('section',undefined,content);card.className='panel';ui.el('h2',receipt.receipt_number,card);const state=ui.receiptState(receipt);
    const headline={recorded:ui.t('Recorded','Registrado'),voided:ui.t('VOIDED — recorded, then voided. It no longer counts in stock.','ANULADO — se registró y luego se anuló. Ya no cuenta en el inventario.'),partly_voided:ui.t('PARTLY VOIDED — some entries on this receipt were voided.','ANULADO EN PARTE — algunas entradas de este recibo se anularon.')}[state.state]||ui.t('NOT recorded — ','NO registrado — ')+receipt.status;
    const status=ui.el('p',headline,card);status.className=state.tone;status.dataset.state=state.state;
    if(state.voidReceipt){const p=ui.el('p',ui.t('Voided by receipt ','Anulado por el recibo '),card);const a=ui.el('a',state.voidReceipt,p);a.href=FLReceipts.link(state.voidReceipt);}
    const data={action:receipt.action,status:receipt.status,actor:receipt.actor,happened_at:receipt.happened_at,entered_at:receipt.entered_at,supplier_lot_code:receipt.supplier_lot_code,bol_reference:receipt.bol_reference,transactions:receipt.transactions,lots:receipt.lots,shipment:receipt.shipment,order:receipt.order,warnings:receipt.warnings,acknowledged:receipt.acknowledged,draft:receipt.draft,response:receipt.response};
    (receipt.warnings||[]).forEach(w=>{const p=ui.el('p',ui.text(w,ui.lang),card);p.className='notice';});
    const view=ui.el('div',undefined,card);view.innerHTML=ui.draftHtml(ui.receiptData(data));
    const photos=receipt.attachments||receipt.response?.attachments||[];
    if(photos.length){const section=ui.el('section',undefined,card);section.className='receipt-photos';ui.el('h3',ui.t('Receipt photos','Fotos del recibo'),section);
      const names={lot_tag:['Lot tag','Etiqueta de lote'],bol:['BOL','BOL'],truck_photo:['Truck','Camión'],signed_packing_slip:['Signed packing slip','Nota de empaque firmada'],correction_photo:['Correction','Corrección']};
      for(const photo of photos){const id=photo.id||photo.attachment_id;if(!Number.isSafeInteger(id))continue;const name=ui.t(...(names[photo.kind]||['Attached photo','Foto adjunta']));const button=ui.el('button',ui.t('View ','Ver ')+name,section);button.type='button';button.onclick=async()=>{button.disabled=true;try{const blob=await ui.request('/attachments/'+id+'/file',undefined,{read:true,binary:true});const url=URL.createObjectURL(blob),dialog=ui.el('dialog',undefined,document.body),image=ui.el('img',undefined,dialog);image.src=url;image.alt=String(name);image.style.maxWidth='min(85vw,800px)';image.style.maxHeight='75vh';const close=ui.el('button',ui.t('Close','Cerrar'),dialog);close.onclick=()=>dialog.close();dialog.addEventListener('close',()=>{URL.revokeObjectURL(url);dialog.remove();});dialog.showModal();}catch(error){ui.failure(error);}finally{button.disabled=false;}};}
    }
    if(ui.session.token)for(const txn of ui.voidChoices(receipt,FLDashboard.writesEnabled)){
      const controls=ui.el('div',undefined,card);controls.dataset.flWriteUi='';
      ui.el('p',ui.t('Void reason: data-entry error. This removes entry #','Motivo: error de captura. Esto anula la entrada #')+txn.id+ui.t(' from stock totals.',' de los totales de inventario.'),controls);
      const noteLabel=ui.el('label',ui.t('Note (optional)','Nota (opcional)'),controls),note=ui.el('textarea',undefined,noteLabel);note.name='note';note.dataset.voidNote=txn.id;note.maxLength=2000;note.value=voidNotes.get(txn.id)||'';note.disabled=busy;note.oninput=()=>{voidNotes.set(txn.id,note.value);ticket?.invalidate();ticket=null;resolution.replaceChildren();};
      const button=ui.el('button',ui.t('Prepare void for entry #','Preparar anulación de entrada #')+txn.id,controls);
      button.type='button';button.disabled=busy;button.dataset.void=txn.id;
      button.onclick=()=>{
        if(busy||!ui.session.token||!FLDashboard.writesEnabled)return;
        ticket?.invalidate();resolution.replaceChildren();
        const request=ui.voidRequest(txn.id,note.value);
        ticket=ui.ticketCard(resolution,request.path,request.body,{receiptPrefix:'VD',allowPhoto:false,
          onBusy(value){busy=value;document.getElementById('sign-out').disabled=value;render();},
          onRecorded(){load();}});
        resolution.scrollIntoView({block:'start',behavior:'smooth'});
      };
    }
    const link=ui.el('a',ui.t('Open exceptions','Ver excepciones'),card);link.dataset.flWriteUi='';link.href=FLReceipts.page('exceptions.html',{receipt:receipt.receipt_number});
  }
  async function load({background=false}={}){
    const current=++generation,number=new URLSearchParams(location.search).get('number');
    if(!number){ui.notice(ui.t('Enter a Receipt # above.','Ingresa un número de recibo arriba.'),'notice');return;}
    try{const data=await ui.request('/receipts/'+encodeURIComponent(number),undefined,{read:true,background});
      if(current!==generation)return;receipt=data;missing=false;document.getElementById('notice').hidden=true;render();
    }catch(error){if(current!==generation)return;receipt=null;content.replaceChildren();
      missing=error.status===404&&error.result?.detail?.error_code==='RECEIPT_NOT_FOUND';
      if(missing)ui.notice(ui.t('No receipt with this number','No hay recibo con este número'),'neutral');
      else if(error.accessUnavailable){ui.notice(error.message,'neutral');FLDashboard.renderAccess(document.getElementById('notice'),load,ui.lang);}
      else ui.notice(error.message,'neutral');
    }
  }
  function authChanged(){ticket?.invalidate();ticket=null;resolution.replaceChildren();receipt=null;content.replaceChildren();load();}
  window.addEventListener('ledger-language',()=>{if(!receipt&&!missing)load();render();ui.relabel();ui.translate();if(missing)ui.notice(ui.t('No receipt with this number','No hay recibo con este número'),'neutral');});
  window.addEventListener('ledger-auth',authChanged);window.addEventListener('ledger-signout',authChanged);load({background:true});
})();
