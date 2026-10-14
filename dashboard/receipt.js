(() => {
  const ui=FLLedger;ui.bind();let receipt;
  const content=document.getElementById('content');
  function render(){if(!receipt)return;content.replaceChildren();const card=ui.el('section',undefined,content);card.className='panel';ui.el('h2',receipt.receipt_number,card);const state=ui.receiptState(receipt);
    const headline={recorded:ui.t('Recorded','Registrado'),voided:ui.t('VOIDED — recorded, then voided. It no longer counts in stock.','ANULADO — se registró y luego se anuló. Ya no cuenta en el inventario.'),partly_voided:ui.t('PARTLY VOIDED — some entries on this receipt were voided.','ANULADO EN PARTE — algunas entradas de este recibo se anularon.')}[state.state]||ui.t('NOT recorded — ','NO registrado — ')+receipt.status;
    const status=ui.el('p',headline,card);status.className=state.tone;status.dataset.state=state.state;
    if(state.voidReceipt){const p=ui.el('p',ui.t('Voided by receipt ','Anulado por el recibo '),card);const a=ui.el('a',state.voidReceipt,p);a.href=FLReceipts.link(state.voidReceipt);}
    const data={action:receipt.action,status:receipt.status,actor:receipt.actor,happened_at:receipt.happened_at,entered_at:receipt.entered_at,transactions:receipt.transactions,lots:receipt.lots,shipment:receipt.shipment,order:receipt.order,warnings:receipt.warnings,acknowledged:receipt.acknowledged,draft:receipt.draft,response:receipt.response};
    (receipt.warnings||[]).forEach(w=>{const p=ui.el('p',ui.text(w,ui.lang),card);p.className='notice';});
    const view=ui.el('div',undefined,card);view.innerHTML=ui.draftHtml(data);
    const link=ui.el('a',ui.t('Open exceptions','Ver excepciones'),card);link.dataset.flWriteUi='';link.href=FLReceipts.page('exceptions.html',{receipt:receipt.receipt_number});
  }
  async function load(){const number=new URLSearchParams(location.search).get('number');if(!number){ui.notice(ui.t('Enter a Receipt # above.','Ingresa un número de recibo arriba.'),'notice');return;}try{receipt=await ui.request('/receipts/'+encodeURIComponent(number),undefined,{read:true});render();}catch(error){ui.failure(error);}}
  window.addEventListener('ledger-language',render);window.addEventListener('ledger-auth',load);load();
})();
