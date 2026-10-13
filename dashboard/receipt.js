(() => {
  const ui=FLLedger;ui.bind();let receipt;
  const content=document.getElementById('content');
  function render(){if(!receipt)return;content.replaceChildren();const card=ui.el('section',undefined,content);card.className='panel';ui.el('h2',receipt.receipt_number,card);const posted=receipt.status==='committed';const status=ui.el('p',posted?ui.t('Recorded','Registrado'):ui.t('NOT recorded — ','NO registrado — ')+receipt.status,card);status.className=posted?'good':'bad';
    const data={action:receipt.action,status:receipt.status,actor:receipt.actor,happened_at:receipt.happened_at,entered_at:receipt.entered_at,transactions:receipt.transactions,lots:receipt.lots,shipment:receipt.shipment,order:receipt.order,warnings:receipt.warnings,acknowledged:receipt.acknowledged,draft:receipt.draft,response:receipt.response};
    (receipt.warnings||[]).forEach(w=>{const p=ui.el('p',ui.text(w,ui.lang),card);p.className='notice';});
    const view=ui.el('div',undefined,card);view.innerHTML=ui.draftHtml(data);
    const link=ui.el('a',ui.t('Open exceptions','Ver excepciones'),card);link.href=FLReceipts.page('exceptions.html',{receipt:receipt.receipt_number});
  }
  async function load(){const number=new URLSearchParams(location.search).get('number');if(!number){ui.notice(ui.t('Enter a Receipt # above.','Ingresa un número de recibo arriba.'),'notice');return;}try{receipt=await ui.request('/receipts/'+encodeURIComponent(number),undefined,{read:true});render();}catch(error){ui.failure(error);}}
  window.addEventListener('ledger-language',render);window.addEventListener('ledger-auth',load);load();
})();
