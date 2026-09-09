(() => {
  const params=new URLSearchParams(location.search),day=document.getElementById('history-day'),kind=document.getElementById('history-type'),status=document.getElementById('history-status'),results=document.getElementById('history-results');
  day.value=/^\d{4}-\d{2}-\d{2}$/.test(params.get('day')||'')?params.get('day'):new Date().toLocaleDateString('en-CA',{timeZone:'America/New_York'});kind.value=params.get('type')||'';
  let generation=0;
  const names={make:'Production',pack:'Packing',receive:'Receiving',ship:'Shipping',adjust:'Adjustment'};
  function text(tag,value,parent){const el=document.createElement(tag);el.textContent=value;parent.append(el);return el;}
  async function load(){
    if(!day.checkValidity())return;
    const request=++generation;status.textContent='Loading history…';results.replaceChildren();
    const url=new URL(location.href);url.searchParams.set('day',day.value);if(kind.value)url.searchParams.set('type',kind.value);else url.searchParams.delete('type');history.replaceState({},'',url);
    try{
      const q=new URLSearchParams({since:day.value,until:day.value,limit:1000});if(kind.value)q.set('transaction_type',kind.value);
      const response=await FL.fetchWithTimeout('https://fastapi-production-b73a.up.railway.app/transactions/history?'+q,{headers:{'X-API-Key':'dashboard-key-2026'}});if(!response.ok)throw Error('HTTP '+response.status);const data=await response.json();if(request!==generation)return;
      const records=data.transactions||[],selected=new URL(location.href).searchParams.get('transaction');
      status.textContent=records.length+' transactions for '+FLDesign.date(day.value)+(records.length>=1000?'. Limit reached: narrow by type; older records may be omitted.':'.');
      records.forEach(tx=>{const article=document.createElement('article');article.className='recent-entry-card';article.id='transaction-'+tx.id;const details=document.createElement('details');details.className='record-details';details.open=String(tx.id)===selected;const title=document.createElement('summary');title.textContent='TX-'+tx.id+' · '+(names[tx.type]||tx.type)+' · '+(tx.status||'Status unavailable');details.append(title);const p=text('p','Occurred: '+FLDesign.time(tx.occurred_at || (tx.date+' '+tx.time))+' · Entered: '+FLDesign.time(tx.created_at),details);
        const link=document.createElement('a');link.className='record-link';const target=new URL(location.href);target.searchParams.set('transaction',tx.id);link.href=target.pathname+target.search;link.textContent='Link to TX-'+tx.id;details.append(link);
        const list=document.createElement('ul');(tx.lines||[]).forEach(line=>{const li=text('li',(line.product_name||'Unknown product')+' · '+Number(line.quantity_lb).toLocaleString('en-US')+(['ingredient','batch','finished'].includes(line.product_type)?' lb':' stock units (unit unavailable)')+(line.lot_code?' · Lot '+line.lot_code:''),list);if(line.lot_code){const trace=document.createElement('a');trace.className='record-link';trace.href=FLDesign.traceURL(line.lot_code,line.product_id);trace.textContent='Trace lot →';li.append(trace);}});details.append(list);
        if(tx.notes)text('p',tx.notes,details);if(tx.correction_chain?.length){text('h3','Corrections',details);tx.correction_chain.forEach(c=>text('p',(c.event_type||'Correction')+' · '+FLDesign.time(c.created_at)+(c.reason?' · '+c.reason:''),details));}article.append(details);results.append(article);
      });
      if(selected){const target=document.getElementById('transaction-'+selected);if(target){target.tabIndex=-1;target.focus();target.scrollIntoView({block:'start'});}else status.textContent+=' Requested TX-'+selected+' was not found in this date/type scope. Change the date or type to locate it.';}
    }catch(e){if(request===generation)status.textContent='History unavailable. Select Load history to retry. '+e.message;}
  }
  document.getElementById('history-form').addEventListener('submit',e=>{e.preventDefault();const u=new URL(location.href);u.searchParams.delete('transaction');history.replaceState({},'',u);load();});load();
})();
