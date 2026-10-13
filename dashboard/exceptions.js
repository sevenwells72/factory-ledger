(() => {
  const ui=FLLedger;ui.bind();let rows=[],selected;
  const content=document.getElementById('content'),resolution=document.getElementById('resolution');
  const modes={SHORTAGE:['counted','missing_movement','voided'],NEGATIVE_BALANCE:['counted','missing_movement','voided'],UNIDENTIFIED_LOT:['identified','written_off'],LARGE_CORRECTION:['approved','declined'],KOSHER_ATTESTATION:['approved','declined'],BOL_PHOTO_MISSING:['photo_attached','waived'],SHIPMENT_PROOF_MISSING:['photo_attached','waived'],LATE_ENTRY:['acknowledged'],PRE_MAKE_ADJUST:['acknowledged']};
  const labels={counted:['Counted adjustment','Ajuste contado'],missing_movement:['Link missing movement receipt','Vincular recibo de movimiento faltante'],voided:['Posting was voided','Registro anulado'],identified:['Identify supplier lot','Identificar lote de proveedor'],written_off:['Owner write-off','Baja por propietario'],approved:['Owner approve','Aprobar como propietario'],declined:['Owner reject','Rechazar como propietario'],photo_attached:['Attach BOL photo','Adjuntar foto BOL'],waived:['Owner waive','Eximir como propietario'],acknowledged:['Acknowledge','Reconocer'],resolved:['Resolve','Resolver']};
  function render(){content.replaceChildren();const refresh=ui.el('button',ui.t('Refresh exceptions','Actualizar excepciones'),content);refresh.onclick=load;
    if(!rows.length)ui.el('p',ui.t('No open exceptions.','No hay excepciones abiertas.'),content);
    for(const [kind,items] of ui.groups(rows)){const section=ui.el('section',undefined,content);section.className='panel';ui.el('h2',kind+' · '+items.length,section);
      for(const item of items){const row=ui.el('article',undefined,section);row.className='exception-row'+(item.overdue?' overdue':'');
        const desc=ui.el('div',undefined,row),number=item.receipt_number||item.ticket_receipt_number;
        const a=ui.el('a',number||ui.t('Receipt unavailable','Recibo no disponible'),desc);if(number)a.href=FLReceipts.link(number);a.className='record-id';ui.el('p',[item.product?.name,item.lot?.lot_code].filter(Boolean).join(' · '),desc);
        ui.el('div',ui.t('Owner: ','Responsable: ')+(item.owner?.name||ui.t('Unassigned','Sin asignar')),row);
        ui.el('div',(item.overdue?ui.t('Overdue · ','Vencido · '):'')+(item.due_at?new Date(item.due_at).toLocaleString(ui.lang==='es'?'es-US':'en-US',{timeZone:'America/New_York'}):ui.t('No deadline','Sin fecha límite')),row);
        const button=ui.el('button',ui.t('Resolve','Resolver'),row);button.onclick=()=>open(item.id);
      }
    }
  }
  async function load(){try{const result=await ui.request('/exceptions?status=open&limit=1000');rows=result.exceptions||[];const number=new URLSearchParams(location.search).get('receipt');if(number)rows=rows.filter(r=>(r.receipt_number||r.ticket_receipt_number)===number);render();if(result.count>=1000)ui.notice(ui.t('First 1,000 items shown.','Se muestran los primeros 1.000 elementos.'),'notice');}catch(error){ui.failure(error);}}
  async function open(id){try{selected=await ui.request('/exceptions/'+id);renderForm();resolution.scrollIntoView({block:'start'});}catch(error){ui.failure(error);}}
  function renderForm(){resolution.replaceChildren();const item=selected;if(!item)return;
    const panel=ui.el('section',undefined,resolution);panel.className='panel';ui.el('h2',ui.t('Resolve ','Resolver ')+item.kind+' #'+item.id,panel);const detail=ui.el('div',undefined,panel);detail.innerHTML=ui.draftHtml(item.detail);
    const form=ui.el('form',undefined,panel);form.className='fields';
    const label=ui.el('label',ui.t('Resolution','Resolución'),form),kind=ui.el('select',undefined,label);(modes[item.kind]||['resolved','waived']).forEach(k=>{const o=ui.el('option',ui.t(...labels[k]),kind);o.value=k;});
    const fields=ui.el('div',undefined,form);fields.className='wide fields';
    function field(name,en,es,type='text'){const l=ui.el('label',ui.t(en,es),fields),i=ui.el(type==='textarea'?'textarea':'input',undefined,l);i.name=name;if(type!=='textarea')i.type=type;return i;}
    let note,count,receipt,photo,code,reason;
    function update(){fields.replaceChildren();note=field('note','Note','Nota','textarea');note.required=true;count=receipt=photo=code=reason=null;
      if(kind.value==='counted'){count=field('counted_lb','Physically counted lb','Libras contadas físicamente','number');count.step='any';count.min='0';count.required=true;}
      if(kind.value==='missing_movement'){receipt=field('receipt_number','Evidence Receipt #','Recibo de evidencia #');receipt.required=true;}
      if(kind.value==='identified'){code=field('supplier_lot_code','Supplier lot code','Código de lote del proveedor');code.required=true;const l=ui.el('label',ui.t('Reason','Motivo'),fields);reason=ui.el('select',undefined,l);[['data_entry_error','Data-entry error / Error de captura'],['wrong_lot','Wrong lot used / Lote equivocado']].forEach(([v,t])=>{const o=ui.el('option',t,reason);o.value=v;});}
      if(['counted','photo_attached'].includes(kind.value)){photo=field('photo','Photo','Foto','file');photo.accept='image/jpeg,image/png,image/webp';if(kind.value==='photo_attached')photo.required=true;}
      preview.replaceChildren();
    }
    const button=ui.el('button',ui.t('Review resolution','Revisar resolución'),form);button.type='submit';button.className='primary';const preview=ui.el('div',undefined,panel);kind.onchange=update;update();form.oninput=()=>preview.replaceChildren();
    form.onsubmit=async event=>{event.preventDefault();button.disabled=true;preview.replaceChildren();try{
      if(kind.value==='identified'){ui.ticketCard(preview,'/lots/'+item.lot.id+'/supplier-lot/prepare',{supplier_lot_code:code.value,reason_code:reason.value,note:note.value,client_source:'dashboard'},{onRecorded:load});return;}
      const body={resolution_kind:kind.value,note:note.value};if(count)body.counted_lb=Number(count.value);if(receipt)body.receipt_number=receipt.value;
      if(photo?.files[0])body.attachment_id=(await ui.upload(photo.files[0],kind.value==='photo_attached'?'bol':'correction_photo')).attachment_id;
      // A3b exposes explicit resolve/approve/reject operations, not a ticket prepare.
      // Show the latest FL snapshot and exact submitted evidence before the decision.
      const current=await ui.request('/exceptions/'+item.id);ui.el('h3',ui.t('Review — no decision recorded yet','Revisa — aún no se ha registrado la decisión'),preview);const data=ui.el('div',undefined,preview);data.innerHTML=ui.draftHtml(current.detail)+ui.draftHtml(body);
      const record=ui.el('button',ui.t('Confirm resolution','Confirmar resolución'),preview);record.type='button';record.className='primary';record.onclick=async()=>{record.disabled=true;const action=body.resolution_kind==='approved'?'approve':body.resolution_kind==='declined'?'reject':'resolve';const send=action==='approve'?{note:body.note}:body;
        try{const result=await ui.request('/exceptions/'+item.id+'/'+action,send);
          if(ui.recorded(result)){ui.notice(ui.t('Recorded — receipt ','Registrado — recibo ')+result.receipt_number,'good');}
          else if(result.rejected||result.status==='resolved'||result.status==='waived'||result.evidence){ui.notice(ui.t('FL returned exception status: ','FL devolvió estado de excepción: ')+(result.ticket_status||result.status)+ui.t('. No new receipt returned. Check History before retrying a counted adjustment.','. Sin recibo nuevo. Revisa el historial antes de reintentar un ajuste contado.'),'notice');}
          else ui.failure(Error(ui.errorText(result,ui.lang)));await load();preview.replaceChildren();
        }catch(error){ui.failure(error);record.disabled=false;}};
    }catch(error){ui.failure(error);}finally{button.disabled=false;}};
  }
  window.addEventListener('ledger-auth',load);window.addEventListener('ledger-language',()=>{render();renderForm();});window.addEventListener('ledger-signout',()=>{rows=[];selected=null;content.replaceChildren();resolution.replaceChildren();});ui.ready.then(()=>{if(ui.session.actor)load();}).catch(()=>{});
})();
