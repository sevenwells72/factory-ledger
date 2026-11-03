/* Formula-driven Make (docs/production/formula-driven-entry.md; make-entry simplification PR-1).
   FL's draft lists each recipe ingredient with ONE suggested lot. The worker taps the lot
   they really used (no typed code), opens "Change lot or pounds" only when something
   differed, and may add a second lot with its pounds. A suggestion is never confirmed
   until it is tapped. The browser only collects choices; FL re-checks and posts on Record. */
(function(root){
  const ui=root.FLLedger;
  // Spec D3 — must match formula_entry.REASONS (tests/test_formula_entry.py checks it).
  const reasons=[['out_of_stock','Out of stock','Sin existencias',false],['not_used','Not used','No se usó',false],['substituted','Substituted','Sustituido',false],['quality_problem','Quality problem','Problema de calidad',false],['owner_instruction','Owner instruction','Instrucción del dueño',false],['other','Other','Otro',true]];
  ui.makeReasons=reasons;
  ui.makeReasonSelect=function(host,name,value=''){const label=ui.el('label',ui.t('Reason','Motivo'),host),select=ui.el('select',undefined,label);select.name=name;ui.el('option',ui.t('Choose reason','Elige motivo'),select).value='';reasons.forEach(([code,en,es])=>{const o=ui.el('option',ui.t(en,es),select);o.value=code;});select.value=value||'';return select;};
  const round=n=>Math.round(n*100)/100;
  // A typed amount is kept exactly as typed. Negative or unreadable pounds are invalid
  // (never treated as 0 / Not used); only the "Not used (0 lb)" button marks an ingredient unused.
  function pounds(input){const raw=String(input.value??'').trim();if(raw==='')return {value:0,invalid:input.validity?.badInput===true};const n=Number(raw);return Number.isFinite(n)&&n>=0?{value:n,invalid:false}:{value:0,invalid:true};}
  const plural=(n,one,many)=>n===1?one:many;
  // Supplier lot as FL stores it: pooled stock never names one supplier lot.
  function supplierText(c){const s=c?.supplier_lot;if(!s)return null;if(s.kind==='pooled')return ui.t('pooled — no single supplier lot','mezclado — sin un solo lote de proveedor');if(s.kind==='no_lot')return ui.t('no lot on label','sin lote en la etiqueta');if(s.kind==='not_recorded')return ui.t('supplier lot not recorded','lote de proveedor no registrado');return ui.t('supplier lot '+s.code,'lote proveedor '+s.code);}
  function lotText(c,code){const parts=[c?.lot_code||code];const s=supplierText(c);if(s)parts.push(s);if(c?.received_on)parts.push(ui.t('received '+c.received_on,'recibido '+c.received_on));const book=c?.book_lb;if(book!=null)parts.push(book>0?ui.t('on books '+ui.formatNumber(book)+' lb','en sistema '+ui.formatNumber(book)+' lb'):ui.t('on books '+ui.formatNumber(book)+' lb (none available)','en sistema '+ui.formatNumber(book)+' lb (nada disponible)'));else if(c?.available_lb!=null)parts.push(ui.formatNumber(c.available_lb)+' lb');return parts.map(String).join(' · ');}
  ui.formulaEntry=function(card,block,{confirmations=[],dirty=()=>{},update=()=>{},changed=()=>{},checkHost=card}={}){
    const pansEn=block.pans+plural(block.pans,' pan',' pans'),pansEs=block.pans+plural(block.pans,' bandeja',' bandejas');
    const intro=ui.el('p',ui.t(pansEn+' × '+ui.formatNumber(block.pan_lb)+' lb = '+ui.formatNumber(block.output_lb)+' lb. Tap the lot you used for each ingredient. Open "Change lot or pounds" only if something was different.',pansEs+' × '+ui.formatNumber(block.pan_lb)+' lb = '+ui.formatNumber(block.output_lb)+' lb. Toca el lote que usaste para cada ingrediente. Abre "Cambiar lote o libras" solo si algo fue diferente.'),card);intro.className='muted';intro.dataset.formulaSummary='';
    const previous=new Map(confirmations.map(c=>[c.lot_id,c])),cards=[];let lastEdited=null;
    const byRole=role=>cards.find(c=>c.ing.swap_role===role);
    function swapLb(){const from=byRole('from'),to=byRole('to');if(!block.swap||!from||!to)return 0;return Math.max(0,round(Math.min(from.ing.standard_lb-from.sum(),to.sum()-to.ing.standard_lb)));}
    let check=null;
    function refreshAll(){cards.forEach(c=>c.refresh());if(check)check.disabled=invalidCount()>0;changed();}
    for(const ing of block.ingredients){
      const box=ui.el('fieldset',undefined,card);box.className='formula-ingredient';box.dataset.productId=String(ing.product_id);
      const touch=()=>{lastEdited=box;dirty();};
      const legend=ui.el('legend',ing.product_name||'—',box);
      if(ing.substituted_for){const p=ui.el('p',ui.t('Swapped in for a recipe ingredient.','Sustituye a un ingrediente de la receta.'),box);p.className='muted';}
      if(ing.swap_role==='to'){const p=ui.el('p',ui.t('Rolled oats can replace missing oat bran pound for pound; the swapped pounds count toward the oat bran recipe.','La avena en hojuelas puede reemplazar el salvado de avena que falta libra por libra; esas libras cuentan para la receta del salvado.'),box);p.className='muted';}
      const std=ui.el('p',ui.t('Recipe: '+ui.formatNumber(ing.standard_lb)+' lb','Receta: '+ui.formatNumber(ing.standard_lb)+' lb'),box);std.dataset.standard=String(ing.standard_lb);std.className='muted';
      const total=ui.el('p','',box);total.dataset.total='';total.setAttribute('role','status');
      const picks=ui.el('div',undefined,box);picks.className='formula-picks';
      // Everything that changes the plan lives behind one toggle (collapsed by default).
      const edit=ui.el('details',undefined,box);edit.className='formula-edit';ui.el('summary',ui.t('Change lot or pounds','Cambiar lote o libras'),edit);
      const rows=ui.el('div',undefined,edit);rows.className='formula-bags';const bags=[];
      const reasonArea=ui.el('div',undefined,box);reasonArea.className='formula-reason';
      const reason=ui.makeReasonSelect(reasonArea,'ingredient_reason',ing.reason_code),noteLabel=ui.el('label',ui.t('Note (required for Other)','Nota (obligatoria para Otro)'),reasonArea),note=ui.el('input',undefined,noteLabel);note.name='ingredient_note';note.value=ing.note||'';note.maxLength=2000;
      function sum(){return round(bags.reduce((n,b)=>n+pounds(b.lb).value,0));}
      const invalid=()=>bags.filter(b=>pounds(b.lb).invalid).length;
      // Oat bran + the oats that replace it are checked against the recipe bran;
      // the oats card is checked without those pounds (formula_entry.review).
      function refresh(){const used=sum(),swap=swapLb(),role=ing.swap_role,checked=role==='from'?round(used+swap):role==='to'?round(used-swap):used,swapped=role==='from'&&swap>0;
        const pct=ing.standard_lb>0?Math.round((checked-ing.standard_lb)/ing.standard_lb*1000)/10:0,over=ing.standard_lb>0?Math.abs(checked-ing.standard_lb)>ing.standard_lb*block.tolerance_pct/100+1e-9:checked>0.005;
        const diff=pct?' ('+(pct>0?'+':'')+pct+'%)':'';
        ui.setText(legend,swapped?ui.t((block.swap.from_name||'oat bran')+' → '+(block.swap.to_name||'rolled oats')+' (swap)',(block.swap.from_name||'salvado de avena')+' → '+(block.swap.to_name||'avena')+' (sustitución)'):(ing.product_name||'—'));
        const amount=swapped?ui.t(ui.formatNumber(used)+' lb + '+ui.formatNumber(swap)+' lb '+(block.swap.to_name||'rolled oats')+' = '+ui.formatNumber(checked)+' lb',ui.formatNumber(used)+' lb + '+ui.formatNumber(swap)+' lb '+(block.swap.to_name||'avena')+' = '+ui.formatNumber(checked)+' lb')
          :role==='to'&&swap>0?ui.t(ui.formatNumber(used)+' lb ('+ui.formatNumber(swap)+' lb replace oat bran)',ui.formatNumber(used)+' lb ('+ui.formatNumber(swap)+' lb reemplazan salvado)'):ui.formatNumber(used)+' lb';
        // "Not used" only when no lot rows are left (the Not used button, or FL's draft of it).
        const bad=invalid(),zero=!bags.length&&!swapped&&ing.standard_lb>0,noPounds=!bad&&bags.length>0&&used<=0&&!swapped&&ing.standard_lb>0;
        if(bad){ui.setText(total,ui.t('Pounds cannot be negative — fix the amount below. Nothing is recorded until it is fixed.','Las libras no pueden ser negativas — corrige la cantidad abajo. No se registra nada hasta corregirla.'));total.className='bad';total.dataset.over='false';total.dataset.invalid='true';reasonArea.hidden=true;edit.open=true;bags.forEach(b=>b.paint());return;}
        total.dataset.invalid='false';
        if(noPounds){ui.setText(total,ui.t('0 lb entered. Enter the pounds you used, or tap "Not used (0 lb)" if this ingredient was not used.','0 lb escritas. Escribe las libras que usaste, o toca "No se usó (0 lb)" si no usaste este ingrediente.'));total.className='notice';total.dataset.over='false';reasonArea.hidden=true;bags.forEach(b=>b.paint());return;}
        ui.setText(total,zero?ui.t('Not used — 0 lb. Choose a reason.','No se usó — 0 lb. Elige un motivo.'):over?ui.t('Using '+amount+diff+' — more than 10% off the recipe; the owner will review it.','Usando '+amount+diff+' — más de 10% diferente de la receta; el dueño lo revisará.'):ui.t('Using '+amount+diff,'Usando '+amount+diff));
        total.className=!zero&&!over?'muted':'notice';total.dataset.over=String(over&&!zero);
        reasonArea.hidden=!zero;if(zero)edit.open=true;bags.forEach(b=>b.paint());}
      const item={ing,bags,reason,note,box,edit,sum,refresh,invalid};cards.push(item);
      function addBag(bag){
        const row=ui.el('fieldset',undefined,rows);row.className='formula-bag';
        const lb=ui.el('input');lb.name='bag_lb';lb.type='number';lb.step='any';lb.min='0';lb.inputMode='decimal';lb.value=bag?.quantity_lb!=null?String(bag.quantity_lb):'';
        // The compact view: one big button per lot. A tap is the worker's confirmation.
        const pick=ui.el('div',undefined,picks);pick.className='formula-pick';
        const confirm=ui.el('button','',pick);confirm.type='button';confirm.className='lot-confirm';confirm.dataset.lotConfirm='';
        const state=ui.el('p','',pick);state.dataset.bagState='';
        const short=ui.el('p','',pick);short.className='notice';short.dataset.bagShort='';short.hidden=true;
        const error=ui.el('p','',pick);error.className='bad';error.dataset.bagError='';error.setAttribute('role','alert');error.hidden=true;
        // Lot list (radio buttons): tapping another lot is the intentional switch.
        ui.el('p',ui.t('Lot used','Lote usado'),row).className='formula-list-label';
        const list=ui.el('div',undefined,row);list.className='lot-list';list.setAttribute('role','radiogroup');
        const lbLabel=ui.el('label',ui.t('lb from this lot','lb de este lote'),row);lbLabel.append(lb);
        const remove=ui.el('button',ui.t('Remove this lot','Quitar este lote'),row);remove.type='button';remove.className='secondary';
        const prev=bag?.lot_id&&previous.get(bag.lot_id);
        const entry={row,pick,lb,lot:{value:bag?.lot_id?String(bag.lot_id):''},code:bag?.lot_code||'',
          // Evidence FL already accepted (any method), or a tap on this screen.
          evidence:bag?.confirmed?(prev||{lot_id:bag.lot_id,method:bag.confirmation_method||'selected',value:bag.lot_code}):null,paint};
        // A lot already chosen keeps its full identity even at 0 or negative book stock (FL sends it).
        const options=[...ing.candidates];if(bag?.lot_id&&!options.some(c=>c.lot_id===bag.lot_id)){const known=(ing.unstocked_lots||[]).find(l=>l.lot_id===bag.lot_id);options.unshift({...(known||{lot_id:bag.lot_id,lot_code:bag.lot_code,available_lb:null}),zero:true});}
        const choices=options.map((c,index)=>{const b=ui.el('button',lotText(c),list);b.type='button';b.setAttribute('role','radio');b.dataset.lotId=String(c.lot_id);
          if(index===0&&!c.zero)ui.el('span',ui.t(' · oldest',' · más antiguo'),b).className='muted';
          b.onclick=()=>choose(c);return b;});
        function candidate(){return options.find(c=>String(c.lot_id)===entry.lot.value);}
        function choose(c){const same=String(c.lot_id)===entry.lot.value;entry.lot.value=String(c.lot_id);entry.code=c.lot_code;entry.evidence={lot_id:c.lot_id,method:'selected',value:c.lot_code};
          // Spec D7: a different lot changes the plan (FL re-checks); the same lot is only a confirmation.
          refreshAll();if(!same)touch();}
        function paint(){const c=candidate(),ok=!!entry.evidence&&entry.lot.value!=='',amount=pounds(lb);
          error.hidden=!amount.invalid;lb.setAttribute('aria-invalid',String(amount.invalid));
          if(amount.invalid)ui.setText(error,ui.t('"'+lb.value+'" lb is not allowed: pounds cannot be negative. Enter 0 or more for this lot.','"'+lb.value+'" lb no es válido: las libras no pueden ser negativas. Escribe 0 o más para este lote.'));
          ui.setText(confirm,entry.lot.value?(ok?ui.t('✓ ','✓ '):ui.t('Tap to confirm: ','Toca para confirmar: '))+lotText(c,entry.code):ui.t('Choose a lot in "Change lot or pounds"','Elige un lote en "Cambiar lote o libras"'));
          confirm.className=ok?'lot-confirm confirmed':'lot-confirm';confirm.setAttribute('aria-pressed',String(ok));confirm.disabled=!entry.lot.value;
          // An invalid amount never reads as "Confirmed — 0 lb": the lot stays confirmed, the pounds do not.
          if(amount.invalid){ui.setText(state,ok?ui.t('Lot confirmed — pounds "'+lb.value+'" must be fixed','Lote confirmado — corrige las libras "'+lb.value+'"'):ui.t('Not confirmed — fix the pounds and tap the lot you used','Sin confirmar — corrige las libras y toca el lote que usaste'));state.className='bad';}
          else{ui.setText(state,ok?ui.t('Confirmed — '+ui.formatNumber(amount.value)+' lb from this lot','Confirmado — '+ui.formatNumber(amount.value)+' lb de este lote'):ui.t('Not confirmed — tap the lot you used','Sin confirmar — toca el lote que usaste'));state.className=ok?'good':'muted';}
          choices.forEach(b=>{const on=b.dataset.lotId===entry.lot.value;b.setAttribute('aria-checked',String(on));b.className=on?'lot-choice selected':'lot-choice';});
          const want=amount.value,book=c?.book_lb??c?.available_lb??(c?.zero?0:null),have=book==null?null:Math.max(book,0);
          short.hidden=!(have!=null&&want>have+0.005);
          // Same numbers as FL's shortage warning: book before, pounds used, NEW shortage, book after.
          const extra=round(want-(have||0)),after=book==null?null:round(book-want),f=ui.formatNumber;
          if(!short.hidden&&book<=0)ui.setText(short,ui.t('This lot is already at '+f(book)+' lb on the books — none available. Using '+f(want)+' lb adds '+f(extra)+' lb of shortage; book balance after: '+f(after)+' lb. If you also used another lot, open "Change lot or pounds" and add it with its pounds.','Este lote ya está en '+f(book)+' lb en el sistema — nada disponible. Usar '+f(want)+' lb agrega '+f(extra)+' lb de faltante; saldo en el sistema después: '+f(after)+' lb. Si también usaste otro lote, abre "Cambiar lote o libras" y agrégalo con sus libras.'));
          else if(!short.hidden)ui.setText(short,ui.t('This lot shows only '+f(have)+' lb on the books. Using '+f(want)+' lb is '+f(extra)+' lb more than it has (recorded as a shortage on this lot); book balance after: '+f(after)+' lb. If you also used another lot, open "Change lot or pounds" and add it with its pounds.','Este lote solo tiene '+f(have)+' lb en el sistema. Usar '+f(want)+' lb son '+f(extra)+' lb más de lo que tiene (se registra como faltante en este lote); saldo en el sistema después: '+f(after)+' lb. Si también usaste otro lote, abre "Cambiar lote o libras" y agrégalo con sus libras.'));}
        confirm.onclick=()=>{if(!entry.lot.value)return;entry.evidence=entry.evidence?null:{lot_id:Number(entry.lot.value),method:'selected',value:entry.code};refreshAll();};
        lb.oninput=lb.onchange=()=>{refreshAll();touch();};
        remove.onclick=()=>{const index=bags.indexOf(entry),moved=pounds(lb).value;bags.splice(index,1);row.remove();pick.remove();const target=bags[index-1]||bags[index];if(target&&moved&&!pounds(target.lb).invalid)target.lb.value=String(round(pounds(target.lb).value+moved));refreshAll();touch();};
        bags.push(entry);return entry;
      }
      ing.bags.forEach(addBag);
      if(!ing.bags.length&&ing.standard_lb>0)edit.open=true;
      const tools=ui.el('div',undefined,edit);tools.className='formula-tools';
      const add=ui.el('button',ui.t('Add another lot','Agregar otro lote'),tools);add.type='button';add.className='secondary';
      // A second lot starts with no lot chosen and no pounds: the worker picks both.
      add.onclick=()=>{addBag(null);edit.open=true;refreshAll();touch();};
      const none=ui.el('button',ui.t('Not used (0 lb)','No se usó (0 lb)'),tools);none.type='button';none.className='secondary';
      none.onclick=()=>{bags.splice(0).forEach(b=>{b.row.remove();b.pick.remove();});refreshAll();touch();};
      reason.onchange=note.oninput=touch;
    }
    refreshAll();
    const problem=ui.el('p','',checkHost);problem.className='bad';problem.hidden=true;problem.setAttribute('role','alert');
    const kept=bags=>bags.filter(b=>b.lot.value&&!pounds(b.lb).invalid&&pounds(b.lb).value>0);
    function selections(){return cards.flatMap(c=>kept(c.bags)).filter(b=>b.evidence).map(b=>({lot_id:Number(b.lot.value),method:b.evidence.method,value:b.evidence.value}));}
    function pending(){return cards.flatMap(c=>kept(c.bags)).filter(b=>!b.evidence).length;}
    function invalidCount(){return cards.reduce((n,c)=>n+c.invalid(),0);}
    function read(){const lines=[];let issue=null;
      for(const {ing,bags,reason,note,invalid} of cards){const used=kept(bags);
        if(invalid())issue=issue||ui.t('Pounds cannot be negative ('+ing.product_name+'). Fix the amount; nothing was changed.','Las libras no pueden ser negativas ('+ing.product_name+'). Corrige la cantidad; no se cambió nada.');
        if(bags.length&&!used.length&&!invalid()&&ing.standard_lb>0&&!(ing.swap_role==='from'&&swapLb()>0))issue=issue||ui.t('Enter the pounds used for '+ing.product_name+', or tap "Not used (0 lb)".','Escribe las libras usadas de '+ing.product_name+', o toca "No se usó (0 lb)".');
        if(bags.some(b=>!b.lot.value&&pounds(b.lb).value>0))issue=issue||ui.t('Choose a lot for every pound entered ('+ing.product_name+').','Elige un lote para cada libra escrita ('+ing.product_name+').');
        const zero=!used.length&&ing.standard_lb>0&&!(ing.swap_role==='from'&&swapLb()>0);lines.push({ingredient_product_id:ing.product_id,bags:used.map(b=>({lot_id:Number(b.lot.value),quantity_lb:Number(b.lb.value)})),reason_code:zero&&reason.value?reason.value:null,note:zero&&note.value.trim()?note.value.trim():null});}
      return {lines,confirmations:selections(),issue};}
    check=ui.el('button',ui.t('Check lots and pounds → new draft','Verificar lotes y libras → nuevo borrador'),checkHost);check.type='button';check.dataset.formulaUpdate='';check.disabled=invalidCount()>0;
    check.onclick=()=>{if(check.disabled)return;const {lines,confirmations:next,issue}=read();if(issue){problem.hidden=false;ui.setText(problem,issue);return;}problem.hidden=true;update(lines,next);};
    return {read,check,selections,pending,invalid:invalidCount,anchor:()=>lastEdited};
  };
})(window);
