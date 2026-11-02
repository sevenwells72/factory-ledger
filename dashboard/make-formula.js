/* Formula-driven Make (docs/production/formula-driven-entry.md): FL's draft lists each
   recipe ingredient with the bags its oldest-first plan chose. The worker edits pounds,
   splits bags, marks an ingredient not used (with a reason) and types each bag's code.
   The browser only collects choices; FL recalculates, re-checks and posts on Record. */
(function(root){
  const ui=root.FLLedger;
  // Spec D3 — must match formula_entry.REASONS (tests/test_formula_entry.py checks it).
  const reasons=[['out_of_stock','Out of stock','Sin existencias',false],['not_used','Not used','No se usó',false],['substituted','Substituted','Sustituido',false],['quality_problem','Quality problem','Problema de calidad',false],['owner_instruction','Owner instruction','Instrucción del dueño',false],['other','Other','Otro',true]];
  ui.makeReasons=reasons;
  ui.makeReasonSelect=function(host,name,value=''){const label=ui.el('label',ui.t('Reason','Motivo'),host),select=ui.el('select',undefined,label);select.name=name;ui.el('option',ui.t('Choose reason','Elige motivo'),select).value='';reasons.forEach(([code,en,es])=>{const o=ui.el('option',ui.t(en,es),select);o.value=code;});select.value=value||'';return select;};
  const round=n=>Math.round(n*100)/100;
  ui.formulaEntry=function(card,block,{confirmations=[],dirty=()=>{},update=()=>{},checkHost=card}={}){
    const intro=ui.el('p',ui.t(block.pans+' pans × '+ui.formatNumber(block.pan_lb)+' lb = '+ui.formatNumber(block.output_lb)+' lb. Check each ingredient, then type the code on every bag you used.',block.pans+' pans × '+ui.formatNumber(block.pan_lb)+' lb = '+ui.formatNumber(block.output_lb)+' lb. Revisa cada ingrediente y escribe el código de cada bolsa que usaste.'),card);intro.className='muted';intro.dataset.formulaSummary='';
    const previous=new Map(confirmations.map(c=>[c.lot_id,c])),cards=[];let lastEdited=null;
    const byRole=role=>cards.find(c=>c.ing.swap_role===role);
    function swapLb(){const from=byRole('from'),to=byRole('to');if(!block.swap||!from||!to)return 0;return Math.max(0,round(Math.min(from.ing.standard_lb-from.sum(),to.sum()-to.ing.standard_lb)));}
    function refreshAll(){cards.forEach(c=>c.refresh());}
    for(const ing of block.ingredients){
      const box=ui.el('fieldset',undefined,card);box.className='formula-ingredient';box.dataset.productId=String(ing.product_id);
      const touch=()=>{lastEdited=box;dirty();};
      const legend=ui.el('legend',ing.product_name||'—',box);
      if(ing.substituted_for){const p=ui.el('p',ui.t('Swapped in for a recipe ingredient.','Sustituye a un ingrediente de la receta.'),box);p.className='muted';}
      if(ing.swap_role==='to'){const p=ui.el('p',ui.t('Rolled oats can replace missing oat bran pound for pound; the swapped pounds count toward the oat bran recipe.','La avena en hojuelas puede reemplazar el salvado de avena que falta libra por libra; esas libras cuentan para la receta del salvado.'),box);p.className='muted';}
      const std=ui.el('p',ui.t('Recipe: '+ui.formatNumber(ing.standard_lb)+' lb','Receta: '+ui.formatNumber(ing.standard_lb)+' lb'),box);std.dataset.standard=String(ing.standard_lb);
      const total=ui.el('p','',box);total.dataset.total='';total.setAttribute('role','status');
      const rows=ui.el('div',undefined,box);rows.className='formula-bags';const bags=[];
      const reasonArea=ui.el('div',undefined,box);reasonArea.className='formula-reason';
      const reason=ui.makeReasonSelect(reasonArea,'ingredient_reason',ing.reason_code),noteLabel=ui.el('label',ui.t('Note (required for Other)','Nota (obligatoria para Otro)'),reasonArea),note=ui.el('input',undefined,noteLabel);note.name='ingredient_note';note.value=ing.note||'';note.maxLength=2000;
      function sum(){return round(bags.reduce((n,b)=>n+(Number(b.lb.value)||0),0));}
      // Oat bran + the oats that replace it are checked against the recipe bran;
      // the oats card is checked without those pounds (formula_entry.review).
      function refresh(){const used=sum(),swap=swapLb(),role=ing.swap_role,checked=role==='from'?round(used+swap):role==='to'?round(used-swap):used,swapped=role==='from'&&swap>0;
        const pct=ing.standard_lb>0?Math.round((checked-ing.standard_lb)/ing.standard_lb*1000)/10:0,over=ing.standard_lb>0?Math.abs(checked-ing.standard_lb)>ing.standard_lb*block.tolerance_pct/100+1e-9:checked>0.005;
        const diff=pct?' ('+(pct>0?'+':'')+pct+'%)':'';
        ui.setText(legend,swapped?ui.t((block.swap.from_name||'oat bran')+' → '+(block.swap.to_name||'rolled oats')+' (swap)',(block.swap.from_name||'salvado de avena')+' → '+(block.swap.to_name||'avena')+' (sustitución)'):(ing.product_name||'—'));
        const amount=swapped?ui.t(ui.formatNumber(used)+' lb + '+ui.formatNumber(swap)+' lb '+(block.swap.to_name||'rolled oats')+' = '+ui.formatNumber(checked)+' lb',ui.formatNumber(used)+' lb + '+ui.formatNumber(swap)+' lb '+(block.swap.to_name||'avena')+' = '+ui.formatNumber(checked)+' lb')
          :role==='to'&&swap>0?ui.t(ui.formatNumber(used)+' lb ('+ui.formatNumber(swap)+' lb replace oat bran)',ui.formatNumber(used)+' lb ('+ui.formatNumber(swap)+' lb reemplazan salvado)'):ui.formatNumber(used)+' lb';
        const zero=used<=0&&!swapped&&ing.standard_lb>0;
        ui.setText(total,zero?ui.t('Not used — 0 lb. Choose a reason.','No se usó — 0 lb. Elige un motivo.'):over?ui.t('Using '+amount+diff+' — more than 10% off the recipe; the owner will review it.','Usando '+amount+diff+' — más de 10% diferente de la receta; el dueño lo revisará.'):ui.t('Using '+amount+diff,'Usando '+amount+diff));
        total.className=!zero&&!over?'muted':'notice';total.dataset.over=String(over&&!zero);
        reasonArea.hidden=!zero;}
      const item={ing,bags,reason,note,box,sum,refresh};cards.push(item);
      function addBag(bag){
        const row=ui.el('fieldset',undefined,rows);row.className='lot-confirmation-row formula-bag';
        const controls=ui.el('div',undefined,row);controls.className='lot-confirmation-controls';
        const lotLabel=ui.el('label',ui.t('Bag','Bolsa'),controls),lot=ui.el('select',undefined,lotLabel);lot.name='bag_lot';lotLabel.className='formula-bag-lot';
        const options=[...ing.candidates];if(bag?.lot_id&&!options.some(c=>c.lot_id===bag.lot_id))options.unshift({lot_id:bag.lot_id,lot_code:bag.lot_code,available_lb:null});
        ui.el('option',ui.t('Choose bag','Elige bolsa'),lot).value='';
        for(const c of options){const o=ui.el('option',c.lot_code+(c.received_on?' · '+c.received_on:'')+(c.available_lb!=null?' · '+ui.formatNumber(c.available_lb)+' lb':''),lot);o.value=String(c.lot_id);}
        lot.value=bag?.lot_id?String(bag.lot_id):'';
        const lbLabel=ui.el('label',ui.t('lb from this bag','lb de esta bolsa'),controls),lb=ui.el('input',undefined,lbLabel);lb.name='bag_lb';lb.type='number';lb.step='any';lb.min='0';lb.value=bag?.quantity_lb!=null?String(bag.quantity_lb):'';
        const methodLabel=ui.el('label',ui.t('Confirmation method','Método de confirmación'),controls),method=ui.el('select',undefined,methodLabel);method.name='bag_method';methodLabel.className='formula-bag-method';
        [['last4',ui.t('Last 4 (with hyphen)','Últimos 4 (con guion)')],['full_code',ui.t('Full code','Código completo')],['pallet',ui.t('Pallet moved (full code)','Palé movido (código completo)')]].forEach(([v,t])=>{const o=ui.el('option',t,method);o.value=v;});
        const codeLabel=ui.el('label',ui.t('Code on the bag label','Código en la etiqueta'),controls),code=ui.el('input',undefined,codeLabel);code.name='bag_code';code.autocomplete='off';
        const prev=bag?.lot_id&&previous.get(bag.lot_id);if(prev){method.value=prev.method;code.value=prev.value;}
        const footer=ui.el('div',undefined,row);footer.className='formula-bag-footer';const state=ui.el('p','',footer);state.dataset.bagState='';
        const show=ok=>{ui.setText(state,ok?ui.t('✓ Confirmed','✓ Confirmado'):ui.t('Not confirmed — type the code','Sin confirmar — escribe el código'));state.className=ok?'good':'muted';};show(!!bag?.confirmed);
        const remove=ui.el('button',ui.t('Remove bag','Quitar bolsa'),footer);
        const choices=ui.el('div',undefined,row);choices.className='lot-confirmation-choices';remove.type='button';remove.className='secondary';
        const entry={row,lot,lb,method,code,choices,state};bags.push(entry);
        // Spec D7: new pounds keep the confirmation; a different bag must be confirmed again.
        lot.onchange=()=>{code.value='';show(false);choices.replaceChildren();touch();};
        lb.oninput=()=>{refreshAll();touch();};method.onchange=code.oninput=()=>{choices.replaceChildren();touch();};
        remove.onclick=()=>{const index=bags.indexOf(entry),moved=Number(lb.value)||0;bags.splice(index,1);row.remove();const target=bags[index-1]||bags[index];if(target&&moved)target.lb.value=String(round((Number(target.lb.value)||0)+moved));refreshAll();touch();};
        return entry;
      }
      ing.bags.forEach(addBag);
      const tools=ui.el('div',undefined,box);tools.className='formula-tools';
      const add=ui.el('button',ui.t('Add another bag','Agregar otra bolsa'),tools);add.type='button';add.className='secondary';
      add.onclick=()=>{const used=new Set(bags.map(b=>b.lot.value));const next=ing.candidates.find(c=>!used.has(String(c.lot_id)));const entry=addBag(next?{lot_id:next.lot_id,lot_code:next.lot_code}:null);entry.lb.value='';refreshAll();touch();};
      const none=ui.el('button',ui.t('Not used (0 lb)','No se usó (0 lb)'),tools);none.type='button';none.className='secondary';
      none.onclick=()=>{bags.splice(0).forEach(b=>b.row.remove());refreshAll();touch();};
      reason.onchange=note.oninput=touch;
    }
    refreshAll();
    const problem=ui.el('p','',checkHost);problem.className='bad';problem.hidden=true;problem.setAttribute('role','alert');
    function read(){const lines=[],next=[];let issue=null;
      for(const {ing,bags,reason,note} of cards){const kept=bags.filter(b=>b.lot.value&&Number(b.lb.value)>0);
        if(bags.some(b=>!b.lot.value&&Number(b.lb.value)>0))issue=issue||ui.t('Choose a bag for every pound entered ('+ing.product_name+').','Elige una bolsa para cada libra escrita ('+ing.product_name+').');
        const zero=!kept.length&&ing.standard_lb>0&&!(ing.swap_role==='from'&&swapLb()>0);lines.push({ingredient_product_id:ing.product_id,bags:kept.map(b=>({lot_id:Number(b.lot.value),quantity_lb:Number(b.lb.value)})),reason_code:zero&&reason.value?reason.value:null,note:zero&&note.value.trim()?note.value.trim():null});
        for(const b of kept){const value=b.code.value.trim();if(!value)continue;
          if(b.method.value==='last4'&&value.length===4){const matches=ing.candidates.filter(c=>c.lot_code.toUpperCase().endsWith(value.toUpperCase()));
            if(matches.length>1){b.choices.replaceChildren();ui.el('p',ui.t('More than one bag ends with '+value+'. Type the full code from the bag.','Más de una bolsa termina en '+value+'. Escribe el código completo de la bolsa.'),b.choices);issue=issue||ui.t('Type the full code where more than one bag matches.','Escribe el código completo donde coincide más de una bolsa.');continue;}}
          next.push({lot_id:Number(b.lot.value),method:b.method.value,value});}}
      return {lines,confirmations:next,issue};}
    const check=ui.el('button',ui.t('Check bags and pounds → new draft','Verificar bolsas y libras → nuevo borrador'),checkHost);check.type='button';check.dataset.formulaUpdate='';
    check.onclick=()=>{const {lines,confirmations:next,issue}=read();if(issue){problem.hidden=false;ui.setText(problem,issue);return;}problem.hidden=true;update(lines,next);};
    return {read,check,anchor:()=>lastEdited};
  };
})(window);
