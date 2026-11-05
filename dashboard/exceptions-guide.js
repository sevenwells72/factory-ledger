/* Exceptions page guidance: plain-language help, per-item explanations and the
   one-line confirmation. Text only — FL owns every rule. Each effect sentence
   below describes what the server really does (exceptions_enforcement.py,
   supplier_lot_tickets.py); change them together. */
(function(root){
  'use strict';
  const num=value=>Number.isFinite(Number(value))?new Intl.NumberFormat('en-US',{maximumFractionDigits:2}).format(Number(value)):null;
  const pick=(pair,lang)=>pair[lang==='es'?1:0];

  // What each resolution choice is for and what it really does.
  const OPTIONS={
    counted:{
      use:['You can weigh or count what is left of this lot right now.','Puedes pesar o contar ahora lo que queda de este lote.'],
      example:['The system shows canola lot 26-09-17-CBSF-001 at −14 lb. You weigh the drum: 10 lb left. Enter 10.','El sistema muestra el lote de canola 26-09-17-CBSF-001 en −14 lb. Pesas el tambo: quedan 10 lb. Escribe 10.'],
      effect:['Sets this lot to the pounds you enter (a physical-count correction with its own receipt) and closes the exception. A change over 500 lb needs a photo.','Pone este lote en las libras que escribas (una corrección por conteo físico con su propio recibo) y cierra la excepción. Un cambio de más de 500 lb necesita foto.']},
    missing_movement:{
      use:['Stock for this same lot was entered late, after the shortage.','Se registró tarde, después del faltante, inventario de este mismo lote.'],
      example:['The canola delivery was entered today as RCV-261105-002 on the same lot. Enter that receipt number.','La entrega de canola se registró hoy como RCV-261105-002 en el mismo lote. Escribe ese número de recibo.'],
      effect:['Changes no stock. Uses that receipt\'s pounds on this lot to explain the shortage. If it covers only part, the exception stays open for the rest.','No cambia el inventario. Usa las libras de ese recibo en este lote para explicar el faltante. Si cubre solo una parte, la excepción sigue abierta por el resto.']},
    voided:{
      use:['The entry that caused the shortage was wrong and is already voided.','La entrada que causó el faltante estaba mal y ya está anulada.'],
      example:['The make was entered twice. The duplicate was voided on its receipt page.','La producción se registró dos veces. El duplicado se anuló en su página de recibo.'],
      effect:['Changes no stock. Closes the exception. FL refuses if that entry is not voided yet.','No cambia el inventario. Cierra la excepción. FL lo rechaza si esa entrada aún no está anulada.']},
    unverified:{
      use:['You cannot count it now and no other fix fits, but you know what happened.','No puedes contarlo ahora y ninguna otra solución aplica, pero sabes qué pasó.'],
      example:['The lot was used up and the empty bag thrown out before anyone counted. Arturo saw it was empty.','El lote se terminó y la bolsa vacía se tiró antes de contarlo. Arturo vio que estaba vacía.'],
      effect:['Changes no stock: the lot keeps its book balance. Closes the exception and marks it Unverified for the owner to review.','No cambia el inventario: el lote conserva su saldo en el sistema. Cierra la excepción y la marca Sin verificar para que el dueño la revise.']},
    identified:{
      use:['You can read the supplier\'s lot code on the bag, box, label or BOL.','Puedes leer el código de lote del proveedor en la bolsa, caja, etiqueta o BOL.'],
      example:['The oat bag label says LOT 2409-117. Enter 2409-117.','La etiqueta de la bolsa de avena dice LOT 2409-117. Escribe 2409-117.'],
      effect:['Saves that supplier lot code on this lot, marks the lot identified and closes the exception. Stock does not change.','Guarda ese código de lote del proveedor en este lote, marca el lote como identificado y cierra la excepción. El inventario no cambia.']},
    written_off:{
      use:['The supplier lot code can never be found.','El código de lote del proveedor no se podrá encontrar nunca.'],
      example:['The bags are gone and the BOL shows no lot code.','Las bolsas ya no están y el BOL no tiene código de lote.'],
      effect:['Changes no stock. Closes the exception as waived. The lot stays without a supplier lot code.','No cambia el inventario. Cierra la excepción como eximida. El lote se queda sin código de lote del proveedor.']},
    approved:{
      use:['You checked the held entry and it is right.','Revisaste la entrada retenida y está bien.'],
      example:['A 600 lb correction after a full recount. The count sheet matches.','Una corrección de 600 lb después de un reconteo completo. La hoja de conteo coincide.'],
      effect:['Posts the held entry exactly as it was prepared (stock changes as it shows) and closes the exception.','Registra la entrada retenida tal como se preparó (el inventario cambia como se muestra) y cierra la excepción.']},
    declined:{
      use:['The held entry is wrong.','La entrada retenida está mal.'],
      example:['The correction was typed as 6,000 lb instead of 60 lb.','La corrección se escribió como 6,000 lb en vez de 60 lb.'],
      effect:['Posts nothing. Rejects the held entry and closes the exception.','No registra nada. Rechaza la entrada retenida y cierra la excepción.']},
    photo_attached:{
      use:['You have a photo of the signed BOL.','Tienes una foto del BOL firmado.'],
      example:['The driver\'s signed BOL is in the shipping folder. Take a clear photo of it.','El BOL firmado por el chofer está en la carpeta de envíos. Tómale una foto clara.'],
      effect:['Attaches the photo to the shipment and closes the exception. Stock does not change.','Adjunta la foto al envío y cierra la excepción. El inventario no cambia.']},
    waived:{
      use:['The photo or proof will never exist and you accept that.','La foto o prueba nunca existirá y lo aceptas.'],
      example:['The customer picked up and no BOL was signed.','El cliente recogió y no se firmó ningún BOL.'],
      effect:['Changes no stock. Closes the exception as waived.','No cambia el inventario. Cierra la excepción como eximida.']},
    acknowledged:{
      use:['You have read it and no fix is needed.','Lo leíste y no hace falta corregir nada.'],
      example:['A late entry was for a real delivery that came on Friday.','Un registro tardío era de una entrega real que llegó el viernes.'],
      effect:['Changes no stock. Closes the exception as reviewed.','No cambia el inventario. Cierra la excepción como revisada.']},
  };
  function option(kind,lang){const o=OPTIONS[kind];return o?{use:pick(o.use,lang),example:pick(o.example,lang),effect:pick(o.effect,lang)}:null;}

  // Short helper lines under fields (always visible, so they work on phones).
  const FIELDS={
    resolution:['Pick the fix that matches what really happened.','Elige la solución que coincide con lo que de verdad pasó.'],
    counted_lb:['Weight of THIS lot left right now. Enter 0 if it is empty.','Peso de ESTE lote que queda ahora. Escribe 0 si está vacío.'],
    note:['What happened and how you know.','Qué pasó y cómo lo sabes.'],
    note_unverified:['Required. What happened, how you know, and who checked.','Obligatoria. Qué pasó, cómo lo sabes y quién revisó.'],
    photo:['Add one when it helps: the scale, the empty pallet. Needed if the change is over 500 lb.','Agrega una cuando ayude: la báscula, la tarima vacía. Necesaria si el cambio es de más de 500 lb.'],
    photo_bol:['A clear photo of the signed BOL.','Una foto clara del BOL firmado.'],
    receipt_number:['The receipt # of the late receive or make, e.g. RCV-261105-002. Find it in Today\'s entries.','El # de recibo de la recepción o producción tardía, p. ej. RCV-261105-002. Búscalo en Entradas de hoy.'],
    supplier_lot_code:['Printed on the supplier\'s bag, box, label or BOL. Often after "LOT" or "L".','Impreso en la bolsa, caja, etiqueta o BOL del proveedor. Suele ir después de "LOT" o "L".'],
    reason:['Data-entry error: typed wrong or left out. Wrong lot used: the entry named the wrong lot. Both save the code the same way.','Error de captura: se escribió mal o faltó. Lote equivocado: la entrada nombró otro lote. Ambos guardan el código igual.'],
  };
  const field=(name,lang)=>FIELDS[name]?pick(FIELDS[name],lang):'';

  const TIPS={
    refresh:['Load the latest open exceptions from FL.','Carga las excepciones abiertas más recientes de FL.'],
    help:['What exceptions are and how to fix them.','Qué son las excepciones y cómo resolverlas.'],
    explain:['Show what happened with this item, in plain words.','Muestra qué pasó con este elemento, en palabras simples.'],
    resolve:['Open the fix form for this item. Nothing is saved yet.','Abre el formulario para este elemento. Aún no se guarda nada.'],
    review:['Check what will be saved. Nothing is saved yet.','Revisa lo que se guardará. Aún no se guarda nada.'],
    confirm:['Save this fix now. It gets a receipt.','Guarda esta solución ahora. Recibe un recibo.'],
  };
  const tip=(name,lang)=>TIPS[name]?pick(TIPS[name],lang):'';

  // "How this works" panel.
  const HELP=[
    {title:['What is an exception?','¿Qué es una excepción?'],lines:[['Something in the records does not add up, or is missing.','Algo en los registros no cuadra, o falta.'],['Example: a make used more of a lot than the system had.','Ejemplo: una producción usó más de un lote de lo que el sistema tenía.']]},
    {title:['What to do','Qué hacer'],lines:[['1. Open the item. Tap Explain to see what happened.','1. Abre el elemento. Toca Explicar para ver qué pasó.'],['2. Find out what really happened: check the lot, the paperwork, ask the team.','2. Averigua qué pasó de verdad: revisa el lote, los papeles, pregunta al equipo.'],['3. Tap Resolve and pick the fix that matches.','3. Toca Resolver y elige la solución que coincide.'],['4. Review, then confirm.','4. Revisa y luego confirma.']]},
    {title:['If you are not sure','Si no estás seguro'],lines:[['Leave it open. Tell the owner. Do not guess.','Déjala abierta. Avisa al dueño. No adivines.']]},
    {title:['What Resolve changes','Qué cambia Resolver'],lines:[['Each fix shows "This will:" before you save.','Cada solución muestra "Esto hará:" antes de guardar.'],['Only two fixes change stock: Counted adjustment sets the lot to your count, and the owner\'s Approve posts a held entry. All others change no stock.','Solo dos soluciones cambian el inventario: Ajuste contado pone el lote en tu conteo, y Aprobar del dueño registra una entrada retenida. Las demás no cambian el inventario.'],['Every fix is saved with your name and a receipt. Nothing is deleted.','Cada solución se guarda con tu nombre y un recibo. No se borra nada.']]},
  ];
  const help=lang=>HELP.map(s=>({title:pick(s.title,lang),lines:s.lines.map(l=>pick(l,lang))}));

  const ACTIONS={make:['The make','La producción'],pack:['The pack','El empaque'],ship:['The shipment','El envío'],adjust:['The correction','La corrección']};
  function names(item){return {product:item?.product?.name||'',lot:item?.lot?.lot_code||'',id:item?.id};}

  // Plain-language, item-specific explanation built only from the item's data.
  function explain(item,lang='en'){
    const d=item?.detail||{},{product,lot}=names(item),es=lang==='es';
    const what=[product,lot&&(es?'lote ':'lot ')+lot].filter(Boolean).join(es?' del ':' from ')||(es?'este lote':'this lot');
    if(item?.kind==='SHORTAGE'||item?.kind==='NEGATIVE_BALANCE'){
      const used=num(d.needed_lb),had=num(d.on_hand_before_lb),after=num(d.balance_after_lb),short=num(d.short_lb??item?.shortage_flag?.short_lb);
      const act=pick(ACTIONS[d.action]||['The entry','La entrada'],lang);
      const parts=[];
      if(used!==null&&had!==null){
        const ingredient=es?`${product?product+' ':''}${lot?'del lote '+lot:''}`.trim():`${product?product+' ':''}${lot?'from lot '+lot:''}`.trim();
        parts.push(Number(d.on_hand_before_lb)>0
          ?(es?`${act} usó ${used} lb de ${ingredient}, pero el sistema solo tenía ${had} lb.`:`${act} used ${used} lb of ${ingredient}, but the system only had ${had} lb.`)
          :(es?`${act} usó ${used} lb de ${ingredient}, pero el sistema ya estaba en ${had} lb (nada disponible).`:`${act} used ${used} lb of ${ingredient}, but the system already showed ${had} lb (none available).`));
        if(after!==null)parts.push(es?`Después de esa entrada el lote quedó en ${after} lb en el sistema.`:`After that entry the lot was at ${after} lb on the books.`);
      }else if(short!==null)parts.push(es?`Se usaron ${short} lb más de ${what} de lo que el sistema tenía.`:`${short} lb more of ${what} was used than the system had.`);
      else parts.push(es?`Se usó más de ${what} de lo que el sistema tenía.`:`More of ${what} was used than the system had.`);
      const ev=item?.evidence;
      if(ev&&Number(ev.claimed_lb)>0&&num(ev.remaining_lb)!==null)parts.push(es?`Recibos ya explican ${num(ev.claimed_lb)} lb; faltan ${num(ev.remaining_lb)} lb.`:`Receipts already explain ${num(ev.claimed_lb)} lb; ${num(ev.remaining_lb)} lb is still unexplained.`);
      parts.push(es?'Causas probables: una entrega que nunca se registró, se tomó el lote equivocado, o una entrada anterior registró de más.':'Likely causes: a delivery never entered, the wrong lot picked, or an earlier entry overstated.');
      return parts.join(' ');
    }
    if(item?.kind==='UNIDENTIFIED_LOT'){
      const base=es?`${what} se registró sin el código de lote del proveedor. Sin él no se puede rastrear.`:`${what} was entered without the supplier's lot code. Without it the lot cannot be traced.`;
      const open=item.is_opening_stock?(es?' Este lote viene del inventario inicial.':' This lot came from the opening stock count.'):'';
      return base+open+(es?' Búscalo en la bolsa, caja, etiqueta o BOL.':' Look for it on the bag, box, label or BOL.');
    }
    const generic={
      LARGE_CORRECTION:['A correction over 500 lb was entered without a photo. It waits for the owner. Stock has not changed yet.','Se registró una corrección de más de 500 lb sin foto. Espera al dueño. El inventario aún no cambia.'],
      KOSHER_ATTESTATION:['An extra-kosher entry waits for the owner\'s approval. Stock has not changed yet.','Una entrada extra-kosher espera la aprobación del dueño. El inventario aún no cambia.'],
      LATE_ENTRY:['This entry was recorded well after it happened. The owner reviews it.','Esta entrada se registró mucho después de que pasó. El dueño la revisa.'],
      BOL_PHOTO_MISSING:['A shipment has no photo of its signed BOL yet.','Un envío aún no tiene foto de su BOL firmado.'],
      SHIPMENT_PROOF_MISSING:['A shipment has no proof photo yet.','Un envío aún no tiene foto de prueba.'],
      PRE_MAKE_ADJUST:['Stock was added to a lot shortly before a make used it. The owner checks it was real.','Se agregó inventario a un lote poco antes de que una producción lo usara. El dueño revisa que fuera real.'],
      ASSISTANT_OFFLINE:['The Assistant was offline. The owner reviews what happened meanwhile.','El Asistente estuvo sin conexión. El dueño revisa lo que pasó mientras tanto.'],
    }[item?.kind];
    const message=es?d.message_es||d.message:d.message;
    return [generic?pick(generic,lang):(es?'Algo en los registros necesita revisión.':'Something in the records needs a check.'),message].filter(Boolean).join(' ');
  }

  // One line shown right before the final save, built from the values to be sent.
  function confirmLine(item,kind,values={},lang='en'){
    const {product,lot,id}=names(item),es=lang==='es';
    const lotName=es?(lot?`el lote ${lot}${product?' de '+product:''}`:'el lote'):([product,lot&&'lot '+lot].filter(Boolean).join(' ')||'the lot');
    const head=es?'Estás a punto de: ':'You are about to: ';
    const lines={
      counted:()=>es?`poner ${lotName} en ${num(values.counted_lb)??'?'} lb y cerrar la excepción #${id}.`:`set ${lotName} to ${num(values.counted_lb)??'?'} lb and close exception #${id}.`,
      missing_movement:()=>es?`usar el recibo ${values.receipt_number||'?'} para explicar el faltante de ${lotName} y cerrar la excepción #${id} (sigue abierta si el recibo cubre menos). El inventario no cambia.`:`use receipt ${values.receipt_number||'?'} to explain the shortage on ${lotName} and close exception #${id} (it stays open if the receipt covers less). Stock does not change.`,
      voided:()=>es?`cerrar la excepción #${id} porque su entrada fue anulada. El inventario no cambia.`:`close exception #${id} because its entry was voided. Stock does not change.`,
      unverified:()=>es?`cerrar la excepción #${id} sin conteo. ${lotName} conserva su saldo en el sistema. Quedará marcada Sin verificar.`:`close exception #${id} without a count. ${lotName} keeps its book balance. It will be marked Unverified.`,
      identified:()=>es?`guardar el lote del proveedor ${values.supplier_lot_code||'?'} en ${lotName} y cerrar la excepción #${id}.`:`save supplier lot ${values.supplier_lot_code||'?'} on ${lotName} and close exception #${id}.`,
      written_off:()=>es?`cerrar la excepción #${id} como eximida. ${lotName} se queda sin código del proveedor.`:`close exception #${id} as waived. ${lotName} stays without a supplier lot code.`,
      approved:()=>es?`registrar la entrada retenida tal como se preparó y cerrar la excepción #${id}.`:`post the held entry as prepared and close exception #${id}.`,
      declined:()=>es?`rechazar la entrada retenida (no se registra nada) y cerrar la excepción #${id}.`:`reject the held entry (nothing posts) and close exception #${id}.`,
      photo_attached:()=>es?`adjuntar la foto del BOL al envío y cerrar la excepción #${id}.`:`attach the BOL photo to the shipment and close exception #${id}.`,
      waived:()=>es?`cerrar la excepción #${id} como eximida, sin foto.`:`close exception #${id} as waived, without the photo.`,
      acknowledged:()=>es?`cerrar la excepción #${id} como revisada. El inventario no cambia.`:`close exception #${id} as reviewed. Stock does not change.`,
    };
    return head+(lines[kind]?lines[kind]():(es?`cerrar la excepción #${id}.`:`close exception #${id}.`));
  }

  const api={OPTIONS,option,field,tip,help,explain,confirmLine};
  if(typeof module==='object'&&module.exports)module.exports=api;
  if(root)root.FLExceptionsGuide=api;
})(typeof window==='undefined'?null:window);
