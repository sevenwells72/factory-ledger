/* Office choices -> A7 prepare -> shared reviewed ticket -> Record. */
(() => {
  'use strict';
  const ui=FLLedger,logic=FLOffice;ui.bind();
  const content=document.getElementById('content'),draft=document.getElementById('resolution');
  const initial=new URLSearchParams(location.search);
  function todayLink(){const link=document.getElementById('today-entries-link');if(link)link.href='receipt.html?view=today&lang='+encodeURIComponent(ui.lang||'en');}
  todayLink();
  let action=initial.get('action')||'create_order',grants={},controller,revision=0,authRevision=0,readyIntent=null,selectedCustomer=null;
  if(action==='mark_order_ready'){
    try{const saved=JSON.parse(sessionStorage.getItem('fl-order-ready-draft')||'null');if(saved&&String(saved.order_id)===initial.get('order')&&typeof saved.ready==='boolean')readyIntent=saved;}catch(_){}
    sessionStorage.removeItem('fl-order-ready-draft');
  }
  const reasonText={
    shipped_recorded:['Shipped, recorded','Enviado, registrado'],shipped_not_recorded:['Shipped, not recorded (legacy)','Enviado, no registrado (histórico)'],short_closed:['Short closed','Cierre parcial'],
    customer_cancelled:['Customer cancelled','Cancelado por cliente'],cns_declined:['CNS declined','CNS rechazó'],duplicate:['Duplicate','Duplicado'],superseded:['Replaced by another order','Reemplazado por otro pedido'],other:['Other','Otro'],
    new:['New','Nuevo'],confirmed:['Confirmed','Confirmado'],in_production:['In production','En producción'],ready:['Ready to ship','Listo para enviar'],cancelled:['Cancelled','Cancelado'],invoiced:['Invoiced','Facturado'],closed:['Closed','Cerrado']
  };
  function invalidate(){revision++;controller?.invalidate();controller=null;draft.replaceChildren();document.getElementById('notice').hidden=true;}
  function picker(host,name,en,es,kind,changed=()=>{},missing=null){
    const box=ui.el('fieldset',undefined,host);box.dataset.picker=name;ui.el('legend',ui.t(en,es),box);
    const label=ui.el('label',ui.t('Search name or code','Buscar nombre o código'),box),input=ui.el('input',undefined,label);input.autocomplete='off';
    const button=ui.el('button',ui.t('Search','Buscar'),box);button.type='button';
    const list=ui.el('div',undefined,box);list.className='picker-options';const status=ui.el('p','',box);status.setAttribute('role','status');
    let value=null,generation=0,timer=null;
    const clear=()=>{clearTimeout(timer);generation++;value=null;button.disabled=false;list.replaceChildren();status.textContent='';invalidate();changed(null);};
    input.oninput=()=>{clear();if(input.value.trim().length>=2)timer=setTimeout(()=>find(),300);};
    function select(candidate,requestOptions={}){clearTimeout(timer);generation++;button.disabled=false;value=candidate;input.value=candidate.label||candidate.name;list.replaceChildren();status.textContent=ui.t('Selected: ','Seleccionado: ')+input.value;invalidate();changed(candidate,requestOptions);}
    async function find(offset=0){clearTimeout(timer);if(!input.value.trim()||!box.isConnected||input.disabled)return;const version=++generation;button.disabled=true;try{
      const result=await ui.request('/resolve',{kind,query:input.value.trim(),context:{action:'order',group:'office'},limit:8,offset});
      if(version!==generation||!box.isConnected)return;list.replaceChildren();
      const candidates=result.outcome==='match'?[result.match]:result.candidates||[],rows=kind==='product'?logic.productChoices(candidates):candidates.filter(Boolean);status.textContent=rows.length?ui.t('Choose the exact record.','Elige el registro exacto.'):ui.t('No matches. Try the full name or code.','Sin resultados. Prueba el nombre completo o código.');
      for(const item of rows){
        const extra=kind==='product'&&item.kosher_tier==='extra_kosher',text=item.label||item.name;
        const b=ui.el('button',extra?text.replace(/\s*[—-]\s*EXTRA-KOSHER\s*(?:\(SS\))?/gi,'').trim():text,list);b.type='button';
        if(extra){b.className='extra-kosher-choice';const tag=ui.el('span',' EXTRA-KOSHER',b);tag.className='product-tier';}
        b.onclick=()=>select(item);
      }
      if(rows.length===8){const more=ui.el('button',ui.t('More matches','Más resultados'),list);more.type='button';more.onclick=()=>find(offset+8);}
      if(offset===0&&missing)missing({query:input.value.trim(),host:list});
    }catch(error){if(version===generation)ui.failure(error);}finally{if(version===generation)button.disabled=false;}}
    button.onclick=()=>find();input.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();find();}};
    return {input,select,clear,setDisabled(flag){input.disabled=flag;button.disabled=flag;for(const option of list.querySelectorAll?.('button')||list.children)option.disabled=flag;},get value(){return value;},id(){if(!value)throw Error(ui.t('Choose ','Elige ')+ui.t(en,es));return value.id;}};
  }
  function render({background=false}={}){
    invalidate();content.replaceChildren();
    if(!FLDashboard.writesEnabled)return;
    const available=logic.permitted(grants,true);
    if(!available.length){ui.el('p',ui.t('Sign in with permission for this action.','Inicia sesión con permiso para esta acción.'),content);return;}
    if(!available.some(([key])=>key===action))action=available[0][0];
    const starts=ui.el('nav',undefined,content);starts.className='office-start toolbar';starts.setAttribute('aria-label',ui.t('Start here','Empieza aquí'));
    for(const [key,en,es] of [['create_order','New order','Pedido nuevo'],['create_customer','New customer','Cliente nuevo'],['create_expected_receipt','Expected delivery','Entrega esperada']]){
      if(!available.some(([value])=>value===key))continue;
      const start=ui.el('button',ui.t(en,es),starts);start.type='button';start.dataset.officeStart=key;if(action===key){start.className='primary';start.setAttribute('aria-current','true');}start.onclick=()=>{action=key;render();};
    }
    const label=ui.el('label',ui.t('Office action','Acción de oficina'),content),selector=ui.el('select',undefined,label);selector.id='office-action';
    for(const [key,en,es] of available){ui.el('option',ui.t(en,es),selector).value=key;}
    selector.value=action;selector.onchange=()=>{action=selector.value;render();};
    const form=ui.el('form',undefined,content);form.id='office-form';form.className='panel';
    const fields=ui.el('div',undefined,form);fields.className='fields';let controls={},choices={},rows=[],orderGeneration=0,documentLoaded=null,expectedLoaded=null,readyLoaded=null,customerHost,customerController=null,customerGeneration=0,customerOpen=false,customerClose=null;
    const initialRead={background};
    function field(name,en,es,type='text',required=false,parent=fields,registry=controls){
      const label=ui.el('label',ui.t(en,es),parent),el=ui.el(type==='textarea'?'textarea':'input',undefined,label);el.name=name;if(type!=='textarea')el.type=type;el.required=required;if(type==='number')el.step='any';if(type==='checkbox')label.className='check';registry[name]=el;return el;
    }
    function select(name,en,es,items,parent=fields){
      const label=ui.el('label',ui.t(en,es),parent),el=ui.el('select',undefined,label);el.name=name;controls[name]=el;
      ui.el('option',ui.t('Choose','Elige'),el).value='';
      for(const [value,en,es] of items)ui.el('option',ui.t(en,es),el).value=value;
      return el;
    }
    function values(registry=controls){return Object.fromEntries(Object.entries(registry).map(([k,c])=>[k,c.type==='checkbox'?c.checked:c.type==='number'&&c.value!==''?Number(c.value):c.value]));}
    function noPO(){const check=field('no_po','No PO','Sin PO','checkbox');check.onchange=()=>{controls.customer_po.disabled=check.checked;if(check.checked)controls.customer_po.value='';invalidate();};}
    function notes(){field('notes','Notes — English','Notas — inglés','textarea');field('notes_es','Notes — Spanish','Notas — español','textarea');}
    function customerFields(parent=fields,registry=controls,name=''){
      field('name','Customer name','Nombre del cliente','text',true,parent,registry).value=name;
      field('contact_name','Contact name (optional)','Contacto (opcional)','text',false,parent,registry);
      field('email','Email (optional)','Correo (opcional)','email',false,parent,registry);
      field('phone','Phone (optional)','Teléfono (opcional)','tel',false,parent,registry);
      field('address','Address (optional)','Dirección (opcional)','textarea',false,parent,registry);
      field('notes','Customer notes — English (optional)','Notas del cliente — inglés (opcional)','textarea',false,parent,registry);
      field('notes_es','Customer notes — Spanish (optional)','Notas del cliente — español (opcional)','textarea',false,parent,registry);
    }
    function expectedQuantity(){
      const unit=select('expected_unit','Enter expected quantity in','Ingresa la cantidad esperada en',[['lb','Pounds (lb)','Libras (lb)'],['cases','Cases','Cajas']]);unit.value='lb';
      const poundsHost=ui.el('div',undefined,fields),pounds=field('expected_qty','Expected lb','lb esperadas','number',true,poundsHost);pounds.min='0.000001';
      const casesHost=ui.el('div',undefined,fields);casesHost.hidden=true;
      const cases=field('expected_cases','Cases','Cajas','number',false,casesHost),size=field('expected_case_size','Case size (lb each)','Tamaño de caja (lb cada una)','number',false,casesHost);cases.min=size.min='0.000001';
      const total=ui.el('p','',casesHost);total.setAttribute('role','status');total.className='neutral';
      function calculate(){try{pounds.value=logic.expectedPounds(Number(cases.value),Number(size.value));total.textContent=ui.t('Expected total: ','Total esperado: ')+pounds.value+' lb';}catch(_){pounds.value='';total.textContent=ui.t('Enter cases and lb per case to calculate the total.','Ingresa cajas y lb por caja para calcular el total.');}}
      cases.oninput=size.oninput=calculate;
      unit.onchange=()=>{const byCases=unit.value==='cases';casesHost.hidden=!byCases;poundsHost.hidden=byCases;pounds.disabled=byCases;pounds.required=!byCases;cases.required=size.required=byCases;if(byCases)calculate();invalidate();};
    }
    function addLine(seed={}){
      invalidate();const row=ui.el('fieldset',undefined,lineHost);ui.el('legend',ui.t('Order / document line','Línea de pedido / documento'),row);const inputs={};
      const product=picker(row,'product','Product (including Pallet Charge)','Producto (incluye cargo por palé)','product');
      const doc=action.startsWith('approve_');
      if(doc){
        field('description','Description from document','Descripción del documento','text',true,row,inputs).value=seed.customer_description||seed.vendor_description||seed.description||'';
        field('quantity_lb',action==='approve_er'?'Expected lb':'Reviewed order lb',action==='approve_er'?'lb esperadas':'lb revisadas','number',true,row,inputs).value=seed.quantity_lb||seed.expected_qty_lb||'';
        field('save_alias','Remember this product match','Recordar este producto','checkbox',false,row,inputs).checked=true;
        ui.el('p',ui.t('Check the extracted quantity against the source document; enter the reviewed pounds.','Verifica la cantidad extraída contra el documento; ingresa las libras revisadas.'),row);
        if(seed.quantity!==undefined)ui.el('p',ui.t('Extracted: ','Extraído: ')+seed.quantity+' '+(seed.unit||''),row);
      }else{
        field('quantity','Quantity','Cantidad','number',true,row,inputs).min='0.000001';
        const unitLabel=ui.el('label',ui.t('Unit','Unidad'),row),unit=ui.el('select',undefined,unitLabel);unit.name='unit';inputs.unit=unit;
        for(const [value,en,es] of [['cases','Cases','Cajas'],['lb','lb','lb'],['bags','Bags','Bolsas'],['boxes','Boxes','Cajas'],['each','Each (services / pallets)','Unidad (servicios / palés)'],['pouches','Pouches → cases','Bolsitas → cajas']])ui.el('option',ui.t(en,es),unit).value=value;
        ui.el('p',ui.t('Choose Pallet Charge from the catalog with unit Each. FL keeps service lines out of stock pounds.','Elige el cargo por palé del catálogo con unidad Unidad. FL excluye los servicios del peso de inventario.'),row);
      }
      if(action!=='approve_er')field('unit_price','Price per unit (optional)','Precio por unidad (opcional)','number',false,row,inputs);
      const conversion=ui.el('p','',row);conversion.className='neutral';conversion.hidden=true;
      const item={row,inputs,product,conversion};rows.push(item);
      const remove=ui.el('button',ui.t('Remove line','Quitar línea'),row);remove.type='button';remove.onclick=()=>{rows=rows.filter(r=>r!==item);row.remove();invalidate();};
    }
    let lineHost;
    const orderAction=['add_order_lines','update_order_line','cancel_order_line','update_order_header','update_order_status','mark_order_ready','close_order','cancel_order','reopen_order'].includes(action);
    if(orderAction){
      choices.order=picker(fields,'order','Sales order','Pedido de venta','order',async (selected,requestOptions={})=>{
        const generation=++orderGeneration;readyLoaded=null;if(!selected)return;
        try{const [order,options]=await Promise.all([ui.request('/sales/orders/'+selected.id,undefined,requestOptions),ui.request('/sales/orders/'+selected.id+'/office-options',undefined,requestOptions)]);
          if(generation!==orderGeneration||!form.isConnected)return;
          if(action==='mark_order_ready'){
            let mark=String(readyIntent?.order_id)===String(selected.id)?readyIntent:null;
            if(!mark){
              const params=new URLSearchParams({state:order.state,limit:'200'});if(order.customer)params.set('customer',order.customer);
              const result=await ui.request('/sales/orders?'+params,undefined,requestOptions);if(generation!==orderGeneration||!form.isConnected)return;
              mark=(result.orders||[]).find(row=>String(row.order_id)===String(selected.id));
              if(!mark||!Object.prototype.hasOwnProperty.call(mark,'note'))throw Error(ui.t('The ready-to-ship note could not be loaded. Choose the order again.','No se pudo cargar la nota de listo para enviar. Elige el pedido otra vez.'));
            }
            controls.ready.checked=Boolean(mark.ready);controls.note.value=mark.note||'';readyLoaded=selected.id;readyIntent=null;
          }
          if(controls.line_id){controls.line_id.replaceChildren();ui.el('option',ui.t('Choose line','Elige línea'),controls.line_id).value='';for(const line of order.lines||[])ui.el('option','#'+line.line_id+' · '+line.product+' · '+line.quantity_lb+' lb · '+line.line_status,controls.line_id).value=line.line_id;}
          if(controls.status&&action==='update_order_status'){controls.status.replaceChildren();for(const status of options.statuses||[])ui.el('option',ui.t(...(reasonText[status]||[status,status])),controls.status).value=status;}
          if(action==='close_order'){controls.reason.replaceChildren();for(const reason of options.close_reasons||[])ui.el('option',ui.t(...reasonText[reason]),controls.reason).value=reason;if(!options.close_reasons?.length)ui.notice(ui.t('FL offers no close reason for this order.','FL no ofrece un motivo de cierre para este pedido.'),'neutral');}
          if(action==='update_order_header'){for(const name of ['customer_po','requested_ship_date','notes','notes_es'])controls[name].value=order[name]||'';controls.no_po.checked=!order.customer_po;controls.customer_po.disabled=!order.customer_po;}
        }catch(error){ui.failure(error);}
      });
    }
    if(action==='create_customer')customerFields();
    if(['create_order','approve_so','update_order_header'].includes(action)){
      choices.customer=picker(fields,'customer',action==='update_order_header'?'New customer (optional)':'Customer',action==='update_order_header'?'Cliente nuevo (opcional)':'Cliente','customer',()=>{if(customerOpen)customerClose?.();},action==='create_order'&&grants.create_customer===true?offerCustomer:null);
      if(selectedCustomer){choices.customer.select(selectedCustomer);selectedCustomer=null;}
      if(action==='create_order'&&grants.create_customer===true){customerHost=ui.el('section',undefined,fields);customerHost.className='wide';}
      field('customer_po','Customer PO','PO del cliente','text',action==='approve_so');if(action!=='approve_so')noPO();
      field('requested_ship_date','Ship by','Enviar antes de','date');if(action!=='update_order_header')field('order_date','Order date','Fecha del pedido','date');notes();
    }
    if(['update_order_line','cancel_order_line'].includes(action)){
      select('line_id','Order line','Línea del pedido',[]).required=true;
      if(action==='update_order_line'){field('quantity_lb','New total ordered lb (optional)','Nuevas lb totales (opcional)','number');field('unit_price','New price per unit (optional)','Nuevo precio por unidad (opcional)','number');}
    }
    if(action==='mark_order_ready'){field('ready','Ready to ship','Listo para enviar','checkbox');field('note','Ready to ship note','Nota de listo para enviar','textarea');}
    if(action==='update_order_status')select('status','New status — FL choices','Estado nuevo — opciones FL',[]).required=true;
    if(['close_order','cancel_order'].includes(action)){
      const reasons=action==='close_order'?[]:['customer_cancelled','cns_declined','duplicate','superseded','other'];
      select('reason','Reason','Motivo',reasons.map(x=>[x,...reasonText[x]])).required=true;
      field('note','Reason note','Nota del motivo','textarea',true);
      if(action==='cancel_order')choices.related=picker(fields,'related','Related order (duplicate / replaced only)','Pedido relacionado (duplicado / reemplazado)','order');
    }
    if(action==='reopen_order')field('note','Reason note','Nota del motivo','textarea',true);
    if(['create_expected_receipt','approve_er'].includes(action)){
      choices.supplier=picker(fields,'supplier','Supplier','Proveedor','supplier');
      field('reference_number','PO / delivery reference','PO / referencia de entrega');field('expected_date','Expected date','Fecha esperada','date');
    }
    if(action==='create_expected_receipt'){choices.product=picker(fields,'product','Product','Producto','product');expectedQuantity();field('notes','Notes','Notas','textarea');}
    if(['update_expected_receipt','cancel_expected_receipt'].includes(action)){
      const input=field('expected_receipt_id','Expected delivery #','Entrega esperada #','number',true);input.step='1';
      const button=ui.el('button',ui.t('Load expected delivery','Cargar entrega esperada'),fields);button.type='button';const view=ui.el('div',undefined,fields);view.className='wide';
      input.oninput=()=>{expectedLoaded=null;view.replaceChildren();};
      button.onclick=async({background=false}={})=>{expectedLoaded=null;const id=input.value;try{const result=await ui.request('/expected-receipts/'+encodeURIComponent(id),undefined,{background});if(input.value!==id||!form.isConnected)return;expectedLoaded=Number(id);const record=result.expected_receipt||result;view.innerHTML=ui.draftHtml(record);for(const key of ['expected_qty','expected_date','reference_number','notes'])if(controls[key])controls[key].value=record[key]??'';}catch(error){ui.failure(error);}};
      if(action==='update_expected_receipt'){field('expected_qty','Expected lb','lb esperadas','number',true);field('expected_date','Expected date (blank clears)','Fecha esperada (vacío borra)','date');field('reference_number','Reference (blank clears)','Referencia (vacío borra)');field('notes','Notes (blank clears)','Notas (vacío borra)','textarea');select('status','Delivery state','Estado de entrega',[['','Keep open','Mantener abierta'],['closed','Close delivery','Cerrar entrega']]);}
      if(initial.get('expected')){input.value=initial.get('expected');initial.delete('expected');button.onclick(initialRead);}
    }
    if(['create_order','add_order_lines','approve_so','approve_er'].includes(action)){
      lineHost=ui.el('section',undefined,fields);lineHost.className='wide';lineHost.dataset.lines='';
      if(action.startsWith('approve_')){
        const input=field('document_id','Extracted document #','Documento extraído #','number',true);input.step='1';
        const load=ui.el('button',ui.t('Load extracted document','Cargar documento extraído'),fields);load.type='button';const status=ui.el('p','',fields);
        input.oninput=()=>{documentLoaded=null;status.textContent='';};
        load.onclick=async({background=false}={})=>{documentLoaded=null;const id=input.value;try{const data=await ui.request('/purchase-documents/'+encodeURIComponent(id),undefined,{background});if(input.value!==id||!form.isConnected)return;
          if(data.status!=='extracted'||data.document_kind!==(action==='approve_so'?'sales':'purchase'))throw Error(ui.t('Choose an extracted document of this type.','Elige un documento extraído de este tipo.'));
          const ex=data.extraction||{};documentLoaded=Number(id);status.textContent=data.original_filename||'#'+id;
          if(controls.customer_po)controls.customer_po.value=ex.po_number||'';if(controls.reference_number)controls.reference_number.value=ex.reference_number||ex.po_number||'';
          if(choices.customer)choices.customer.input.value=ex.customer_name||'';if(choices.supplier)choices.supplier.input.value=ex.supplier_name||'';
          rows=[];lineHost.replaceChildren();for(const row of ex.lines||[])addLine(row);if(!rows.length)addLine();
        }catch(error){ui.failure(error);}};
        if(initial.get('document')){input.value=initial.get('document');initial.delete('document');load.onclick(initialRead);}
      }else addLine();
      const add=ui.el('button',ui.t('Add line','Agregar línea'),fields);add.type='button';add.onclick=()=>addLine();
    }
    function offerCustomer({query,host}){
      if(!query)return;
      const separator=ui.el('hr',undefined,host);separator.className='new-customer-divider';
      ui.el('p',ui.t('Customer not listed? Create a separate customer record.','¿El cliente no está en la lista? Crea un registro de cliente aparte.'),host).className='muted';
      const create=ui.el('button',ui.t("Add '"+query+"' as a new customer","Agregar '"+query+"' como cliente nuevo"),host);create.type='button';create.className='new-customer-option secondary';create.dataset.newCustomer='';
      create.onclick=()=>openCustomer(query);
    }
    function openCustomer(name){
      invalidate();customerController?.invalidate();customerGeneration++;customerHost.replaceChildren();customerOpen=true;prepare.disabled=true;
      const editor=ui.el('div',undefined,customerHost);editor.dataset.customerEditor='';
      ui.el('h2',ui.t('New customer','Cliente nuevo'),editor);
      ui.el('p',ui.t('Review the customer first. After you press Record, we will select this customer and keep your order details.','Revisa el cliente primero. Después de presionar Registrar, seleccionaremos este cliente y conservaremos los detalles del pedido.'),editor);
      const customerInputs={};
      customerFields(editor,customerInputs,name);
      const review=ui.el('button',ui.t('Prepare customer','Preparar cliente'),editor);review.type='button';
      const cancel=ui.el('button',ui.t('Cancel new customer','Cancelar cliente nuevo'),editor);cancel.type='button';
      const customerDraft=ui.el('div',undefined,customerHost),suggestion=ui.el('div',undefined,customerHost);customerDraft.dataset.customerDraft='';
      function close(){customerGeneration++;customerController?.invalidate();customerController=null;customerOpen=false;customerClose=null;choices.customer.setDisabled(false);customerHost.replaceChildren();prepare.disabled=false;}
      customerClose=close;cancel.onclick=close;
      editor.scrollIntoView({block:'nearest'});customerInputs.name.focus?.();
      function showExisting(result){
        suggestion.replaceChildren();const existing=logic.existingCustomer(result);if(!existing)return;
        if(existing.active===false){ui.el('p',ui.t('This customer is inactive. Ask an owner to review it before placing an order.','Este cliente está inactivo. Pide al propietario que lo revise antes de hacer un pedido.'),suggestion);return;}
        const use=ui.el('button',ui.t('Use existing customer: ','Usar cliente existente: ')+existing.name,suggestion);use.type='button';use.onclick=()=>{close();choices.customer.select(existing);};
      }
      function edit(){customerGeneration++;customerController?.invalidate();customerController=null;customerDraft.replaceChildren();suggestion.replaceChildren();editor.hidden=false;choices.customer.setDisabled(false);}
      review.onclick=()=>{
        if(!Object.values(customerInputs).every(input=>input.reportValidity()))return;
        const request=logic.build('create_customer',values(customerInputs)),version=++customerGeneration;
        customerController?.invalidate();customerDraft.replaceChildren();suggestion.replaceChildren();editor.hidden=true;choices.customer.setDisabled(true);
        customerController=ui.ticketCard(customerDraft,request.path,request.body,{receiptPrefix:request.prefix,allowPhoto:false,
          onEdit:edit,onCancel:close,onBusy:flag=>{busy(flag);prepare.disabled=flag||customerOpen;},
          onPrepared:result=>{if(version===customerGeneration&&form.isConnected)showExisting(result);},
          onFailure:error=>{if(version===customerGeneration&&form.isConnected)showExisting(error.result);},
          onRecorded:result=>{
            if(version!==customerGeneration||!form.isConnected)return;
            const customer=logic.customerChoice(result);if(!customer){ui.notice(ui.t('Customer recorded. Open the receipt to check the customer, then search for it above.','Cliente registrado. Abre el recibo para verificar el cliente y búscalo arriba.'));choices.customer.setDisabled(false);customerOpen=false;prepare.disabled=false;return;}
            customerController=null;customerOpen=false;choices.customer.setDisabled(false);choices.customer.select(customer);prepare.disabled=false;suggestion.replaceChildren();editor.remove();
            ui.el('p',ui.t('Customer selected. Continue preparing your order below.','Cliente seleccionado. Continúa preparando el pedido abajo.'),customerHost);
          }
        });
      };
    }
    const prepare=ui.el('button',action==='create_customer'?ui.t('Prepare customer','Preparar cliente'):ui.t('Prepare draft','Preparar borrador'),form);prepare.type='submit';prepare.className='primary';
    form.addEventListener('input',invalidate);
    const busy=flag=>{for(const start of starts.children)start.disabled=flag;selector.disabled=flag;document.getElementById('language').disabled=flag;document.getElementById('sign-out').disabled=flag;prepare.disabled=flag||customerOpen;};
    form.onsubmit=async e=>{
      e.preventDefault();if(customerOpen){ui.notice(ui.t('Finish or cancel the new customer first.','Termina o cancela el cliente nuevo primero.'));return;}invalidate();const version=revision;busy(true);
      try{
        const v=values(),conversions=[];
        if(action==='create_expected_receipt'&&v.expected_unit==='cases')v.expected_qty=logic.expectedPounds(v.expected_cases,v.expected_case_size);
        for(const name of ['customer','supplier','product','order'])if(choices[name]?.value)v[name+'_id']=choices[name].id();
        if(choices.related?.value)v.related_so_id=choices.related.id();
        if(action==='mark_order_ready'&&readyLoaded!==v.order_id)throw Error(ui.t('Choose and load the sales order first.','Elige y carga el pedido primero.'));
        if(controls.document_id&&documentLoaded!==Number(v.document_id))throw Error(ui.t('Load and review the extracted document first.','Carga y revisa el documento primero.'));
        if(controls.expected_receipt_id&&expectedLoaded!==Number(v.expected_receipt_id))throw Error(ui.t('Load the expected delivery first.','Carga la entrega esperada primero.'));
        if(lineHost){
          if(!rows.length)throw Error(ui.t('Add at least one line.','Agrega al menos una línea.'));
          v.lines=[];
          for(const item of rows){
            const value=values(item.inputs),pid=item.product.id();
            if(action==='approve_so')v.lines.push({product_id:pid,quantity_lb:value.quantity_lb,customer_description:value.description,save_alias:value.save_alias,...(value.unit_price!==''?{unit_price:value.unit_price}:{})});
            else if(action==='approve_er')v.lines.push({product_id:pid,expected_qty_lb:value.quantity_lb,vendor_description:value.description,save_alias:value.save_alias});
            else{
              const result=await ui.request('/resolve',{kind:'unit',query:value.unit,quantity:value.quantity,context:{product_id:pid,action:'order',group:'office'}});
              if(version!==revision)return;
              const line=logic.resolvedLine(result,pid,value.quantity,value.unit,value.unit_price!==''?{unit_price:value.unit_price}:{});
              if(result.conversion){conversions.push(result.conversion.from_quantity+' '+ui.t('pouches → ','bolsitas → ')+line.quantity+' '+ui.t('cases','cajas'));item.conversion.hidden=false;item.conversion.textContent=result.conversion.from_quantity+' '+ui.t('pouches → ','bolsitas → ')+line.quantity+' '+ui.t('cases','cajas');}
              v.lines.push(line);
            }
          }
        }
        if(version!==revision)return;
        const request=logic.build(action,v);form.hidden=true;for(const conversion of conversions)ui.el('p',conversion,draft).className='neutral';
        controller=ui.ticketCard(draft,request.path,request.body,{receiptPrefix:request.prefix,allowPhoto:false,onBusy:busy,
          onEdit:()=>{invalidate();form.hidden=false;},onCancel:()=>render(),
          onPrepared:result=>{if(action==='create_customer')offerExistingCustomer(result);},
          onFailure:error=>{if(action==='create_customer')offerExistingCustomer(error.result);},
          onRecorded:()=>{const another=ui.el('button',ui.t('Another office action','Otra acción de oficina'),draft);another.type='button';another.onclick=render;}
        });draft.scrollIntoView({block:'start'});
      }catch(error){ui.failure(error);}finally{busy(false);}
    };
    function offerExistingCustomer(result){
      if(!form.isConnected)return;
      const existing=logic.existingCustomer(result);if(!existing||existing.active===false||grants.create_order!==true)return;
      const use=ui.el('button',ui.t('Use existing customer: ','Usar cliente existente: ')+existing.name,draft);use.type='button';
      use.onclick=()=>{selectedCustomer=existing;action='create_order';render();};
    }
    if(orderAction&&initial.get('order')){const id=initial.get('order');initial.delete('order');ui.request('/sales/orders/'+encodeURIComponent(id),undefined,initialRead).then(o=>{if(form.isConnected)choices.order.select({id:o.order_id,label:o.order_number},initialRead);}).catch(ui.failure);}
  }
  async function refresh({background=false}={}){const version=++authRevision;grants={};render({background});if(!FLDashboard.writesEnabled||!ui.session.token)return;
    try{const who=await ui.request('/auth/whoami',undefined,{background});if(version!==authRevision)return;grants=who.permissions||{};render({background});}catch(error){ui.failure(error);}
  }
  window.addEventListener('ledger-language',()=>{ui.relabel();todayLink();});
  window.addEventListener('ledger-auth',refresh);window.addEventListener('ledger-signout',refresh);
  ui.ready.then(on=>{if(on)refresh({background:true});}).catch(()=>{});
})();
