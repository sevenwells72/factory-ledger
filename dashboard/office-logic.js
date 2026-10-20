/* Routes and request shapes only. FL validates quantities, conversions and permissions. */
(function(root){
  'use strict';
  const actions=[
    ['create_order','Create order','Crear pedido'],
    ['create_customer','New customer','Cliente nuevo'],
    ['add_order_lines','Add order lines','Agregar líneas'],
    ['update_order_line','Update order line','Modificar línea'],
    ['cancel_order_line','Cancel order line','Cancelar línea'],
    ['update_order_header','Update order header','Modificar encabezado'],
    ['update_order_status','Update order status','Cambiar estado'],
    ['mark_order_ready','Ready to ship and note','Listo para enviar y nota'],
    ['close_order','Close order','Cerrar pedido'],
    ['cancel_order','Cancel order','Cancelar pedido'],
    ['reopen_order','Reopen order','Reabrir pedido'],
    ['create_expected_receipt','Create expected delivery','Crear entrega esperada'],
    ['update_expected_receipt','Update expected delivery','Modificar entrega esperada'],
    ['cancel_expected_receipt','Cancel expected delivery','Cancelar entrega esperada'],
    ['approve_so','Approve extracted customer PO','Aprobar PO de cliente extraída'],
    ['approve_er','Approve extracted supplier document','Aprobar documento de proveedor extraído']
  ];
  const permission=action=>({approve_so:'create_order',approve_er:'create_expected_receipt'}[action]||action);
  const permitted=(grants,on)=>on===true?actions.filter(([a])=>grants?.[permission(a)]===true):[];
  const id=value=>{if(!Number.isSafeInteger(Number(value))||Number(value)<=0)throw Error('Choose a record / Elige un registro');return Number(value);};
  function build(action,v){
    const body={client_source:'dashboard'},copy=names=>names.forEach(k=>{if(v[k]!==undefined&&v[k]!=='')body[k]=v[k];});
    let path,prefix='ORD',order;
    if(['add_order_lines','update_order_line','cancel_order_line','update_order_header','update_order_status','mark_order_ready','close_order','cancel_order','reopen_order'].includes(action))order='/sales/orders/'+id(v.order_id);
    if(action==='create_customer'){path='/customers/prepare';prefix='CUS';body.name=v.name;copy(['contact_name','email','phone','address','notes','notes_es']);}
    if(action==='create_order'){path='/sales/orders/prepare';body.customer_id=id(v.customer_id);body.customer_po=v.no_po?null:(v.customer_po||null);copy(['requested_ship_date','order_date','notes','notes_es']);body.lines=v.lines;}
    if(action==='add_order_lines'){path=order+'/lines/prepare';body.lines=v.lines;}
    if(action==='update_order_line'){path=order+'/lines/'+id(v.line_id)+'/update/prepare';copy(['quantity_lb','unit_price']);}
    if(action==='cancel_order_line')path=order+'/lines/'+id(v.line_id)+'/cancel/prepare';
    if(action==='update_order_header'){path=order+'/header/prepare';body.customer_po=v.no_po?null:(v.customer_po||null);for(const k of ['requested_ship_date','notes','notes_es'])if(v[k]!==undefined)body[k]=v[k]||null;if(v.customer_id)body.customer_id=id(v.customer_id);}
    if(action==='update_order_status'){path=order+'/status/prepare';body.status=v.status;}
    if(action==='mark_order_ready'){path=order+'/ready/prepare';body.ready=Boolean(v.ready);body.note=v.note||null;}
    if(['close_order','cancel_order','reopen_order'].includes(action)){path=order+'/'+({close_order:'close',cancel_order:'cancel',reopen_order:'reopen'}[action])+'/prepare';copy(['reason','note']);if(v.related_so_id)body.related_so_id=id(v.related_so_id);}
    if(action.includes('expected_receipt')){
      prefix='ER';
      if(action==='create_expected_receipt'){path='/expected-receipts/prepare';body.product_id=id(v.product_id);body.supplier_id=id(v.supplier_id);copy(['expected_qty','expected_date','reference_number','notes']);}
      else {path='/expected-receipts/'+id(v.expected_receipt_id)+(action==='cancel_expected_receipt'?'/cancel/prepare':'/update/prepare');if(action==='update_expected_receipt'){copy(['expected_qty','status']);for(const k of ['expected_date','reference_number','notes'])if(v[k]!==undefined)body[k]=v[k]||null;}}
    }
    if(action==='approve_so'){path='/sales/orders/extract/approve/prepare';body.document_id=id(v.document_id);body.customer_id=id(v.customer_id);body.customer_po=v.customer_po;body.lines=v.lines;copy(['order_date','requested_ship_date','notes']);}
    if(action==='approve_er'){prefix='ER';path='/expected-receipts/extract/approve/prepare';body.document_id=id(v.document_id);body.supplier_id=id(v.supplier_id);body.lines=v.lines;copy(['reference_number','expected_date']);}
    if(!path)throw Error('Unsupported action');
    return {path,body,prefix};
  }
  function resolvedLine(result,product,quantity,unit,extras={}){
    const d=result?.draft;
    if(result?.outcome!=='match'||!d||d.product_id!==product||!Number.isFinite(d.quantity)||!['lb','cases','bags','boxes','each'].includes(d.unit))throw Error(result?.ask||result?.note||'Resolve the unit / Resuelve la unidad');
    if(unit!=='pouches'&&(d.quantity!==quantity||d.unit!==unit))throw Error('FL returned a different quantity / FL devolvió otra cantidad');
    return {...d,...extras};
  }
  function customerChoice(value){
    const customer=value?.customer||value;
    const customerId=customer?.id??customer?.customer_id;
    if(!Number.isSafeInteger(customerId)||customerId<=0||typeof customer?.name!=='string'||!customer.name.trim())return null;
    return {id:customerId,name:customer.name,label:customer.name,...(typeof customer.active==='boolean'?{active:customer.active}:{})};
  }
  function existingCustomer(result){
    const detail=result?.detail||result;
    return customerChoice(detail?.existing_customer||(detail?.blockers||[]).find(b=>b.code==='CUSTOMER_ALREADY_EXISTS')?.existing_customer);
  }
  function productChoices(rows){
    return rows.filter(Boolean).map((item,index)=>({item,index})).sort((a,b)=>Number(a.item.kosher_tier==='extra_kosher')-Number(b.item.kosher_tier==='extra_kosher')||a.index-b.index).map(({item})=>item);
  }
  function expectedPounds(cases,caseSize){
    if(!Number.isFinite(cases)||cases<=0||!Number.isFinite(caseSize)||caseSize<=0)throw Error('Enter cases and lb per case / Ingresa cajas y lb por caja');
    const pounds=cases*caseSize;
    if(!Number.isFinite(pounds)||pounds<=0)throw Error('Enter a valid expected weight / Ingresa un peso esperado válido');
    return Number(pounds.toFixed(6));
  }
  const api={actions,permission,permitted,build,resolvedLine,customerChoice,existingCustomer,productChoices,expectedPounds};
  if(typeof module!=='undefined')module.exports=api;root.FLOffice=api;
})(typeof window==='undefined'?globalThis:window);
