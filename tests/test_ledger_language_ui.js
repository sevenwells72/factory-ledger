const {test}=require('node:test');
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.hidden=false;}
 append(node){node.parent=this;this.children.push(node);}
 replaceChildren(){for(const node of this.children)node.parent=null;this.children=[];}
 set textContent(value){this.replaceChildren();this.append({nodeValue:String(value),get isConnected(){return !!this.parent?.isConnected;}});}
 get textContent(){return this.children.map(node=>node.nodeValue??node.textContent).join('');}
 get isConnected(){return this.root||!!this.parent?.isConnected;}
}
function harness(){
 const body=new Element('body');body.root=true;const notice=new Element('div'),identity=new Element('p'),language=new Element('button');body.append(notice);body.append(identity);body.append(language);
 const nodes={notice,identity,language};const document={documentElement:{},createElement:tag=>new Element(tag),createTextNode:value=>({nodeValue:value,get isConnected(){return !!this.parent?.isConnected;}}),getElementById:id=>nodes[id],querySelectorAll:()=>[]};
 const storage={getItem(){return null;}},root={document,sessionStorage:storage,fetch(){},FLDashboard:{base:'http://127.0.0.1',ready:Promise.resolve(false),accessMessage:lang=>lang==='es'?'Sin acceso':'Access unavailable'}};
 vm.runInNewContext(fs.readFileSync('dashboard/ledger-ui.js','utf8'),{window:root,document,sessionStorage:storage,FLKeySessions:{create:()=>({clear(){}})},setTimeout,clearTimeout});return {ui:root.FLLedger,body,notice,document};
}
test('real shared bilingual text setter relabels replacement statuses and preserves neighboring controls',()=>{
 const h=harness(),status=h.ui.el('p',h.ui.t('Searching…','Buscando…'),h.body),label=h.ui.el('label',h.ui.t('Physical tag','Etiqueta física'),h.body),input=h.ui.el('input',undefined,label);input.value='OATS-001';
 h.ui.lang='es';h.ui.translate();assert.equal(status.textContent,'Buscando…');assert.equal(label.textContent,'Etiqueta física');assert.equal(input.value,'OATS-001');
 h.ui.setText(status,h.ui.t('Selected: OATS-001','Seleccionado: OATS-001'));h.ui.lang='en';h.ui.translate();assert.equal(status.textContent,'Selected: OATS-001');assert.equal(label.textContent,'Physical tag');assert.equal(input.value,'OATS-001');
 h.ui.setText(status,'');h.ui.lang='es';h.ui.translate();assert.equal(status.textContent,'');
});
test('real shared notice and server failure messages refresh both languages',()=>{
 const h=harness();h.ui.notice(h.ui.t('Photo attached','Foto adjunta'),'good');h.ui.lang='es';h.ui.translate();assert.equal(h.notice.textContent,'Foto adjunta');assert.equal(h.notice.className,'good');
 h.ui.failure({result:{detail:{message:'Choose the physical lot',message_es:'Elige el lote físico'}}});assert.equal(h.notice.textContent,'NO registrado — Elige el lote físico');h.ui.lang='en';h.ui.translate();assert.equal(h.notice.textContent,'NOT recorded — Choose the physical lot');
});

test('local validation notices preserve both languages when an Error is created',()=>{
 const h=harness();h.ui.failure(h.ui.problem(h.ui.t('Choose Product','Elige Producto')));assert.equal(h.notice.textContent,'NOT recorded — Choose Product');h.ui.lang='es';h.ui.translate();assert.equal(h.notice.textContent,'NO registrado — Elige Producto');
});
