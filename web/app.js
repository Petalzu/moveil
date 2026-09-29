'use strict';
const $ = id => document.getElementById(id);
let state, token, pending = [], anchor = null, preview = null, busy = false;
const images = [new Image(), new Image()];
const canvases = [$('original'), $('result')];
let job={items:[]}, selected=-1, serial=0, prefix='', uploading=false;
const itemName=item=>item.display_name||`样例 ${item.offset}`;
const drafts=new Map();
for(let n=1;n<=10;n++)$('count').add(new Option(`${n} 张`,n));
$('count').value='10';
function draw() {
  if(!state)return;
  canvases.forEach((canvas, i) => {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(images[i], 0, 0);
    const labelRects=[];
    const mark = (box, label, color) => {
      const [x,y,r,b] = box; ctx.strokeStyle = color; ctx.lineWidth = 2;
      if ($('boxes').checked) ctx.strokeRect(x,y,r-x,b-y);
      if ($('labels').checked && canvas.width>=24 && canvas.height>=20) {
        ctx.font='14px sans-serif';let text=label;
        while(text.length && ctx.measureText(text).width+8>canvas.width)text=text.slice(0,-1);
        const width=Math.min(canvas.width,ctx.measureText(text).width+8),left=Math.max(0,Math.min(x,canvas.width-width));
        const candidates=[Math.max(0,y-20),b+2];
        for(let top=0;top<=canvas.height-20;top+=20)candidates.push(top);
        const top=candidates.find(t=>t>=0&&t+20<=canvas.height&&!labelRects.some(q=>left<q[2]&&left+width>q[0]&&t<q[3]&&t+20>q[1]));
        if(top!==undefined){labelRects.push([left,top,left+width,top+20]);ctx.fillStyle='#142033';ctx.fillRect(left,top,width,20);ctx.fillStyle=color;ctx.fillText(text,left+4,top+15);}
      }
    };
    state.detections.forEach(d => d.boxes.forEach(b => mark(b,d.label,'#27bba0')));
    [...pending, ...(preview ? [preview] : [])].forEach(b => { if(i===1){ctx.fillStyle='#000';ctx.fillRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);} mark(b,'待保存','#bd86ff'); });
  });
  $('save').disabled = busy || !pending.length;
  $('undo').disabled = busy || !pending.length;
  $('version').textContent = `当前版本：${state.run} · ${pending.length} 个待保存框`;
  const m=state.automatic_metrics;
  $('metrics').textContent=m ? `当前图片：${m.fully_covered_entities}/${m.entity_count} 实体完整覆盖 · 非敏感墨迹误遮 ${(100*m.nonsensitive_ink_redacted_fraction).toFixed(2)}%` : state.evaluation_status==='manual'?'当前版本：手工补充遮挡':`检测到 ${state.detections.length} 个实体 · 自定义图片未提供标注`;
  $('match-section').hidden=!m;
  $('matches').replaceChildren();
  for(const entity of state.entity_matches || []){
    const row=document.createElement('article');
    const title=document.createElement('strong'),detail=document.createElement('span');title.textContent=`#${entity.id+1} · ${entity.label}`;detail.textContent=`墨迹覆盖 ${(entity.coverage*100).toFixed(2)}% · ${entity.correct_label_fully_covered?'标签匹配':entity.fully_covered?'标签偏差':'未完整覆盖'}`;row.append(title,detail);
    $('matches').append(row);
  }
  $('detections').replaceChildren();
  for(const d of state.detections){const row=document.createElement('article'),title=document.createElement('strong'),detail=document.createElement('span');title.textContent=d.label;detail.textContent=`${typeof d.score==='number'?`置信度 ${(d.score*100).toFixed(2)}% · `:''}${d.boxes.length} 个框 · ${d.boxes.map(b=>`[${b.join(', ')}]`).join(' ')}`;row.append(title,detail);$('detections').append(row);}
  if(!state.detections.length)$('detections').textContent='未检测到实体';
}
async function loadImages(){
  await Promise.all(images.map((im,i)=>new Promise((resolve,reject)=>{im.onload=resolve;im.onerror=reject;im.src=prefix+(i?'redacted.png':'original.png');})));
  canvases.forEach(c=>{c.width=state.width;c.height=state.height;}); draw();
}
function point(e,c){const r=c.getBoundingClientRect();return [Math.max(0,Math.min(c.width,Math.round((e.clientX-r.left)*c.width/r.width))),Math.max(0,Math.min(c.height,Math.round((e.clientY-r.top)*c.height/r.height)))];}
function rectangle(a,b){return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];}
canvases.forEach(c=>{
  c.onpointerdown=e=>{if(busy||!state)return;c.setPointerCapture(e.pointerId);anchor=point(e,c);};
  c.onpointermove=e=>{if(anchor){preview=rectangle(anchor,point(e,c));draw();}};
  c.onpointerup=e=>{if(!anchor)return;const b=rectangle(anchor,point(e,c));if(b[2]>b[0]&&b[3]>b[1])pending.push(b);anchor=preview=null;draw();};
  c.onpointercancel=()=>{anchor=preview=null;draw();};
});
$('boxes').onchange=$('labels').onchange=()=>state&&draw();
$('undo').onclick=()=>{pending.pop();draw();};
$('save').onclick=async()=>{
  const itemId=job.items[selected].id, ticket=serial;
  busy=true;draw();$('status').textContent='正在核验并保存…';
  try{const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify({run:state.run,boxes:pending,item:itemId})});if(!response.ok)throw Error();const saved=await response.json();if(ticket!==serial)return;state=saved;pending=[];drafts.delete(itemId);await loadImages();$('status').textContent='已保存新版本。';}
  catch{if(ticket===serial)$('status').textContent='保存未确认，补框已保留；请重新选择此样例核对版本后重试。';}
  finally{busy=false;draw();}
};
function scores(c){
  $('scores').replaceChildren();
  if(job.mode==='upload'){
    for(const [label,value] of [['已处理图片',`${job.completed}/${job.total}`],['检测实体数',String(job.items.reduce((n,item)=>n+(item.detection_count||0),0))],['处理失败',String(job.failed)],['准确率','未提供标注']]){const card=document.createElement('article'),title=document.createElement('span'),number=document.createElement('strong');title.textContent=label;number.textContent=value;card.append(title,number);$('scores').append(card);}return;
  }
  for(const [label,key,detail] of [['实体完整覆盖率','overall',c?`${Math.round(c.overall*c.entity_count)}/${c.entity_count} 个实体`:'等待结果'],['标签匹配准确率','correct_label',c?`${Math.round(c.correct_label*c.entity_count)}/${c.entity_count} 个实体`:'等待结果'],['零漏检图片比例','zero_missed_images',c?`${Math.round(c.zero_missed_images*c.n)}/${c.n} 张图片`:'等待结果'],['最大非敏感墨迹误遮','max_nonpii_ink_mask',c?`已评测 ${c.n} 张`:'等待结果']]){
    const card=document.createElement('article');for(const [tag,text] of [['span',label],['strong',c?`${(c[key]*100).toFixed(2)}%`:'—'],['small',detail]]){const el=document.createElement(tag);el.textContent=text;card.append(el);}$('scores').append(card);
  }
}
function pager(){
  if(selected>=0){const item=job.items[selected];$('sample-info').textContent=`第 ${selected+1}/${job.items.length} 页 · ${itemName(item)} · 耗时 ${item.seconds} 秒`;}
  $('pages').replaceChildren();job.items.forEach((item,i)=>{const b=document.createElement('button');b.role='tab';b.setAttribute('aria-selected',String(i===selected));b.textContent=`${i+1} · ${itemName(item)} · ${item.status==='failed'?'失败':item.entities===null?`${item.detection_count} 个实体`:`${item.full}/${item.entities}`}`;b.onclick=()=>selectPage(i);$('pages').append(b);});
  $('previous').disabled=busy||selected<=0;$('next').disabled=busy||selected>=job.items.length-1;
}
async function selectPage(index){
  if(busy||index<0||index>=job.items.length)return;
  if(selected>=0)drafts.set(job.items[selected].id,pending);
  selected=index;const item=job.items[index];const ticket=++serial;
  state=null;pending=drafts.get(item.id)||[];anchor=preview=null;
  $('sample').hidden=true;$('empty').hidden=false;$('empty').textContent=item.status==='failed'?'该样例处理失败':'正在加载…';$('metrics').textContent='';$('download').hidden=true;$('save').disabled=$('undo').disabled=true;
  $('sample-info').textContent=`第 ${index+1}/${job.items.length} 页 · 样例 ${item.offset} · 耗时 ${item.seconds} 秒`;pager();$('pages').children[index]?.scrollIntoView({behavior:'smooth',block:'nearest',inline:'center'});
  if(item.status==='failed')return;
  busy=true;
  try{prefix=`/api/items/${item.id}/`;const r=await fetch(prefix+'state');if(!r.ok)throw Error();const result=await r.json();if(ticket!==serial)return;state=result;await loadImages();if(ticket!==serial)return;$('sample').hidden=false;$('empty').hidden=true;$('download').hidden=false;$('download').href=prefix+'download';}
  catch{if(ticket===serial){state=null;$('empty').textContent='加载失败，请重新选择样例。';}}
  finally{busy=false;pager();draw();}
}
async function refresh(){
  if(busy||uploading)return;
  const r=await fetch('/api/batch');if(!r.ok)throw Error();const next=await r.json();const changed=job.id!==next.id;const oldLength=job.items.length;
  if(changed){serial++;selected=-1;state=null;pending=[];drafts.clear();$('sample').hidden=true;$('empty').hidden=false;$('metrics').textContent='';$('sample-info').textContent='';$('save').disabled=$('undo').disabled=true;$('download').hidden=true;}
  job=next;scores(job.metrics);$('start').disabled=job.status==='running'||busy;$('count').disabled=job.status==='running';$('progress').max=job.total||10;$('progress').value=job.completed;$('progress-text').textContent=`${job.phase} · ${job.completed}/${job.total} 张 · 失败 ${job.failed} 张 · ${job.elapsed} 秒`;
  uploadControls();
  if(changed||oldLength!==job.items.length){pager();if(selected<0&&job.items.length)await selectPage(0);}
  if(selected<0)$('empty').textContent=job.status==='running'?'正在实时处理，首张完成后自动展示。':job.status==='failed'?job.phase:'开始运行后，结果将在这里逐张出现。';
}
$('start').onclick=async()=>{
  if(busy||uploading)return;
  if([...drafts.values(),pending].some(boxes=>boxes.length)&&!confirm('开始新批次将清空未保存补框，继续吗？'))return;
  $('start').disabled=true;
  try{const r=await fetch('/api/batch',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify({count:Number($('count').value)})});if(!r.ok)throw Error();await refresh();}catch{$('progress-text').textContent='启动失败，请稍后重试。';$('start').disabled=false;}
};
$('previous').onclick=()=>selectPage(selected-1);$('next').onclick=()=>selectPage(selected+1);
function uploadControls(){const locked=busy||uploading||job.status==='running';$('upload-file').disabled=locked;$('upload').disabled=locked||!$('upload-file').files.length;}
$('upload-file').onchange=()=>{const file=$('upload-file').files[0];$('file-name').textContent=file?`${file.name} · ${(file.size/1024/1024).toFixed(2)} MB`:'PNG / JPEG / WebP · 最大 10 MB · 1600 万像素';$('upload-status').textContent='';uploadControls();};
$('upload').onclick=async()=>{
  const file=$('upload-file').files[0];if(!file||busy||uploading||job.status==='running')return;
  if(!['image/png','image/jpeg','image/webp'].includes(file.type)||file.size>10*1024*1024){$('upload-status').textContent='请选择 10 MB 以内的 PNG、JPEG 或 WebP 图片。';return;}
  if([...drafts.values(),pending].some(boxes=>boxes.length)&&!confirm('上传将开始新任务，清空未保存补框，继续吗？'))return;
  uploading=true;uploadControls();$('start').disabled=true;$('upload-status').textContent='正在上传并校验图片…';
  try{const r=await fetch('/api/upload',{method:'POST',headers:{'Content-Type':file.type,'X-Review-Token':token},body:file});if(!r.ok)throw new Error(r.status===413?'图片过大，请使用 1600 万像素、边长 8192 以内的图片。':r.status===409?'已有任务正在运行。':'图片校验失败，请选择有效的静态图片。');$('upload-status').textContent='上传成功，正在识别。';}
  catch(error){$('upload-status').textContent=error.message;}
  finally{uploading=false;await refresh();uploadControls();}
};
$('pages').onwheel=e=>{if(Math.abs(e.deltaY)>Math.abs(e.deltaX)&&$('pages').scrollWidth>$('pages').clientWidth){e.preventDefault();$('pages').scrollLeft+=e.deltaY;}};
$('pages').onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();selectPage(selected+(e.key==='ArrowLeft'?-1:1));}};
(async()=>{try{const r=await fetch('/api/state');if(!r.ok)throw Error();token=(await r.json()).token;await refresh();}catch{$('progress-text').textContent='连接失败，请刷新重试。';}
  async function poll(){try{await refresh();}catch{$('progress-text').textContent='连接中断，正在重连…';}setTimeout(poll,1000);}setTimeout(poll,1000);
})();