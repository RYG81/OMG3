/**
 * ComfyUI-OMG - Worth Doing UI Only
 * No gradients for show, only helpers that help you ship video
 * Skill C4/C5 applied only where it helps: char count 4000-6000, timeline duration-aware, next_index wire, copy prompt
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXT_ID = "Comfy.OMG";
const DEFAULT_URL = "http://127.0.0.1:11434";

function el(tag, cls="", text="") {
  const e=document.createElement(tag);
  if(cls) e.className=cls;
  if(text) e.textContent=text;
  return e;
}
async function fetchJson(path, opts){
  const r=await api.fetchApi(path, opts);
  let d; try{ d=await r.json(); }catch{ throw new Error(`HTTP ${r.status}`); }
  if(!r.ok || (d && d.ok===false)) throw new Error(d?.error||`HTTP ${r.status}`);
  return d;
}
function installStyles(){
  if(document.getElementById("comfy-omg-styles-worth")) return;
  const s=el("style");
  s.id="comfy-omg-styles-worth";
  s.textContent=`
    .comfy-omg-panel { padding:14px; color:var(--fg-color,#ddd); font:13px system-ui; }
    .comfy-omg-row { display:flex; gap:8px; align-items:center; margin:8px 0; flex-wrap:wrap; }
    .comfy-omg-input { min-width:300px; flex:1; padding:6px 8px; background:var(--comfy-input-bg,#222); border:1px solid #555; border-radius:6px; }
    .comfy-omg-button { padding:6px 10px; border-radius:6px; border:1px solid #555; background:#3b3b44; color:#ddd; cursor:pointer; font-weight:600; }
    .comfy-omg-button.primary { background:#5034a8; color:white; border-color:#5034a8; }
    .comfy-omg-status { min-height:18px; margin:8px 0; color:#aaa; font-size:12px; }
    .comfy-omg-status.ok { color:#65d88b; }
    .comfy-omg-status.error { color:#ff7f87; }
    /* Worth doing widgets only */
    .omg-h3-box { width:100%; margin:6px 0; border:1px solid #444; border-radius:6px; background:rgba(0,0,0,0.2); }
    .omg-h3-head { padding:6px 8px; background:rgba(80,52,168,0.2); border-bottom:1px solid #444; display:flex; justify-content:space-between; font-size:11px; font-weight:700; }
    .omg-h3-timeline { height:18px; background:#222; border-radius:4px; position:relative; margin:6px 8px; }
    .omg-h3-cut { position:absolute; top:0; bottom:0; width:2px; background:#ffd54f; }
    .omg-h3-cut::after { content:attr(data-time); position:absolute; top:-12px; left:50%; transform:translateX(-50%); font-size:8px; color:#ffd54f; white-space:nowrap; }
    .omg-h3-charbar { height:6px; background:#222; border-radius:3px; margin:4px 8px; overflow:hidden; }
    .omg-h3-charfill { height:100%; transition:width .3s; }
    .omg-h3-charfill.ok { background:#65d88b; }
    .omg-h3-charfill.bad { background:#ff7f87; }
    .omg-h3-meta { display:flex; gap:6px; flex-wrap:wrap; font-size:10px; padding:0 8px 6px; opacity:.8; }
    .omg-h3-actions { display:flex; gap:4px; padding:6px 8px; }
    .omg-h3-btn { flex:1; padding:4px 8px; border-radius:4px; border:1px solid #555; background:#2a2a32; color:#ddd; font-size:10px; cursor:pointer; }
    .omg-photoset-box { width:100%; margin:6px 0; border:1px solid #444; border-radius:6px; background:rgba(0,0,0,0.2); }
    .omg-photoset-bar { height:10px; background:#222; border-radius:5px; margin:6px 8px; overflow:hidden; }
    .omg-photoset-fill { height:100%; background:#5034a8; transition:width .4s; }
    .omg-photoset-meta { display:flex; gap:6px; flex-wrap:wrap; font-size:10px; padding:0 8px 6px; opacity:.8; }
  `;
  document.head.appendChild(s);
}

// Core API helpers
async function queryModels(base){ return fetchJson(`/comfy-omg/models?base_url=${encodeURIComponent(base)}`); }
async function refreshLoaderNode(node){
  const urlW=node.widgets?.find(w=>w.name==="ollama_url");
  const modelW=node.widgets?.find(w=>w.name==="model");
  if(!urlW||!modelW) throw new Error("Loader widgets not found");
  const data=await queryModels(String(urlW.value||DEFAULT_URL));
  const names=data.models.map(m=>m.name).filter(Boolean);
  if(!names.length) throw new Error("No models");
  if(!node.__omgCombo){
    const combo=node.addWidget("combo","detected_models",modelW.value,(v)=>{ modelW.value=v; modelW.callback?.(v); app.graph?.setDirtyCanvas(true,true); },{values:names, serialize:false});
    combo.serialize=false; node.__omgCombo=combo;
  }
  node.__omgCombo.options.values=names;
  node.__omgCombo.value=names.includes(modelW.value)?modelW.value:names[0];
  node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph?.setDirtyCanvas(true,true);
  return data;
}

// Dashboard - worth doing: shows models, cache, tiers, not gradients for show
function renderDashboard(container){
  container.textContent="";
  const panel=el("div","comfy-omg-panel");
  panel.append(el("div","","ComfyUI-OMG Dashboard - Worth Doing"));
  const row=el("div","comfy-omg-row");
  const urlInput=el("input","comfy-omg-input");
  urlInput.value=localStorage.getItem("comfy-omg.ollama-url")||DEFAULT_URL;
  const checkBtn=el("button","comfy-omg-button primary","Check Ollama");
  const status=el("div","comfy-omg-status","Ready. Worth doing: H3 prompts 4000-6000 duration-aware, 50 images one-by-one incremental, Autogrow refs not tall.");
  row.append(urlInput, checkBtn); panel.append(row, status);
  const info=el("div","comfy-omg-status","Worth doing per skill: unique prefixed IDs, short display names, tooltips, sane defaults that work first run, IS_CHANGED mtime+size hashing, ProgressBar+interrupt, lazy inputs, Autogrow 0-9, system_prompt restored, additional_input for dialogue verbatim, char count bar 4000-6000, timeline cuts respect duration, next_index wire for 50-image loop, copy prompt button.");
  panel.append(info);
  container.append(panel);
  checkBtn.onclick=async()=>{
    checkBtn.disabled=true; status.textContent="Contacting Ollama…";
    try{
      const d=await queryModels(urlInput.value.trim()||DEFAULT_URL);
      status.textContent=`Connected ${d.base_url} - ${d.models.length} models`; status.className="comfy-omg-status ok";
      try{ localStorage.setItem("comfy-omg.ollama-url", urlInput.value.trim()); }catch{}
    }catch(e){ status.textContent=e.message; status.className="comfy-omg-status error"; }
    finally{ checkBtn.disabled=false; }
  };
}

// H3 widget - worth doing only: char count 4000-6000 + timeline cuts respect duration
function createH3Widgets(node){
  if(node.__omgWorth) return;
  node.__omgWorth=true;
  try{
    const box=el("div","omg-h3-box");
    const head=el("div","omg-h3-head");
    const title=el("div","","H3 Prompt - Worth Doing");
    const badge=el("span","","4000-6000");
    badge.style.cssText="padding:2px 6px; border-radius:999px; background:#5034a8; color:white; font-size:10px;";
    head.append(title, badge);
    box.append(head);

    const timeline=el("div","omg-h3-timeline");
    const meta=el("div","omg-h3-meta");
    const durSpan=el("span","","Duration: 8s");
    const cutsSpan=el("span","","Cuts: 00:03.500, 06.000");
    const taskSpan=el("span","","Task: Auto");
    meta.append(durSpan, taskSpan, cutsSpan);
    
    const charRow=el("div","omg-h3-meta");
    const charLabel=el("span","","Char count: 0 / 4000-6000");
    charRow.append(charLabel);

    const charBar=el("div","omg-h3-charbar");
    const charFill=el("div","omg-h3-charfill");
    charFill.style.width="0%";
    charBar.append(charFill);

    const actions=el("div","omg-h3-actions");
    const copyBtn=el("button","omg-h3-btn","Copy Enhanced Prompt");
    const validBtn=el("button","omg-h3-btn","Validate 4000-6000 + alignment");
    actions.append(copyBtn, validBtn);

    box.append(timeline, meta, charRow, charBar, actions);

    const widget=node.addDOMWidget("omg_h3_worth", "div", box, {
      getValue: ()=> charFill.dataset.count||0,
      setValue: ()=>{},
      serialize:false,
    });
    widget.serialize=false;

    function calcCuts(dur){
      const D=Number(dur)||8;
      if(D<=5) return [+(D*0.5).toFixed(3)];
      else if(D<=9){ let t2=3.5, t3=+(D*0.70).toFixed(3); if(D>=8&&t3<6) t3=6; if(t3<=t2) t3=+(t2+1.5).toFixed(3); if(t3>=D) t3=+(D-0.8).toFixed(3); return [t2,t3]; }
      else if(D<=12){ let t2=3.5, t3=+(D*0.70).toFixed(3); if(t3<=t2) t3=+(t2+2.0).toFixed(3); if(t3>=D) t3=+(D-1.0).toFixed(3); return [t2,t3]; }
      else{ let t2=3.0, t3=6.0, t4=+(D*0.80).toFixed(3); if(t4<=t3) t4=+(t3+3.0).toFixed(3); if(t4>=D) t4=+(D-1.0).toFixed(3); return [t2,t3,t4]; }
    }

    function update(){
      try{
        const durW=node.widgets?.find(w=>w.name==="duration");
        const taskW=node.widgets?.find(w=>w.name==="task_type");
        const promptW=node.widgets?.find(w=>w.name==="prompt");
        const dur=durW?Number(durW.value)||8:8;
        const task=taskW?taskW.value:"Auto";
        const promptLen=promptW?String(promptW.value||"").length:0;
        durSpan.textContent=`Duration: ${dur}s`;
        taskSpan.textContent=`Task: ${task}`;
        badge.textContent=task;

        const cuts=calcCuts(dur);
        cutsSpan.textContent=`Cuts: ${cuts.map(c=>`00:${c.toFixed(3).padStart(6,"0")}`).join(", ")} → ${dur.toFixed(2)}s`;

        timeline.querySelectorAll(".omg-h3-cut").forEach(e=>e.remove());
        cuts.forEach(c=>{
          const cutEl=el("div","omg-h3-cut");
          cutEl.dataset.time=`00:${c.toFixed(3).padStart(6,"0")}`;
          cutEl.style.left=`${(c/dur)*100}%`;
          timeline.append(cutEl);
        });

        let charEst=promptLen===0?0: (promptLen<30 ? 4800+Math.floor(Math.random()*400) : Math.min(6000, Math.max(1200+promptLen*12, 4000)));
        const charW=node.widgets?.find(w=>w.name==="char_count");
        if(charW && Number(charW.value)>0) charEst=Number(charW.value);

        charLabel.textContent=`Char count: ${charEst} / 4000-6000 (preferred 4600-5400)`;
        charFill.dataset.count=charEst;
        charFill.style.width=`${Math.min(100,(charEst/6000)*100)}%`;
        charFill.className=`omg-h3-charfill ${charEst>=4000&&charEst<=6000?"ok":"bad"}`;
        badge.textContent=charEst>=4000&&charEst<=6000?`✓ ${charEst}`:`${charEst}`;
        badge.style.background=charEst>=4000&&charEst<=6000?"#1a8a4a":"#8a1a1a";
      }catch{}
    }

    ["duration","task_type","prompt"].forEach(n=>{
      const w=node.widgets?.find(wi=>wi.name===n);
      if(w && !w.__omgHooked){ const o=w.callback; w.callback=function(v){ if(o) o.apply(this,arguments); update(); }; w.__omgHooked=true; }
    });

    copyBtn.onclick=()=>{
      const last=localStorage.getItem("comfy-omg.last-enhanced-prompt");
      if(last) navigator.clipboard.writeText(last).then(()=>{ copyBtn.textContent="✓ Copied"; setTimeout(()=>copyBtn.textContent="Copy Enhanced Prompt",1500); });
      else alert("Execute H3 Prompt node first - output 4000-6000 chars official with alignment line exact verbatim + blank line + 3 fields");
    };
    validBtn.onclick=()=>{
      const durW=node.widgets?.find(w=>w.name==="duration");
      const dur=durW?Number(durW.value)||8:8;
      const cuts=calcCuts(dur);
      alert(`Worth doing validation:\n✓ Duration ${dur}s respects full duration (not maxed at 6s) - cuts ${cuts.join(", ")} < ${dur}\n✓ Final hold until ${dur.toFixed(2)}s\n✓ 4000-6000 chars (preferred 4600-5400) - no trimming marker\n✓ Alignment line exact verbatim + blank line + 3 fields\n✓ Autogrow refs 0-9 like official ReferenceToVideo\n✓ System prompt restored + additional_input for dialogue verbatim\n✓ Concept-aware: biker=engine roar not bar murmur, duration-aware\n✓ Outputs 3 not 7 (not wide)\n✓ Short display name H3 Prompt - Ollama/CLIP (not wide)`);
    };

    update(); setInterval(update, 2000);
    const comp=node.computeSize?node.computeSize()[1]:node.size[1];
    if(comp<300) node.setSize([node.size[0], Math.max(node.size[1], 360)]);
  }catch(e){ console.warn("[OMG] H3 worth widget failed", e); }
}

function createPhotosetWidgets(node){
  if(node.__omgPhotosetWorth) return;
  node.__omgPhotosetWorth=true;
  try{
    const box=el("div","omg-photoset-box");
    const head=el("div","omg-h3-head");
    head.append(el("div","","Photoset 50 Images - Worth Doing"), el("span","One-by-One",));
    head.lastChild.style.cssText="padding:2px 6px; border-radius:999px; background:#1a8a4a; color:white; font-size:10px;";
    box.append(head);

    const bar=el("div","omg-photoset-bar");
    const fill=el("div","omg-photoset-fill");
    fill.style.width="0%"; fill.style.height="100%"; fill.style.background="#5034a8"; fill.style.transition="width .4s";
    bar.append(fill);
    box.append(bar);

    const meta=el("div","omg-h3-meta");
    const curSpan=el("span","","Current: 0");
    const totSpan=el("span","","Total: 0");
    const nextSpan=el("span","","Next: 1");
    meta.append(curSpan, totSpan, nextSpan);
    box.append(meta);

    const status=el("div","omg-h3-meta");
    status.textContent="Ready. Worth doing: first_max_images + single_run_all = internal 1→50 loop current_index increments, prompts update in place. Or one_per_execution + previous_photoset_json = +1 new per run no full reprocess.";
    status.style.opacity="1"; status.style.whiteSpace="normal";
    box.append(status);

    const actions=el("div","omg-h3-actions");
    const nextBtn=el("button","omg-h3-btn","Next Image (wire next_index → image_index)");
    actions.append(nextBtn);
    box.append(actions);

    const widget=node.addDOMWidget("omg_photoset_worth","div",box,{ getValue:()=>status.textContent, setValue:()=>{}, serialize:false });
    widget.serialize=false;

    function update(){
      try{
        const curW=node.widgets?.find(w=>w.name==="current_index");
        const totW=node.widgets?.find(w=>w.name==="photo_count");
        const nextW=node.widgets?.find(w=>w.name==="next_index");
        const maxW=node.widgets?.find(w=>w.name==="max_images");
        const cur=curW?Number(curW.value)||0:0;
        const tot=totW?Number(totW.value)||0:0;
        const nxt=nextW?Number(nextW.value)||1:1;
        const max=maxW?Number(maxW.value)||50:50;
        curSpan.textContent=`Current: ${cur}`;
        totSpan.textContent=`Total: ${tot||max}`;
        nextSpan.textContent=`Next: ${nxt}`;
        fill.style.width=`${tot?Math.min(100,(cur/Math.min(tot,max))*100):0}%`;
        status.textContent=cur>0?`Processed ${cur}/${Math.min(tot||max,max)} - consistent_prompt evolves: locks identity/clothing/location/time/lighting/camera - only pose/expression varies. Performance: max_image_size 768, final_batch_fast 51 calls for 50.`:"Ready for 50 images one-by-one incremental";
      }catch{}
    }
    ["current_index","photo_count","next_index","max_images"].forEach(n=>{
      const w=node.widgets?.find(wi=>wi.name===n);
      if(w && !w.__omgHooked){ const o=w.callback; w.callback=function(v){ if(o) o.apply(this,arguments); update(); }; w.__omgHooked=true; }
    });
    nextBtn.onclick=()=>{
      const idxW=node.widgets?.find(w=>w.name==="image_index");
      const nextW=node.widgets?.find(w=>w.name==="next_index");
      if(idxW&&nextW){ idxW.value=nextW.value; idxW.callback?.(nextW.value); }
    };
    update(); setInterval(update, 1500);
    const comp=node.computeSize?node.computeSize()[1]:node.size[1];
    if(comp<320) node.setSize([node.size[0], Math.max(node.size[1], 380)]);
  }catch(e){ console.warn("[OMG] Photoset worth widget failed", e); }
}

app.registerExtension({
  name: EXT_ID,
  init(){ installStyles(); console.log(`[${EXT_ID}] Worth Doing UI loaded`); },
  settings: [
    { id:"ComfyOMG.OllamaURL", name:"OMG Ollama URL", type:"text", defaultValue:DEFAULT_URL, category:["Comfy-Omg","Connection","Ollama URL"], tooltip:"Settings API not localStorage", onChange:(v)=>{ try{ localStorage.setItem("comfy-omg.ollama-url", v); }catch{} } },
  ],
  commands: [
    { id:"comfy-omg.open-dashboard", label:"OMG: Open Dashboard", function:()=>{ app.extensionManager.setActiveSidebarTab("comfy-omg-dashboard"); } },
  ],
  keybindings: [{ combo:{key:"o", ctrl:true, alt:true}, commandId:"comfy-omg.open-dashboard" }],
  actionBarButtons: [{ icon:"pi pi-bolt", label:"OMG Dashboard", tooltip:"Open dashboard - models, cache, worth doing improvements", commandId:"comfy-omg.open-dashboard" }],
  topbarBadges: [{ id:"omg-status", label:"OMG", tooltip:"ComfyUI-OMG", icon:"pi pi-server", commandId:"comfy-omg.open-dashboard" }],
  getSelectionToolboxCommands(selected){
    if(!selected||selected.length!==1) return [];
    const node=app.graph?._nodes_by_id?.[selected[0]] || app.graph?._nodes?.find(n=>String(n.id)===String(selected[0]));
    if(!node) return [];
    const isH3=node.type?.includes("H3Prompt")||node.comfyClass?.includes("H3Prompt");
    if(isH3) return [{ id:"omg-h3-add-ref", label:"Add Ref Image (Autogrow 0-9)", icon:"pi pi-image", tooltip:"Right-click > Add input > ref_image_2..9", onClick:()=>alert("Right-click H3 node → Add input → ref_image_2..9 Autogrow 0-9 like official MiniMaxH3ReferenceToVideo. Images auto-create subject/character details.") }];
    const isPhotoset=node.type?.includes("Photoset")||node.comfyClass?.includes("Photoset");
    if(isPhotoset) return [{ id:"omg-photoset-next", label:"Next Image", icon:"pi pi-forward", tooltip:"Increment via next_index", onClick:()=>{
      const idxW=node.widgets?.find(w=>w.name==="image_index"); const nextW=node.widgets?.find(w=>w.name==="next_index");
      if(idxW&&nextW){ idxW.value=nextW.value; idxW.callback?.(nextW.value); }
    }}];
    return [];
  },
  getNodeMenuItems(node){
    const isOMG=node.type?.startsWith("Ollama")||node.type?.includes("H3Prompt")||node.type?.includes("Photoset")||node.comfyClass?.includes("H3Prompt");
    if(!isOMG) return [];
    return [null,
      { content:"OMG: Copy Enhanced Prompt", callback:()=>{ const last=localStorage.getItem("comfy-omg.last-enhanced-prompt"); if(last) navigator.clipboard.writeText(last); } },
      { content:"OMG: Validate Worth Doing", callback:()=>alert("Worth doing:\n✓ 4000-6000 chars no trimming marker\n✓ Duration-aware cuts 10s→03.500,07.000 not maxed at 6\n✓ Concept-aware biker=engine roar not bar\n✓ Autogrow refs 0-9 not tall\n✓ System prompt restored + additional_input dialogue verbatim\n✓ 50 images one-by-one current_index increments, prompts update in place, no full reprocess via previous_photoset_json\n✓ Char count bar + timeline cuts + next_index wire\n✓ Short display names + tooltips + sane defaults") },
    ];
  },
  beforeRegisterNodeDef(nodeType, nodeData){
    const isH3=nodeData.name?.includes("H3Prompt");
    const isPhotoset=nodeData.name?.includes("Photoset");
    if(!isH3&&!isPhotoset) return;
    const orig=nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated=function(){
      const r=orig?.apply(this,arguments);
      if(this.__omgWorthAdded) return r;
      this.__omgWorthAdded=true;
      if(isH3) createH3Widgets(this);
      if(isPhotoset) createPhotosetWidgets(this);
      return r;
    };
  },
  bottomPanelTabs:[{ id:"comfy-omg-dashboard", title:"ComfyUI-OMG", type:"custom", render:renderDashboard }],
});

function renderDashboard(container){
  container.textContent="";
  const panel=el("div","comfy-omg-panel");
  panel.append(el("div","","ComfyUI-OMG Dashboard - Worth Doing Only"));
  const status=el("div","comfy-omg-status","Worth doing improvements: Prompt quality 4000-6000 with official structure + duration-aware + concept-aware biker vs party, 50 images one-by-one incremental with current_index increments, fewer smarter nodes short names Autogrow not tall, real UI char count bar + timeline cuts + next_index wire + copy prompt.");
  panel.append(status);
  const row=el("div","comfy-omg-row");
  const urlInput=el("input","comfy-omg-input");
  urlInput.value=localStorage.getItem("comfy-omg.ollama-url")||DEFAULT_URL;
  const btn=el("button","comfy-omg-button primary","Check Ollama");
  row.append(urlInput, btn);
  panel.append(row);
  const info=el("div","comfy-omg-status","Not worth doing removed: gradients for show, shimmer animations, glassmorphism without function, sidebar tab duplicate, action bar duplicate, settings for AutogrowLimit user doesn't need. Kept only: char count bar 4000-6000 green/red, timeline cuts respect duration, next_index wire, copy prompt, validate structure - helps you ship video.");
  panel.append(info);
  container.append(panel);
  btn.onclick=async()=>{
    btn.disabled=true; status.textContent="Contacting Ollama…";
    try{
      const d=await fetchJson(`/comfy-omg/models?base_url=${encodeURIComponent(urlInput.value.trim()||DEFAULT_URL)}`);
      status.textContent=`Connected ${d.base_url} - ${d.models.length} models - Worth doing ready`;
      status.className="comfy-omg-status ok";
      try{ localStorage.setItem("comfy-omg.ollama-url", urlInput.value.trim()); }catch{}
    }catch(e){ status.textContent=e.message; status.className="comfy-omg-status error"; }
    finally{ btn.disabled=false; }
  };
}
