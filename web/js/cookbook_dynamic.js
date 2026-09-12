/**
 * Cookbook Dynamic Inputs - JS Widget that changes inputs based on style_selection dropdown
 * One node, 66 different input requirements - as user changes dropdown, we change inputs
 * Uses technology as much as possible to make user life easy
 * 
 * C4: addDOMWidget with getValue/setValue/serialize, min_size guard, idempotency
 * C5: api.fetchApi for /comfy-omg/cookbook/styles and /comfy-omg/cookbook/style/{slug}
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function el(tag, cls="", text="") {
  const e=document.createElement(tag);
  if(cls) e.className=cls;
  if(text) e.textContent=text;
  return e;
}

async function fetchJson(path) {
  const r=await api.fetchApi(path);
  let d; try{ d=await r.json(); }catch{ throw new Error(`HTTP ${r.status}`); }
  if(!r.ok || (d && d.ok===false)) throw new Error(d?.error||`HTTP ${r.status}`);
  return d;
}

function installStyles(){
  if(document.getElementById("cookbook-dynamic-styles")) return;
  const s=el("style");
  s.id="cookbook-dynamic-styles";
  s.textContent=`
    .omg-cookbook-dynamic { width:100%; margin:8px 0; border:1px solid #444; border-radius:8px; background:rgba(0,0,0,0.25); }
    .omg-cookbook-header { padding:8px 10px; background:rgba(80,52,168,0.2); border-bottom:1px solid #444; display:flex; justify-content:space-between; align-items:center; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:.5px; }
    .omg-cookbook-body { padding:10px; max-height:400px; overflow-y:auto; }
    .omg-cookbook-var { margin:8px 0; }
    .omg-cookbook-var-label { font-size:11px; font-weight:700; margin-bottom:4px; display:flex; justify-content:space-between; }
    .omg-cookbook-var-label span.desc { font-weight:400; opacity:.7; font-size:10px; max-width:65%; text-align:right; }
    .omg-cookbook-var-input { width:100%; padding:6px 8px; background:var(--comfy-input-bg,#222); border:1px solid #555; border-radius:6px; color:inherit; font-size:12px; }
    .omg-cookbook-var-input:focus { border-color:#8a4de8; outline:none; box-shadow:0 0 0 2px rgba(138,77,232,0.25); }
    .omg-cookbook-var-textarea { min-height:50px; resize:vertical; }
    .omg-cookbook-actions { display:flex; gap:6px; padding:8px 10px; border-top:1px solid #333; }
    .omg-cookbook-btn { flex:1; padding:5px 10px; border-radius:6px; border:1px solid #555; background:#2a2a32; color:#ddd; font-size:10px; font-weight:700; cursor:pointer; }
    .omg-cookbook-btn:hover { background:#3a3a44; border-color:#8a4de8; }
    .omg-cookbook-status { font-size:10px; opacity:.7; padding:6px 10px; background:rgba(120,120,140,.08); border-radius:0 0 8px 8px; }
  `;
  document.head.appendChild(s);
}

function createDynamicWidget(node){
  if(node.__cookbookDynamicAdded) return;
  node.__cookbookDynamicAdded=true;

  try{
    const wrapper=el("div","omg-cookbook-dynamic");
    const header=el("div","omg-cookbook-header");
    header.append(el("div","","Dynamic Inputs (changes on dropdown)"), el("div","","66 Styles"));
    header.lastChild.style.cssText="padding:2px 6px; border-radius:999px; background:#5034a8; color:white; font-size:10px;";
    wrapper.append(header);

    const body=el("div","omg-cookbook-body");
    body.textContent="Loading styles… Select a style from dropdown above, inputs will change here automatically to show exactly what that style needs.";
    wrapper.append(body);

    const actions=el("div","omg-cookbook-actions");
    const autoFillBtn=el("button","omg-cookbook-btn","Auto-Fill All from Concept");
    const clearBtn=el("button","omg-cookbook-btn","Clear All");
    actions.append(autoFillBtn, clearBtn);
    wrapper.append(actions);

    const status=el("div","omg-cookbook-status","Worth doing: One node handles 66 different input requirements via JS widget. As you change style_selection dropdown, this widget changes inputs to show exactly what that style needs. Technology as much as possible.");
    wrapper.append(status);

    // Hidden JSON widget that Python reads - stores dynamic vars
    let hiddenWidget = node.widgets?.find(w=>w.name==="dynamic_vars_json");
    if(!hiddenWidget){
      // Create hidden widget if not exists (for classic API)
      hiddenWidget = node.addWidget("text","dynamic_vars_json","{}",null,{multiline:true});
      hiddenWidget.type="customtext";
      hiddenWidget.hidden=true;
      // Hide visually but keep in widgets list for serialization
      hiddenWidget.computeSize=()=> [0, -4];
    }

    // Main DOM widget
    const domWidget=node.addDOMWidget("omg_cookbook_dynamic", "div", wrapper, {
      getValue: () => {
        // Serialize all dynamic inputs to JSON for hidden widget
        const data={};
        body.querySelectorAll("[data-var-name]").forEach(input=>{
          const varName=input.dataset.varName;
          data[varName]=input.value;
        });
        const jsonStr=JSON.stringify(data);
        if(hiddenWidget) {
          hiddenWidget.value=jsonStr;
          hiddenWidget.callback?.(jsonStr);
        }
        return jsonStr;
      },
      setValue: (v)=>{
        try{
          const data=typeof v==="string" ? JSON.parse(v) : v;
          if(hiddenWidget) hiddenWidget.value=JSON.stringify(data);
          // Populate inputs if they exist
          body.querySelectorAll("[data-var-name]").forEach(input=>{
            const varName=input.dataset.varName;
            if(data[varName]!==undefined) input.value=data[varName];
          });
        }catch{}
      },
      serialize: true, // serialize so hidden JSON is saved
    });
    domWidget.serialize=false; // DOM widget itself not serialized, hidden widget is

    // Cache for styles list
    let stylesCache=null;
    let currentStyleData=null;

    async function loadStylesList(){
      if(stylesCache) return stylesCache;
      try{
        const result=await fetchJson("/comfy-omg/cookbook/styles");
        if(result.ok && result.styles){
          stylesCache=result.styles;
          return stylesCache;
        }
      }catch(e){
        console.warn("[Cookbook Dynamic] Failed to fetch styles list via API, using fallback from dropdown options", e);
      }
      return null;
    }

    async function loadStyleDetail(slug){
      try{
        // Try API detail
        const result=await fetchJson(`/comfy-omg/cookbook/style/${encodeURIComponent(slug)}`);
        if(result.ok && result.style){
          return result.style;
        }
      }catch(e){
        console.warn(`[Cookbook Dynamic] Failed to fetch style detail for ${slug} via API`, e);
      }
      // Fallback: try to get from stylesCache
      if(stylesCache){
        const found=stylesCache.find(s=>s.slug===slug || s.display===slug || s.name===slug || slug.includes(s.slug) || s.display.includes(slug));
        if(found && found.variables_detail){
          // Reconstruct minimal style data from cached list
          return {
            style_slug: found.slug,
            style_name: found.name,
            style_summary: found.summary,
            environment_variables: found.variables_detail,
            prompt_template: found.prompt_template || "Create a {ASPECT_RATIO} image in {style_name} style. Subject: {SUBJECT}",
            negative_prompt: "",
            style_fidelity_anchors: [],
          };
        }
      }
      return null;
    }

    function createInputsForStyle(styleData){
      const envVars=styleData.environment_variables || {};
      body.textContent="";

      if(Object.keys(envVars).length===0){
        body.textContent="This style has no environment_variables - ready to use with concept only.";
        return;
      }

      // Create input for each variable that is not auto-filled (ASPECT_RATIO, STYLE_FIDELITY_ANCHORS, SOURCE_CONTENT_TO_AVOID are auto)
      const autoVars=["ASPECT_RATIO","STYLE_FIDELITY_ANCHORS","SOURCE_CONTENT_TO_AVOID"];
      const editableVars=Object.entries(envVars).filter(([k])=>!autoVars.includes(k));

      // Info
      const info=el("div","");
      info.style.fontSize="10px"; info.style.opacity=".7"; info.style.marginBottom="8px";
      info.textContent=`This style needs ${Object.keys(envVars).length} variables (${editableVars.length} editable, ${autoVars.length} auto). Fill all from concept or edit manually. Technology auto-fills from concept but you can override each field here.`;
      body.append(info);

      // Get concept for auto-fill
      const conceptW=node.widgets?.find(w=>w.name==="concept");
      const concept=conceptW?String(conceptW.value||""):"";
      const aspectW=node.widgets?.find(w=>w.name==="aspect_ratio");
      const aspect=aspectW?String(aspectW.value||""):"16:9";

      // Create inputs
      editableVars.forEach(([varName, varDesc])=>{
        const varDiv=el("div","omg-cookbook-var");
        const labelRow=el("div","omg-cookbook-var-label");
        labelRow.append(el("span","",varName), el("span","desc",varDesc.slice(0,80)));
        varDiv.append(labelRow);

        // Choose input type based on variable name
        let input;
        if(varName.includes("MAIN_TEXT") || varName.includes("SUBJECT") || varName.includes("LOCATION") || varName.includes("BACKGROUND") || varName.includes("DESCRIPTION")){
          input=el("textarea","omg-cookbook-var-input omg-cookbook-var-textarea");
          input.rows=2;
        }else{
          input=el("input","omg-cookbook-var-input");
          input.type="text";
        }
        input.dataset.varName=varName;
        input.placeholder=varDesc;

        // Auto-fill from concept heuristics if concept provided
        if(concept){
          const cLower=concept.toLowerCase();
          if(varName.includes("SUBJECT")){
            input.value=concept.slice(0,100);
          }else if(varName.includes("MAIN_TEXT")){
            const words=concept.split().slice(0,6);
            if(words.length>=3) input.value=`${words[0].toLowerCase()}\n${words[1].toLowerCase()}\n${words[2].toLowerCase()}.`;
            else input.value="focus\noutlasts\nnoise.";
          }else if(varName.includes("SECONDARY")){
            input.value="studio log 02:14";
          }else if(varName.includes("ACCENT")||varName.includes("SYMBOL")){
            input.value="a tiny white plus";
          }else if(varName.includes("WARDROBE")){
            if(cLower.includes("biker")) input.value="black leather jacket, helmet, gloves";
            else if(cLower.includes("architect")) input.value="dark work jacket over plain black shirt";
            else input.value="dark plain clothing, structured jacket";
          }else if(varName.includes("LOCATION")){
            if(cLower.includes("biker")||cLower.includes("new york")) input.value="busy New York city road with dense traffic";
            else if(cLower.includes("architect")) input.value="dim concrete studio after midnight";
            else input.value="dim studio, late-night workspace, minimal interior";
          }else if(varName.includes("BACKGROUND")){
            input.value="soft charcoal wall gradient, blurred wall texture, deep negative space";
          }else if(varName.includes("PRODUCT")||varName.includes("PROP")){
            if(cLower.includes("architect")) input.value="a rolled plan tube and a pencil held low";
            else input.value="minimal prop relevant to concept or no prop";
          }
        }

        // On change, update hidden JSON
        input.addEventListener("input", ()=>{
          const data={};
          body.querySelectorAll("[data-var-name]").forEach(inp=>{
            data[inp.dataset.varName]=inp.value;
          });
          const jsonStr=JSON.stringify(data);
          if(hiddenWidget){
            hiddenWidget.value=jsonStr;
            hiddenWidget.callback?.(jsonStr);
          }
          // Also update node's size
          node.setSize([node.size[0], node.computeSize()[1]]);
        });

        varDiv.append(input);
        body.append(varDiv);
      });

      // Initial serialization to hidden widget
      const data={};
      body.querySelectorAll("[data-var-name]").forEach(inp=>{
        data[inp.dataset.varName]=inp.value;
      });
      const jsonStr=JSON.stringify(data);
      if(hiddenWidget){
        hiddenWidget.value=jsonStr;
        hiddenWidget.callback?.(jsonStr);
      }

      // Update size
      node.setSize([node.size[0], node.computeSize()[1]]);
      app.graph?.setDirtyCanvas(true,true);

      status.textContent=`Loaded style ${styleData.style_name||styleData.style_slug} with ${Object.keys(envVars).length} variables (${editableVars.length} editable). Auto-filled from concept "${concept.slice(0,60)}". Edit each field above manually if needed - technology makes life easy, you only typed few words concept.`;
    }

    async function onStyleSelectionChange(){
      try{
        const styleW=node.widgets?.find(w=>w.name==="style_selection");
        if(!styleW) return;
        const selection=String(styleW.value||"");
        if(!selection || selection.startsWith("auto-infer")) {
          body.textContent="Auto-infer mode - will pick best style for concept. Select a specific style from dropdown to see its required inputs change here.";
          return;
        }

        body.textContent=`Loading style ${selection}…`;
        // Extract slug from display like "Mono Noir Type Portrait Poster Style (mono-noir-type-portrait-poster-style)"
        let slug=selection;
        const match=selection.match(/\(([^)]+)\)\s*$/);
        if(match) slug=match[1];

        // Load styles list first if not cached (for fallback)
        await loadStylesList();
        const styleData=await loadStyleDetail(slug);
        if(!styleData){
          body.textContent=`Failed to load style ${slug} - check /comfy-omg/cookbook/styles API route. Styles dir has 66 json files.`;
          return;
        }

        createInputsForStyle(styleData);

      }catch(e){
        body.textContent=`Error loading style: ${e.message}`;
        console.error("[Cookbook Dynamic] Error", e);
      }
    }

    // Hook style_selection dropdown change
    const styleWidget=node.widgets?.find(w=>w.name==="style_selection");
    if(styleWidget && !styleWidget.__cookbookHooked){
      const orig=styleWidget.callback;
      styleWidget.callback=function(v){
        if(orig) orig.apply(this, arguments);
        onStyleSelectionChange();
      };
      styleWidget.__cookbookHooked=true;
    }

    // Hook concept change to auto-fill
    const conceptWidget=node.widgets?.find(w=>w.name==="concept");
    if(conceptWidget && !conceptWidget.__cookbookHooked){
      const orig=conceptWidget.callback;
      conceptWidget.callback=function(v){
        if(orig) orig.apply(this, arguments);
        // Re-fill editable vars from new concept if they are still default
        onStyleSelectionChange();
      };
      conceptWidget.__cookbookHooked=true;
    }

    autoFillBtn.onclick=()=>{ onStyleSelectionChange(); };
    clearBtn.onclick=()=>{
      body.querySelectorAll("[data-var-name]").forEach(inp=>{ inp.value=""; });
      const data={};
      body.querySelectorAll("[data-var-name]").forEach(inp=>{ data[inp.dataset.varName]=inp.value; });
      const jsonStr=JSON.stringify(data);
      if(hiddenWidget){ hiddenWidget.value=jsonStr; hiddenWidget.callback?.(jsonStr); }
    };

    // Initial load
    loadStylesList().then(()=> onStyleSelectionChange());

    const computed=node.computeSize?node.computeSize()[1]:node.size[1];
    if(computed<400) node.setSize([node.size[0], Math.max(node.size[1], 500)]);

  }catch(e){ console.warn("[Cookbook Dynamic] Widget failed", e); }
}

app.registerExtension({
  name: "Comfy.OMG.CookbookDynamic",
  init(){ installStyles(); console.log("[Cookbook Dynamic] Loaded - one node handles 66 different input requirements via JS widget"); },
  beforeRegisterNodeDef(nodeType, nodeData){
    if(nodeData.name!=="CookbookDynamicInputs" && nodeData.comfyClass!=="CookbookDynamicInputs") return;
    const orig=nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated=function(){
      const r=orig?.apply(this, arguments);
      if(this.__cookbookDynamicAdded) return r;
      this.__cookbookDynamicAdded=true;
      createDynamicWidget(this);
      return r;
    };
  }
});
