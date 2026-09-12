"""Generate ComfyUI-OMG native ComfyUI subgraph blueprint JSON files."""
from __future__ import annotations
import json
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'subgraphs'
NS=uuid.UUID('d2c91cb7-9c1a-49d1-8df8-b667ca42f914')

def uid(name): return str(uuid.uuid5(NS,name))

def blueprint(filename,name,description,node_type,node_inputs,node_outputs,exposed_inputs,exposed_outputs,widgets_values,proxy_widgets):
    sgid=uid(filename+':subgraph')
    internal_id=10
    links=[]
    input_defs=[]
    output_defs=[]
    for slot,(input_name,input_type,target_slot,label) in enumerate(exposed_inputs):
        lid=slot+1
        links.append({'id':lid,'origin_id':-10,'origin_slot':slot,'target_id':internal_id,'target_slot':target_slot,'type':input_type})
        input_defs.append({'id':uid(filename+':input:'+input_name),'name':input_name,'type':input_type,'linkIds':[lid],'localized_name':input_name,'label':label,'pos':[100,140+slot*30]})
        node_inputs[target_slot]['link']=lid
    next_link=len(links)+1
    for slot,(output_name,output_type,source_slot,label) in enumerate(exposed_outputs):
        lid=next_link+slot
        links.append({'id':lid,'origin_id':internal_id,'origin_slot':source_slot,'target_id':-20,'target_slot':slot,'type':output_type})
        output_defs.append({'id':uid(filename+':output:'+output_name),'name':output_name,'type':output_type,'linkIds':[lid],'localized_name':output_name,'label':label,'pos':[980,140+slot*30]})
        node_outputs[source_slot]['links']=[lid]
    internal={
      'id':internal_id,'type':node_type,'pos':[350,80],'size':[420,360],'flags':{},'order':0,'mode':0,
      'inputs':node_inputs,'outputs':node_outputs,
      'properties':{'Node name for S&R':node_type,'cnr_id':'comfy-omg','ver':'0.1.0'},
      'widgets_values':widgets_values,
    }
    top_inputs=[{'label':label,'localized_name':name,'name':name,'type':typ,'link':None} for name,typ,_slot,label in exposed_inputs]
    top_outputs=[{'label':label,'localized_name':name,'name':name,'type':typ,'links':[]} for name,typ,_slot,label in exposed_outputs]
    top={
      'id':1,'type':sgid,'pos':[400,300],'size':[340,220],'flags':{},'order':0,'mode':0,
      'inputs':top_inputs,'outputs':top_outputs,'properties':{'proxyWidgets':[[node,widget] for node,widget,_index in proxy_widgets],'cnr_id':'comfy-omg','ver':'0.1.0'},
      'widgets_values':[widgets_values[idx] for _node,widget,idx in proxy_widgets], 'title':name,
    }
    definition={
      'id':sgid,'version':1,'state':{'lastGroupId':0,'lastNodeId':internal_id,'lastLinkId':max(x['id'] for x in links),'lastRerouteId':0},
      'revision':0,'config':{},'name':name,
      'inputNode':{'id':-10,'bounding':[0,100,120,max(60,len(input_defs)*30)]},
      'outputNode':{'id':-20,'bounding':[960,100,120,max(60,len(output_defs)*30)]},
      'inputs':input_defs,'outputs':output_defs,'widgets':[], 'nodes':[internal], 'links':links,'groups':[],
      'extra':{'workflowRendererVersion':'LG'},'category':'ComfyUI-OMG/Blueprints','description':description,
    }
    doc={'revision':0,'last_node_id':1,'last_link_id':0,'nodes':[top],'links':[],'version':0.4,'definitions':{'subgraphs':[definition]},'extra':{'workflowRendererVersion':'LG'}}
    (OUT/f'{filename}.json').write_text(json.dumps(doc,indent=2),encoding='utf-8')

def inp(name,typ,widget=False,shape=None):
    d={'label':name,'localized_name':name,'name':name,'type':typ,'link':None}
    if widget:
        d['widget']={'name':name}
    if shape is not None:
        d['shape']=shape
    return d

def out(name,typ):return {'localized_name':name,'name':name,'type':typ,'links':None}

# Proxy widget tuples are (node id string, widget name, index in widgets_values).
blueprint(
 'prompt_evaluation','OMG Prompt Evaluation','Evaluate a prompt and expose normalized score, verdict, issues, and improved prompt.',
 'OllamaPromptEvaluator',
 [inp('ollama_model','OLLAMA_MODEL'),inp('prompt','STRING',True),inp('target_model','COMBO',True),inp('intended_result','STRING',True),inp('evaluation_focus','COMBO',True),inp('creative_brief','OMG_CREATIVE_BRIEF',shape=7),inp('character_bible','OMG_CHARACTER_BIBLE',shape=7),inp('world_bible','OMG_WORLD_BIBLE',shape=7),inp('shot_list','OMG_SHOT_LIST',shape=7),inp('cache_policy','COMBO',True)],
 [out('evaluation','OMG_EVALUATION'),out('evaluation_json','STRING'),out('overall_score','FLOAT'),out('verdict','STRING'),out('strengths','STRING'),out('issues_json','STRING'),out('improved_prompt','STRING'),out('is_valid','BOOLEAN'),out('validation_report','STRING'),out('provenance_json','STRING')],
 [('ollama_model','OLLAMA_MODEL',0,'ollama_model'),('prompt','STRING',1,'prompt')],
 [('evaluation','OMG_EVALUATION',0,'evaluation'),('overall_score','FLOAT',2,'score'),('verdict','STRING',3,'verdict'),('improved_prompt','STRING',6,'improved_prompt')],
 ['', 'General','', 'balanced','use'],
 [['10','target_model',1],['10','evaluation_focus',3]],
)

blueprint(
 'character_bible_prompt','OMG Character Bible Prompt','Compile a Character Bible into a target prompt with scene context.',
 'OllamaCharacterBibleToPrompt',
 [inp('character_bible','OMG_CHARACTER_BIBLE'),inp('target_model','COMBO',True),inp('scene_context','STRING',True),inp('outfit_override','STRING',True),inp('pose_action','STRING',True),inp('expression','STRING',True),inp('extra_instructions','STRING',True)],
 [out('positive_prompt','STRING'),out('negative_prompt','STRING'),out('continuity_locks','STRING'),out('bible_json','STRING')],
 [('character_bible','OMG_CHARACTER_BIBLE',0,'character_bible'),('scene_context','STRING',2,'scene_context')],
 [('positive_prompt','STRING',0,'positive_prompt'),('negative_prompt','STRING',1,'negative_prompt'),('continuity_locks','STRING',2,'continuity_locks')],
 ['General','','','','',''],
 [['10','target_model',0]],
)

blueprint(
 'rag_search','OMG Local RAG Search','Retrieve local context and citations from a typed RAG index.',
 'OllamaRAGSearch',
 [inp('ollama_model','OLLAMA_MODEL'),inp('rag_index','OMG_RAG_INDEX'),inp('query','STRING',True),inp('top_k','INT',True),inp('minimum_score','FLOAT',True),inp('max_context_characters','INT',True)],
 [out('context','STRING'),out('citations_json','STRING'),out('scores','STRING'),out('result_count','INT'),out('top_source','STRING')],
 [('ollama_model','OLLAMA_MODEL',0,'ollama_model'),('rag_index','OMG_RAG_INDEX',1,'rag_index'),('query','STRING',2,'query')],
 [('context','STRING',0,'context'),('citations_json','STRING',1,'citations'),('result_count','INT',3,'count')],
 ['',5,0.0,12000],
 [['10','top_k',1],['10','minimum_score',2]],
)

blueprint(
 'shot_prompt','OMG Shot Prompt Extractor','Extract one shot from a typed Shot List for image/video generation.',
 'OllamaShotListGetShot',
 [inp('shot_list','OMG_SHOT_LIST'),inp('shot_index','INT',True)],
 [out('shot_id','STRING'),out('duration_seconds','FLOAT'),out('purpose','STRING'),out('first_frame_prompt','STRING'),out('video_prompt','STRING'),out('last_frame_prompt','STRING'),out('negative_prompt','STRING'),out('camera_direction','STRING'),out('lighting','STRING'),out('continuity_locks','STRING'),out('shot_json','STRING')],
 [('shot_list','OMG_SHOT_LIST',0,'shot_list')],
 [('first_frame_prompt','STRING',3,'first_frame'),('video_prompt','STRING',4,'video_prompt'),('last_frame_prompt','STRING',5,'last_frame'),('negative_prompt','STRING',6,'negative'),('continuity_locks','STRING',9,'locks')],
 [1],
 [['10','shot_index',0]],
)
print('Generated',len(list(OUT.glob('*.json'))),'subgraphs')
