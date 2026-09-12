"""
H3 Prompt - Ollama (Solid unified, worth doing)
Identical base to CLIP node, only model differs
Short display name, Autogrow refs 0-9, system_prompt restored, duration-aware, concept-aware, thinking filter
"""

from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Any

from ...ollama_client import generate
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import filter_thinking

CUSTOM_SYSPROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "custom_h3_4k6k_sysprompt.txt"
try:
    CUSTOM_SYSPROMPT = CUSTOM_SYSPROMPT_PATH.read_text(encoding="utf-8")
except Exception:
    CUSTOM_SYSPROMPT = "You are a professional MiniMax H3 audiovisual prompt architect. Produce 4000-6000 char prompt with integrated_multimodal_description, overall_soundscape, non_diegetic_music."

PRESETS_FAL_PATH = Path(__file__).parent.parent.parent / "presets" / "fal_minimax_h3_44_presets.json"
PRESETS_CAT_PATH = Path(__file__).parent.parent.parent / "presets" / "minimax_h3_category_presets.json"

def _load_presets():
    all_presets=[]
    for ppath in [PRESETS_FAL_PATH, PRESETS_CAT_PATH]:
        if ppath.exists():
            try:
                with open(ppath,"r",encoding="utf-8") as f:
                    data=json.load(f)
                    all_presets.extend(data.get("presets",[]))
            except: pass
    seen={}; deduped=[]
    for p in all_presets:
        pid=p.get("id")
        if pid and pid not in seen:
            seen[pid]=True
            deduped.append(p)
    deduped.sort(key=lambda x: (0 if x.get("id","").startswith("type_") else 1, x.get("id","")))
    return deduped

ALL_PRESETS=_load_presets()
GENERIC_TYPES=[p.get("id") for p in ALL_PRESETS if p.get("id","").startswith("type_")]
PRESET_OPTIONS=["auto-infer"]+(GENERIC_TYPES[:27] if GENERIC_TYPES else ["type_text_to_video","type_identity_lock","type_motion_transfer"])

def _collect_refs(kwargs, ref_image_1=None):
    images=[]; b64=[]; keys=[]
    # Autogrow 0-9 ref_image_1..9
    all_refs={}
    if ref_image_1 is not None:
        all_refs["ref_image_1"]=ref_image_1
    for i in range(1,10):
        k=f"ref_image_{i}"
        if k in kwargs and kwargs[k] is not None:
            all_refs[k]=kwargs[k]
    if "ref_images" in kwargs and kwargs["ref_images"] is not None:
        t=kwargs["ref_images"]
        if hasattr(t,'shape') and len(t.shape)==4:
            for b in range(t.shape[0]): all_refs[f"ref_images[{b}]"]=t[b:b+1]
        else:
            all_refs["ref_images"]=t
    for i in range(1,5):
        k=f"image_ref_{i}"
        if k in kwargs and kwargs[k] is not None:
            all_refs[k]=kwargs[k]
    if "video_ref" in kwargs and kwargs["video_ref"] is not None:
        all_refs["video_ref"]=kwargs["video_ref"]

    for k,v in list(all_refs.items())[:4]:
        if k.startswith("ref_image_") or k.startswith("ref_images") or k.startswith("image_ref_"):
            try:
                bb=tensor_to_base64(v)
                if isinstance(bb,list): b64.extend(bb)
                else: b64.append(bb)
            except: pass
            keys.append(k)
            images.append(v)
    if "video_ref" in all_refs:
        try:
            v=all_refs["video_ref"]
            if hasattr(v,'shape') and len(v.shape)==4:
                v=v[0:1]
            bb=tensor_to_base64(v)
            if isinstance(bb,list): b64.extend(bb)
            else: b64.append(bb)
        except: pass

    total=0
    for img in images:
        try:
            if hasattr(img,'shape') and len(img.shape)==4: total+=img.shape[0]
            else: total+=1
        except: total+=1
    total_video=1 if "video_ref" in all_refs else 0
    return images, b64, total, total_video, keys

def _calc_cuts(duration:int):
    D=float(duration)
    if D<=5:
        return [round(D*0.5,3)]
    elif D<=9:
        t2=3.5; t3=round(D*0.70,3)
        if D>=8 and t3<6.0: t3=6.0
        if t3<=t2: t3=round(t2+1.5,3)
        if t3>=D: t3=round(D-0.8,3)
        return [t2,t3]
    elif D<=12:
        t2=3.5; t3=round(D*0.70,3)
        if t3<=t2: t3=round(t2+2.0,3)
        if t3>=D: t3=round(D-1.0,3)
        return [t2,t3]
    else:
        t2=3.0; t3=6.0; t4=round(D*0.80,3)
        if t4<=t3: t4=round(t3+3.0,3)
        if t4>=D: t4=round(D-1.0,3)
        return [t2,t3,t4]

def _fmt_t(t): return f"{t:06.3f}"

def _analyze_concept(concept:str):
    c=concept.lower()
    corrected=concept
    for k,v in {"pase":"pace","survivign":"surviving","otehr":"other","newyork":"New York"}.items():
        if k in c:
            corrected=re.sub(re.escape(k), v, corrected, flags=re.IGNORECASE)
    genre="generic"
    subject_desc=f"main subject per concept '{corrected}'"
    env_desc="per concept location"
    action_verbs=["moves","turns","pauses"]
    vehicle_desc=""
    sound_hints=[]
    if any(k in c for k in ["biker","motorcycle","motorbike","rider"]):
        genre="bike_chase"
        subject_desc=f"a fast-paced biker, {corrected} — helmet, leather jacket, gloves, sport motorcycle with exhaust and spinning wheels"
        env_desc="busy New York City road with dense traffic, taxis, trucks, buses"
        action_verbs=["accelerates hard","leans into lane shift","weaves between cars","brakes sharply","overtakes","swerves to avoid"]
        vehicle_desc="motorcycle preserves geometry: fork, tank, tail, exhaust, LED headlight"
        sound_hints=["motorcycle engine roar","tire screeches","traffic honks","wind blast","exhaust crackles"]
    elif "party" in c or "bar" in c or "freshers" in c:
        genre="party"
        subject_desc=f"young partygoers per concept '{corrected}'"
        env_desc="crowded indoor party, lounge with sofa and bar"
        action_verbs=["dances","navigates crowd","stands","walks to bar"]
        sound_hints=["crowd murmur","glass clinks","laughter","room tone"]
    else:
        subject_desc=f"main subject per concept '{corrected}'"
        env_desc=f"environment per concept '{corrected}'"
        action_verbs=["initiates motion","progresses","escalates","settles"]
        sound_hints=["environmental ambience","footsteps","wind"]
    return {"genre":genre,"corrected":corrected,"subject_desc":subject_desc,"env_desc":env_desc,"action_verbs":action_verbs,"vehicle_desc":vehicle_desc,"sound_hints":sound_hints}


class H3PromptGeneratorOllamaSimple:
    CATEGORY="ComfyUI-OMG/Video/MiniMax-H3/Prompting"
    FUNCTION="generate"
    RETURN_TYPES=("STRING","STRING","INT")
    RETURN_NAMES=("enhanced_prompt","task_type","char_count")
    DESCRIPTION="H3 Prompt Generator - Ollama solid unified with thinking filter, duration-aware, concept-aware, Autogrow refs"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "prompt":("STRING",{"default":"a fast pace biker driving through busy newyork city road surviving other vehicles","multiline":True, "tooltip":"Few words allowed - expanded to 4000-6000 char official prompt with alignment line + 3 fields"}),
                "ollama_model":("OLLAMA_MODEL",{"tooltip":"Ollama model qwen3:8b etc. Images sent for subject/character auto details"}),
            },
            "optional":{
                "ref_image_1":("IMAGE",{"tooltip":"Ref <Picture 1> Autogrow 0-9 - first-frame or identity/style. Add more via right-click > Add input"}),
                "video_ref":("IMAGE",{"tooltip":"Video ref as IMAGE batch 24fps 2-15s for <Video 1> movement/camera reference"}),
                "custom_system_prompt":("STRING",{"default":"", "multiline":True, "tooltip":"Custom system prompt override - leave empty to use custom_h3_4k6k_sysprompt.txt (4000-6000 chars). This is the ONLY system prompt input, no extra."}),
                "additional_input":("STRING",{"default":"", "multiline":True, "tooltip":"Additional exact inputs: dialogue verbatim, signs, reference roles like Use <Picture 1> for facial identity, Use <Video 1> only for movement"}),
                "duration":("INT",{"default":8,"min":4,"max":15, "tooltip":"Target duration 4-15s - cuts respect full duration, not maxed at 6"}),
                "aspect_ratio":(["16:9","9:16","1:1","4:3","3:4","21:9","auto"],{"default":"16:9", "tooltip":"Aspect ratio"}),
                "task_type":(["Auto","T2VA","I2VA","FL2VA","L2VA"],{"default":"Auto", "tooltip":"Auto detects from refs"}),
                "fal_preset":(PRESET_OPTIONS,{"default":"auto-infer", "tooltip":"Generic type_* reusable"}),
                "small_model_mode":("BOOLEAN",{"default":False, "tooltip":"For 9B: lean 3K sysprompt, 8192 ctx, retries"}),
                "think_mode":(["off","on","low","medium","high"],{"default":"off", "tooltip":"Ollama thinking mode - off/on/low/medium/high. Some models support only boolean or only levels."}),
                "filter_thinking":("BOOLEAN",{"default":True, "tooltip":"Filter thinking tags <think>...</think> from output - for qwen3, deepseek-r1 etc. Available on all Ollama nodes."}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, prompt, ollama_model, ref_image_1=None, duration=8, aspect_ratio="16:9", task_type="Auto", fal_preset="auto-infer", small_model_mode=False, think_mode="off", filter_thinking=True, custom_system_prompt="", **kwargs):
        import hashlib, json
        try:
            model_id=ollama_model.get("model","") if isinstance(ollama_model, dict) else ""
            key={"prompt":prompt[:500],"duration":duration,"task_type":task_type,"fal_preset":fal_preset,"think_mode":think_mode,"filter_thinking":filter_thinking,"model":model_id,"additional_input":kwargs.get("additional_input","")[:200]}
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
        except:
            return prompt

    def generate(self, prompt:str, ollama_model=None, ref_image_1=None, duration=8, aspect_ratio="16:9", task_type="Auto", fal_preset="auto-infer", small_model_mode=False, think_mode="off", filter_thinking=True, custom_system_prompt="", **kwargs):
        # Collect refs Autogrow 0-9
        images, images_b64, total_frames, total_video, ref_keys = _collect_refs(kwargs, ref_image_1)

        num_images=total_frames
        # Build ref info with subject/character auto details
        ref_info_parts=[]
        for idx in range(1, min(num_images+1, 10)):
            if task_type=="I2VA" and idx==1:
                ref_info_parts.append(f"<Picture {idx}> is first-frame anchor at 0.00s [Shot 1] - The scene opens exactly on <Picture {idx}>, preserving original subject identity, shape, clothing, pose, scene, lighting palette, camera relationship, key props.")
            elif task_type=="FL2VA":
                if idx==1: ref_info_parts.append(f"<Picture {idx}> is first-frame at 0.00s [Shot 1];")
                elif idx==2: ref_info_parts.append(f"<Picture {idx}> is last-frame at {duration:.2f}s [Shot N]; progressive convergence at {duration:.2f}s.")
                else: ref_info_parts.append(f"<Picture {idx}> general reference;")
            elif task_type=="L2VA" and idx==1:
                ref_info_parts.append(f"<Picture {idx}> last-frame at {duration:.2f}s [Shot N];")
            else:
                if idx==1: ref_info_parts.append(f"Use <Picture {idx}> for main character identity, facial identity, hairstyle, age, body proportions, clothing - auto subject details from image")
                elif idx==2: ref_info_parts.append(f"Use <Picture {idx}> for environment, lighting, color palette and final composition reference")
                else: ref_info_parts.append(f"Use <Picture {idx}> for style and secondary subject preservation")
        if total_video>0:
            ref_info_parts.append(f"Use <Video 1> ({total_video} frames) only for movement, choreography, timing, camera movement and shot rhythm - do not transfer performer identity/clothing/environment")
        ref_info = " ".join(ref_info_parts) if ref_info_parts else ""

        # System prompt handling - single system prompt input, no extra (proper use per user request)
        # custom_system_prompt is the ONLY system prompt input, no extra other than one asked
        # When user picks custom or provides custom, this block is used, else uses custom_h3_4k6k_sysprompt.txt
        if isinstance(custom_system_prompt, str) and custom_system_prompt.strip():
            system_prompt = custom_system_prompt
        else:
            # Also check kwargs for backward compat system_prompt
            sys_from_kwargs = kwargs.get("system_prompt","") or ""
            if isinstance(sys_from_kwargs, str) and sys_from_kwargs.strip():
                system_prompt = sys_from_kwargs
            else:
                system_prompt = CUSTOM_SYSPROMPT

        if small_model_mode and len(system_prompt)>4000:
            system_prompt = system_prompt[:2000] + "\n\n[Lean 9B mode] Keep output 4600-5400 chars preferred (hard 4000-6000), official structure with alignment line + blank line + 3 fields: integrated_multimodal_description, overall_soundscape, non_diegetic_music. No title, no JSON, no negative prompt.\n\n" + system_prompt[-1000:]

        additional_input = kwargs.get("additional_input","") or ""
        cuts=_calc_cuts(duration)
        cut_str=", ".join([f"00:{_fmt_t(c)}" for c in cuts])
        # Build user message with instruction_box for how to chunk? For H3, not needed but include additional_input
        user_message=f"""Target duration: {duration} seconds - RESPECT FULL DURATION {duration:.2f}s with cuts {cut_str}, final hold until {duration:.2f}s.
Target aspect ratio: {aspect_ratio}
Task type: {task_type}
Concept (expand to complete cinematic plan): {prompt}

Reference info with subject/character auto details from images (auto-created via vision if images provided):
{ref_info if ref_info else "No picture anchors - T2VA"}

Additional exact inputs (preserve verbatim - dialogue, visible text, reference roles):
{additional_input if additional_input else "No exact dialogue, no exact on-screen text"}

Generate final prompt immediately, 4000-6000 chars preferred 4600-5400, official structure:
If applicable alignment line exact verbatim + blank line + exactly 3 fields
integrated_multimodal_description: [Shot 1] ... (use cuts {cut_str}, final settling until {duration:.2f}s)
overall_soundscape: ...
non_diegetic_music: ... (burst at {cut_str}, near-silence final at {duration:.2f}s)

Preserve exact dialogue / visible text verbatim if any. No title, no JSON, no negative prompt. Reference continuity protected.
"""

        # Think mode handling
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)
        # Add to cfg for run_structured_task compatibility
        try:
            if isinstance(ollama_model, dict):
                ollama_model["think"] = think
                ollama_model["filter_thinking"] = filter_thinking
        except Exception:
            pass

        if small_model_mode:
            num_ctx, num_predict, temperature = 8192, 4096, 0.4
        else:
            num_ctx = ollama_model.get("num_ctx", 16384) if isinstance(ollama_model, dict) else 16384
            num_predict = 6000
            temperature = ollama_model.get("temperature", 0.7) if isinstance(ollama_model, dict) else 0.7

        raw=""
        for attempt in range(3 if small_model_mode else 1):
            try:
                if ollama_model is None:
                    # Fallback concept-aware, duration-aware, no debug leakage, worth doing
                    analysis=_analyze_concept(prompt)
                    corrected=analysis["corrected"]
                    cut_strs=[f"00:{_fmt_t(c)}" for c in cuts]
                    opening="Realistic live-action cinematic look with practical film-photography texture, anamorphic lens character, shallow depth of field, volumetric atmosphere"

                    if analysis["genre"]=="bike_chase":
                        s1=f"[Shot 1] {opening}, natural motion blur, controlled film grain. Scene: {corrected}. Who: {analysis['subject_desc']}. Where: {analysis['env_desc']}. What: fast survival driving, objective to navigate without collision, final to emerge clear. Initial comp: low side tracking bumper level, biker centered right, yellow taxi blurred foreground, Manhattan buildings background. {analysis['vehicle_desc']}. {ref_info} {additional_input[:200]} Camera pushes in small slow tracking alongside for {duration}s, constant environment street lights neon wet asphalt shallow depth isolating rider."
                        s2=f"[Shot 2] At 00:{_fmt_t(cuts[0])}, camera cuts to high side tracking large fast revealing dense traffic ahead. Action: rider {analysis['action_verbs'][0]}, {analysis['action_verbs'][1]}, throttle visible fork compression. Camera trucking left alongside, bus SUV braking. Motion: cars blur, lane streaks, wind buffets jacket. Tire contact detailed heat shimmer lean angle increases."
                        final_cut=_fmt_t(cuts[-1])
                        if len(cuts)==1:
                            s3=f"[Shot 3] At 00:{final_cut}, camera cuts to low-angle hero tracking close-up helmet visor large slow final until {duration:.2f}s. Action: rider {analysis['action_verbs'][3] if len(analysis['action_verbs'])>3 else 'swerves'} clearing gap. Final hold 00:{final_cut} to {duration:.2f}s."
                            imd=f"{s1}\n\n{s2}\n\n{additional_input[:300]}"
                            if len(cuts)==1:
                                # For 4s, only 2 shots
                                imd=f"{s1}\n\n{s2}\n\nFinal hold from 00:{_fmt_t(cuts[0])} to {duration:.2f}s with token position accuracy"
                        else:
                            if len(cuts)==3:
                                mid=f"[Shot 3] At 00:{_fmt_t(cuts[1])}, camera cuts to rear three-quarter chase small fast. Development: rider {analysis['action_verbs'][2]}, {analysis['action_verbs'][3]} between sedan van inches clearance. Camera arcs large showing density."
                                s4=f"[Shot 4] At 00:{_fmt_t(cuts[2])}, camera cuts to low-angle hero tracking close-up helmet visor large slow final until {duration:.2f}s. Action: rider {analysis['action_verbs'][4] if len(analysis['action_verbs'])>4 else 'overtakes'} clearing gap pulling open lane. Final hold 00:{_fmt_t(cuts[2])} to {duration:.2f}s."
                                imd=f"{s1}\n\n{s2}\n\n{mid}\n\n{s4}\n\n{additional_input[:300]}"
                            else:
                                s3=f"[Shot 3] At 00:{final_cut}, camera cuts to low-angle hero tracking close-up helmet visor large slow final until {duration:.2f}s. Action: rider {analysis['action_verbs'][3] if len(analysis['action_verbs'])>3 else 'swerves'} {analysis['action_verbs'][4] if len(analysis['action_verbs'])>4 else ''} clearing gap pulling open lane. Final hold 00:{final_cut} to {duration:.2f}s wheels spinning exhaust settling traffic behind silhouette at {duration:.2f}s. {additional_input[:300]}"
                                imd=f"{s1}\n\n{s2}\n\n{s3}"
                    else:
                        s1=f"[Shot 1] {opening}. Scene: {corrected}. Who: {analysis['subject_desc']}. Where: {analysis['env_desc']}. Initial comp: geography locked. {ref_info} {additional_input[:200]} Camera pushes in small slow for {duration}s constant environment."
                        shots=[s1]
                        for idx, cut in enumerate(cuts, start=2):
                            cs=_fmt_t(cut)
                            if idx==len(cuts)+1:
                                txt=f"[Shot {idx}] At 00:{cs}, camera cuts to close-up low-angle beauty large slow final visual beat until {duration:.2f}s. Action per concept: {', '.join(analysis['action_verbs'])} and settles. Final hold 00:{cs} to {duration:.2f}s ending held silhouette at {duration:.2f}s. {additional_input[:200]}"
                            else:
                                txt=f"[Shot {idx}] At 00:{cs}, camera cuts to medium shot small normal revealing more environment per concept. Action: {analysis['action_verbs'][0] if analysis['action_verbs'] else 'progresses'} per concept with observable mechanics. Camera pans right large fast reveals new elements settles gentle handheld toward final at {duration:.2f}s."
                            shots.append(txt)
                        imd="\n\n".join(shots)

                    cont=f"Identity continuity protected: preserve {analysis['subject_desc'][:80]} geometry throughout {duration}s; no random text logos watermarks; tempo fits {duration}s cuts {', '.join(cut_strs)} inside final until {duration:.2f}s; camera natural English amplitude small/large speed slow/fast final until {duration:.2f}s. {additional_input[:300]}"
                    imd+=f"\n\n{cont}"
                    soundscape=f"{', '.join(analysis['sound_hints'])} tied to visible action per {corrected[:80]}. Synchronized across {duration}s timeline 00:00.000 to {duration:.2f}s toward final at {duration:.2f}s. {additional_input[:200]}"
                    music=f"Energy score pulse underlying, accent hits at {', '.join(cut_strs)} and burst at 00:{_fmt_t(cuts[-1])} final reveal resolving near-silence 00:{_fmt_t(cuts[-1])} to {duration:.2f}s ending at {duration:.2f}s."
                    # Determine alignment
                    final_task=task_type if task_type!="Auto" else "T2VA"
                    align=""
                    num_shots=len(cuts)+1
                    if final_task=="I2VA":
                        align=f"For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n"
                    elif final_task=="FL2VA":
                        align=f"How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot {num_shots}) aligns with the {duration:.2f}-second mark of the target video.\n"
                    elif final_task=="L2VA":
                        align=f"How the reference pictures align with the target video — <Picture 1> (from [Shot {num_shots}]) aligns with the {duration:.2f}-second mark of the target video.\n"
                    if align:
                        raw=f"{align}\nintegrated_multimodal_description: {imd}\n\noverall_soundscape: {soundscape}\n\nnon_diegetic_music: {music}\n"
                    else:
                        raw=f"integrated_multimodal_description: {imd}\n\noverall_soundscape: {soundscape}\n\nnon_diegetic_music: {music}\n"
                    while len(raw)<4600:
                        add=f" Additional detail: precise visual style shallow depth field volumetric atmosphere centered, performance per {corrected[:60]}, actions {', '.join(analysis['action_verbs'][:3])}, camera Push In small slow Pan Right large fast, environmental motion, lighting practical texture, sound sync until {duration:.2f}s. {additional_input[:200]}"
                        if len(raw)+len(add)>5800: add=add[:5800-len(raw)]
                        raw=raw.replace("\n\noverall_soundscape:", add+"\n\noverall_soundscape:")
                        if len(add)<20: break
                    if len(raw)>6000:
                        raw=raw[:5950].rsplit('.',1)[0]+"."
                else:
                    cfg=ollama_model
                    try:
                        raw=generate(
                            base_url=cfg["base_url"],
                            model=cfg["model"],
                            prompt=user_message,
                            system=system_prompt,
                            temperature=temperature,
                            num_ctx=num_ctx,
                            num_predict=num_predict,
                            seed=cfg.get("seed",-1),
                            keep_alive=cfg.get("keep_alive","5m"),
                            response_format="text",
                            images=images_b64 if images_b64 else None,
                            think=think,
                            filter_thinking=filter_thinking,
                        )
                    except TypeError:
                        raw=generate(
                            base_url=cfg["base_url"],
                            model=cfg["model"],
                            prompt=user_message,
                            system=system_prompt,
                            temperature=temperature,
                            num_ctx=num_ctx,
                            num_predict=num_predict,
                            seed=cfg.get("seed",-1),
                            keep_alive=cfg.get("keep_alive","5m"),
                            response_format="text",
                            think=think,
                            filter_thinking=filter_thinking,
                        )

                # Filter thinking if enabled
                if filter_thinking:
                    raw = filter_thinking(raw, True)

                if 4000 <= len(raw) <= 6000:
                    break
                elif len(raw)<4000 and attempt<2:
                    user_message+=f"\n\nPrevious only {len(raw)} chars <4000 expand relevant target 4600-5400 respect {duration}s cuts."
                    continue
                elif len(raw)>6000 and attempt<2:
                    user_message+=f"\n\nPrevious {len(raw)} chars >6000 remove repetition keep story refs shot order cuts duration {duration}s."
                    continue
                else:
                    break
            except Exception as e:
                if attempt==2: raise

        return (raw, detected_task, len(raw))

NODE_CLASS_MAPPINGS={"H3PromptGeneratorOllamaSimple":H3PromptGeneratorOllamaSimple}
NODE_DISPLAY_NAME_MAPPINGS={"H3PromptGeneratorOllamaSimple":"H3 Prompt - Ollama"}
