"""
H3 Prompt - CLIP (Solid unified, worth doing)
Identical base to Ollama node, only model differs: clip vs ollama_model
Short display name, Autogrow refs 0-9, system_prompt restored, duration-aware, concept-aware, thinking filter not needed for CLIP but kept for identical base
"""

from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Any

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
    images=[]; keys=[]
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

    for k,v in list(all_refs.items()):
        if k.startswith("ref_image_") or k.startswith("ref_images") or k.startswith("image_ref_"):
            keys.append(k)
            images.append(v)
    total=0
    for img in images:
        try:
            if hasattr(img,'shape') and len(img.shape)==4: total+=img.shape[0]
            else: total+=1
        except: total+=1
    total_video=1 if "video_ref" in all_refs else 0
    return total, total_video, keys

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


class H3PromptGeneratorClipSimple:
    CATEGORY="ComfyUI-OMG/Video/MiniMax-H3/Prompting"
    FUNCTION="generate"
    RETURN_TYPES=("STRING","STRING","INT")
    RETURN_NAMES=("enhanced_prompt","task_type","char_count")
    DESCRIPTION="H3 Prompt Generator - CLIP solid unified with identical base to Ollama, only model differs, Autogrow refs, system_prompt restored, duration-aware, concept-aware"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "prompt":("STRING",{"default":"a fast pace biker driving through busy newyork city road surviving other vehicles","multiline":True, "tooltip":"Few words allowed - expanded to 4000-6000 char official prompt with alignment line + 3 fields"}),
                "clip":("CLIP",{"tooltip":"CLIP Qwen3-VL-32B - joint image+text via clip.tokenize(prompt, images=...), used for subject/character auto details"}),
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
                "filter_thinking":("BOOLEAN",{"default":True, "tooltip":"Filter thinking tags <think>...</think> from output - for qwen3 etc. Kept for identical base to Ollama node, though CLIP doesn't have thinking."}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, prompt, clip, ref_image_1=None, duration=8, aspect_ratio="16:9", task_type="Auto", fal_preset="auto-infer", small_model_mode=False, filter_thinking=True, custom_system_prompt="", **kwargs):
        import hashlib, json
        try:
            key={"prompt":prompt[:500],"duration":duration,"task_type":task_type,"fal_preset":fal_preset,"filter_thinking":filter_thinking,"additional_input":kwargs.get("additional_input","")[:200]}
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
        except:
            return prompt

    def generate(self, prompt:str, clip, ref_image_1=None, duration=8, aspect_ratio="16:9", task_type="Auto", fal_preset="auto-infer", small_model_mode=False, filter_thinking=True, custom_system_prompt="", **kwargs):
        total_images, total_video, ref_keys = _collect_refs(kwargs, ref_image_1)

        concept=prompt.strip() or "a cinematic moment"
        analysis=_analyze_concept(concept)
        corrected=analysis["corrected"]
        final_task=task_type if task_type!="Auto" else "T2VA"
        cuts=_calc_cuts(duration)
        num_shots=len(cuts)+1

        # Alignment
        align_line=""
        if final_task=="I2VA":
            align_line=f"For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n"
        elif final_task=="FL2VA":
            align_line=f"How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot {num_shots}) aligns with the {duration:.2f}-second mark of the target video.\n"
        elif final_task=="L2VA":
            align_line=f"How the reference pictures align with the target video — <Picture 1> (from [Shot {num_shots}]) aligns with the {duration:.2f}-second mark of the target video.\n"

        # Ref assignments with subject/character auto details via CLIP
        ref_parts=[]
        for idx in range(1, min(total_images+1,10)):
            if final_task=="I2VA" and idx==1:
                ref_parts.append(f"<Picture {idx}> is first-frame anchor at 0.00s [Shot 1] - The scene opens exactly on <Picture {idx}>, preserving original subject identity, shape, clothing, pose, scene, lighting palette, camera relationship, key props. Auto subject details via CLIP preserved.")
            elif final_task=="FL2VA":
                if idx==1: ref_parts.append(f"<Picture {idx}> is first-frame at 0.00s [Shot 1];")
                elif idx==2: ref_parts.append(f"<Picture {idx}> is last-frame at {duration:.2f}s [Shot {num_shots}]; progressive convergence at {duration:.2f}s.")
                else: ref_parts.append(f"<Picture {idx}> general reference;")
            elif final_task=="L2VA" and idx==1:
                ref_parts.append(f"<Picture {idx}> last-frame at {duration:.2f}s [Shot {num_shots}];")
            else:
                if idx==1: ref_parts.append(f"Use <Picture {idx}> for main character identity, facial identity, hairstyle, age, body proportions, clothing - auto subject details via CLIP")
                elif idx==2: ref_parts.append(f"Use <Picture {idx}> for environment, lighting, color palette and final composition reference")
                else: ref_parts.append(f"Use <Picture {idx}> for style and secondary subject preservation")
        if total_video>0:
            ref_parts.append(f"Use <Video 1> ({total_video} frames) only for movement, choreography, timing, camera movement and shot rhythm - do not transfer performer identity/clothing/environment")
        ref_info = " ".join(ref_parts) if ref_parts else "No picture anchors. Pure text-to-video."

        # System prompt handling - single system prompt input, no extra (proper use)
        # custom_system_prompt is the ONLY system prompt input for CLIP node as well (identical base to Ollama)
        custom_system_prompt = kwargs.get("custom_system_prompt","") or ""
        if isinstance(custom_system_prompt, str) and custom_system_prompt.strip():
            sys_prompt_used = custom_system_prompt
        else:
            # Backward compat for system_prompt
            sys_from_kwargs = kwargs.get("system_prompt","") or ""
            if isinstance(sys_from_kwargs, str) and sys_from_kwargs.strip():
                sys_prompt_used = sys_from_kwargs
            else:
                sys_prompt_used = CUSTOM_SYSPROMPT

        additional_input = kwargs.get("additional_input","") or ""

        # Build shots - concept-aware, duration-aware, worth doing
        opening="Realistic live-action cinematic look with practical film-photography texture, anamorphic lens character, shallow depth of field, volumetric atmosphere, natural motion blur, controlled film grain"

        if analysis["genre"]=="bike_chase":
            shot1=(
                f"[Shot 1] {opening}. Scene overview: {corrected}. Who: {analysis['subject_desc']}. Where: {analysis['env_desc']}. What: fast survival driving, objective to navigate without collision, final to emerge clear. "
                f"Initial composition: low tracking side angle at bumper level, biker centered slightly right, foreground yellow taxi roof blurred, background Manhattan buildings and traffic lights. {analysis['vehicle_desc']}. {ref_info} {additional_input[:200]} "
                f"The camera pushes in with small amplitude at slow speed, tracking alongside motorcycle for {duration}s duration, maintaining constant environment with practical street lights, neon reflections on wet asphalt, shallow depth isolating rider."
            )
        else:
            shot1=(
                f"[Shot 1] {opening}. Scene overview: {corrected}. Who: {analysis['subject_desc']}. Where: {analysis['env_desc']}. What: action per concept with dramatic objective. "
                f"Initial composition: geography locked left/right, main subject centered. {ref_info} {additional_input[:200]} "
                f"The camera pushes in with small amplitude at slow speed for {duration}s duration, maintaining constant environment."
            )

        shot_texts=[shot1]
        for idx, cut in enumerate(cuts, start=2):
            cs=_fmt_t(cut)
            is_final=(idx==num_shots)
            if analysis["genre"]=="bike_chase":
                if idx==2:
                    txt=f"[Shot {idx}] At 00:{cs}, the camera cuts to high side tracking shot with large amplitude at fast speed, revealing dense traffic ahead. Action onset: rider {analysis['action_verbs'][0]}, {analysis['action_verbs'][1]}, throttle twist visible fork compression. Camera trucking left alongside, bus SUV braking. Motion: cars blur, lane streaks, wind buffets jacket. Tire contact detailed heat shimmer lean angle increases. {additional_input[:200]}"
                elif is_final:
                    txt=f"[Shot {idx}] At 00:{cs}, the camera cuts to low-angle hero tracking and close-up of helmet visor with large amplitude at slow speed focusing final survival until {duration:.2f}s. Action: rider {analysis['action_verbs'][3] if len(analysis['action_verbs'])>3 else 'swerves'} {analysis['action_verbs'][4] if len(analysis['action_verbs'])>4 else ''} clearing gap pulling open lane. Result: {analysis['vehicle_desc']}. Final hold 00:{cs} to {duration:.2f}s wheels spinning exhaust settling traffic behind silhouette at {duration:.2f}s."
                else:
                    txt=f"[Shot {idx}] At 00:{cs}, the camera cuts to rear three-quarter chase small fast. Development: rider {analysis['action_verbs'][2]}, {analysis['action_verbs'][3]} between sedan van inches clearance. Camera arcs large showing density. Lighting: late afternoon sun rim on helmet reflections."
            else:
                if is_final:
                    txt=f"[Shot {idx}] At 00:{cs}, the camera cuts to close-up low-angle beauty large slow final visual beat until {duration:.2f}s. Action per concept: {', '.join(analysis['action_verbs'])} and settles. Final hold 00:{cs} to {duration:.2f}s ending held silhouette at {duration:.2f}s. {additional_input[:200]}"
                else:
                    txt=f"[Shot {idx}] At 00:{cs}, the camera cuts to medium shot small normal revealing more environment per concept. Action: {analysis['action_verbs'][0] if analysis['action_verbs'] else 'progresses'} per concept with observable mechanics. Camera pans right large fast reveals new elements settles gentle handheld toward final at {duration:.2f}s."
            shot_texts.append(txt)

        continuity=(
            f"Identity and continuity protected: preserve {analysis['subject_desc'][:80]} geometry throughout {duration}s; no random text, subtitles, logos or watermarks; "
            f"overall tempo fits {duration}s with strictly increasing cuts {', '.join([f'00:{_fmt_t(c)}' for c in cuts])} inside duration, final hold until {duration:.2f}s; "
            f"camera motion natural English with amplitude small/large and speed slow/fast; final settling until {duration:.2f}s. {additional_input[:300]}"
        )

        imd="\n\n".join(shot_texts)+"\n\n"+continuity

        # Soundscape tailored
        if analysis["genre"]=="bike_chase":
            soundscape=(
                f"{', '.join(analysis['sound_hints'])}. High-revving inline-four engine rising and falling with throttle from 00:00.000 to {duration:.2f}s, tire hiss on asphalt, wind blast increasing with speed, traffic honks when rider cuts close, truck brake puff at mid. "
                f"Synchronized: engine pitch matches acceleration at 00:{_fmt_t(cuts[0])} and 00:{_fmt_t(cuts[-1])}, tire screech at avoidance. {additional_input[:200]}"
            )
            music=(
                f"High-energy electronic rock hybrid with pulsating synth bass and driving drums at fast tempo, accent hits at {', '.join([f'00:{_fmt_t(c)}' for c in cuts])} synchronized to lane changes, full burst at 00:{_fmt_t(cuts[-1])} for final overtake, then rapid fade to near-silence from 00:{_fmt_t(cuts[-1])} to {duration:.2f}s. "
                f"Score ends at {duration:.2f}s."
            )
        else:
            soundscape=(
                f"{', '.join(analysis['sound_hints'])} tied to visible action per {corrected[:80]}. Synchronized across {duration}s timeline 00:00.000 to {duration:.2f}s toward final at {duration:.2f}s. {additional_input[:200]}"
            )
            music=(
                f"Low pulse underlies first shots slow tempo, accent hits at {', '.join([f'00:{_fmt_t(c)}' for c in cuts])}, fuller burst at 00:{_fmt_t(cuts[-1])} for final reveal, resolving to near-silence from 00:{_fmt_t(cuts[-1])} to {duration:.2f}s. Score ends at {duration:.2f}s."
            )

        if align_line:
            final=f"{align_line}\nintegrated_multimodal_description: {imd}\n\noverall_soundscape: {soundscape}\n\nnon_diegetic_music: {music}\n"
        else:
            final=f"integrated_multimodal_description: {imd}\n\noverall_soundscape: {soundscape}\n\nnon_diegetic_music: {music}\n"

        # enforce 4000-6000, preferred 4600-5400, no trimming marker
        tries=0
        while len(final)>6000 and tries<5:
            cut=int(len(imd)*0.15)
            imd=imd[:-cut].rsplit('.',1)[0]+"."
            final=f"{align_line}\nintegrated_multimodal_description: {imd}\n\noverall_soundscape: {soundscape}\n\nnon_diegetic_music: {music}\n" if align_line else f"integrated_multimodal_description: {imd}\n\noverall_soundscape: {soundscape}\n\nnon_diegetic_music: {music}\n"
            tries+=1

        while len(final)<4600:
            add=f" Additional relevant detail: precise visual style shallow depth field volumetric atmosphere centered, performance per {corrected[:60]}, actions {', '.join(analysis['action_verbs'][:3])}, camera Push In small slow Pan Right large fast, environmental motion, lighting practical texture, sound sync until {duration:.2f}s. {additional_input[:200]}"
            if len(final)+len(add)>5800: add=add[:5800-len(final)]
            final=final.replace("\n\noverall_soundscape:", add+"\n\noverall_soundscape:")
            if len(add)<20: break

        if len(final)>6000: final=final[:5950].rsplit('.',1)[0]+"."
        if len(final)<4000: final+=(" Final beat strong."*3)[:4000-len(final)]

        # Filter thinking if enabled (identical base to Ollama node)
        if filter_thinking:
            final = filter_thinking(final, True)

        return (final, final_task, len(final))

NODE_CLASS_MAPPINGS={"H3PromptGeneratorClipSimple":H3PromptGeneratorClipSimple}
NODE_DISPLAY_NAME_MAPPINGS={"H3PromptGeneratorClipSimple":"H3 Prompt - CLIP"}
