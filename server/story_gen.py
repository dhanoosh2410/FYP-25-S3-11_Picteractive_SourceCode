from __future__ import annotations
"""
server/story_gen.py — v2.2
- Verifies BLIP nouns against CLIP labels (keeps only what the image supports)
- Cleaner subjects/titles (never uses colors/adjectives as subjects)
- Neutral opener: "In the first picture, ..."
- Simple grammar fixes ("appears", "draws", etc.)
- Deterministic planner with optional LLM polish; safe fallback if LLM unavailable
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import os, re

# ---------------- BLIP captioner ----------------
class BLIPCaptioner:
    def __init__(self, model_root: Path | None = None):
        self.ready = False
        self.err: Optional[str] = None
        self._processor = None
        self._model = None
        self._device = "cpu"
        self._try_load(model_root)

    def _try_load(self, model_root: Path | None):
        try:
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            candidates = []
            if model_root is not None:
                local = (Path(model_root) / "showtell" / "blip-base").resolve()
                if local.exists():
                    candidates.append({"name": str(local), "local_files_only": True})
            candidates.append({"name": "Salesforce/blip-image-captioning-base", "local_files_only": False})

            last = None
            for c in candidates:
                try:
                    self._processor = BlipProcessor.from_pretrained(
                        c["name"], local_files_only=c.get("local_files_only", False)
                    )
                    self._model = BlipForConditionalGeneration.from_pretrained(
                        c["name"], local_files_only=c.get("local_files_only", False)
                    ).to(self._device)
                    self.ready = True
                    self.err = None
                    return
                except Exception as e:
                    last = e
            raise last or RuntimeError("BLIP load failed")
        except Exception as e:
            self.ready = False
            self.err = (
                "blip_load_failed: " + str(e)
                + "; pip install torch transformers pillow accelerate safetensors; "
                + "optional local model at server/models/showtell/blip-base"
            )

    def _clean(self, s: str) -> str:
        t = (s or "").strip()
        if not t:
            return "a drawing"
        t = re.sub(r"\s+", " ", t).strip().rstrip(".!?")
        return t.lower()

    def caption(self, pil, region: Optional[Dict[str, int]] = None) -> str:
        if not self.ready or self._model is None or self._processor is None:
            self._try_load(None)
            if not self.ready:
                raise RuntimeError(self.err or "captioner unavailable")

        if region and all(k in region for k in ("x", "y", "w", "h")):
            x, y, w, h = [int(region[k]) for k in ("x", "y", "w", "h")]
            x = max(0, x); y = max(0, y); w = max(1, w); h = max(1, h)
            pil = pil.crop((x, y, x + w, y + h))

        import torch
        self._model.eval()
        with torch.no_grad():
            inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
            out = self._model.generate(**inputs, max_new_tokens=25, num_beams=3, repetition_penalty=1.05)
            txt = self._processor.decode(out[0], skip_special_tokens=True)

        return self._clean(txt)

# ---------------- Lexicons & utilities ----------------
STOP = set("a an the and with to of on in at for by near under over into onto from".split())

PEOPLE  = {"boy","girl","man","woman","child","kid","baby","person","friend"}
ANIMALS = {"dog","cat","bird","duck","rabbit","fish","worm"}
WEATHER = {"sun","cloud","rain","snow"}
PLACES  = {"park","school","home","house","room","kitchen","garden","forest","beach","playground","tree"}
OBJECTS = {
    "apple","banana","ball","kite","car","bus","bicycle","train","boat","airplane","plane",
    "tree","flower","book","cake","cookie","pizza","guitar","chair","table","bed","umbrella","leaf","shoe","hat"
}

COLOR_WORDS = {"red","green","blue","yellow","black","white","brown","pink","purple","orange"}
ADJ_BLOCK  = COLOR_WORDS | {"big","small","little","cute","happy","sad","bored","smiling"}
TITLE_BLOCKLIST = {"table"}  # avoid odd titles unless central

def words(s: str) -> List[str]:
    return re.findall(r"[a-z']+", (s or "").lower())

def a_an(word: str) -> str:
    if not word: return "a"
    return "an" if word[0] in "aeiou" else "a"

def verb_s(verb: str) -> str:
    v = (verb or "look").strip().lower()
    if v.endswith("s"): return v
    if v.endswith(("sh","ch","x","z","o")): return v + "es"
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou": return v[:-1] + "ies"
    return v + "s"

def pick_action(ws: List[str]) -> str:
    for w in ws:
        if w.endswith("ing"):
            return w
    for w in ("sees","holds","walks","runs","finds","looks","plays","meets","picks","smiles","rides","throws","reads","pecks","shares","helps"):
        if w in ws:
            return w
    return "looks"

def first_subject(ws: List[str], hints: List[str]) -> str:
    # 1) people/animals first
    for w in ws:
        if w in PEOPLE or w in ANIMALS:
            return w
    for h in (hints or []):
        if h in PEOPLE or h in ANIMALS:
            return h
    # 2) concrete objects (avoid colors/adjectives)
    for w in ws:
        if w in OBJECTS and w not in ADJ_BLOCK:
            return w
    for h in (hints or []):
        if h in OBJECTS and h not in ADJ_BLOCK:
            return h
    # 3) weather if nothing else
    for w in ws:
        if w in WEATHER:
            return w
    # 4) fallback to any non-adjective token
    for w in ws:
        if w not in (STOP | ADJ_BLOCK) and not w.endswith("ing"):
            return w
    return "friend"

def object_set(ws: List[str], hints: List[str]) -> List[str]:
    hits = []
    for w in ws:
        if w in (OBJECTS | PLACES | WEATHER | ANIMALS | PEOPLE) and w not in ADJ_BLOCK:
            hits.append(w)
    for h in (hints or []):
        if h in (OBJECTS | PLACES | WEATHER | ANIMALS | PEOPLE) and h not in ADJ_BLOCK:
            hits.append(h)
    # de-dup preserving order
    seen=set(); out=[]
    for x in hits:
        if x in seen: continue
        seen.add(x); out.append(x)
    return out[:6]

def guess_place(noun_list: List[str]) -> str:
    for n in noun_list:
        if n in PLACES:
            return n
    if "tree" in noun_list: return "tree"
    if any(w in noun_list for w in WEATHER): return "outside"
    return "outside"

def place_phrase(place: str) -> str:
    if not place or place == "outside":  return "outside"
    if place == "tree":                 return "under a tree"
    art = a_an(place)
    return f"at {art} {place}"

def detect_color_hint(pil) -> str:
    try:
        from PIL import ImageStat
        r,g,b = ImageStat.Stat(pil.convert("RGB")).mean
        if r>g and r>b and r>110: return "red"
        if g>r and g>b and g>110: return "green"
        if b>r and b>g and b>110: return "blue"
    except Exception:
        pass
    return ""

# ---- BLIP x CLIP reconciliation ----
def reconcile_with_clip(objs: List[str], subj: str, clip_labels: List[str]) -> tuple[List[str], str, dict]:
    """
    Cross-check BLIP-derived objects with CLIP labels.
    - Keeps only objects supported by CLIP (if CLIP returned anything).
    - If no overlap, fallback to CLIP top-1 (ignoring colors) as subject/object.
    """
    clip = [x for x in (clip_labels or []) if isinstance(x, str)]
    qc = {"clip": clip, "used": [], "match_ratio": 0.0, "fallback": False}

    if not clip:
        return objs, subj, qc

    oset = [o for o in objs if o not in COLOR_WORDS]
    overlap = [o for o in oset if o in clip]
    qc["match_ratio"] = (len(overlap) / max(1, len(set(oset))))

    if overlap:
        keep = overlap[:]
        if subj in (PEOPLE | ANIMALS) and subj not in keep:
            keep.insert(0, subj)
        qc["used"] = keep[:]
        new_subj = subj if subj in (PEOPLE | ANIMALS | OBJECTS) else (keep[0] if keep else subj)
        return keep, new_subj, qc

    # no overlap → trust CLIP top-1 (skip color words)
    top = clip[0]
    if top in COLOR_WORDS:
        for c in clip[1:]:
            if c not in COLOR_WORDS:
                top = c; break
    qc["used"] = [top]
    qc["fallback"] = True
    new_subj = top if top not in COLOR_WORDS else (subj or "friend")
    return [top], new_subj, qc

# ---------------- SceneExtractor ----------------
class SceneExtractor:
    def __init__(self, captioner: BLIPCaptioner):
        self.captioner = captioner

    def extract(self, pil, clip_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        cap = ""
        try:
            cap = self.captioner.caption(pil)
        except Exception:
            cap = "a simple drawing"

        ws = words(cap)
        hints = [x for x in (clip_labels or []) if isinstance(x, str)]
        subj = first_subject(ws, hints)
        objs = object_set(ws, hints)
        act  = pick_action(ws)
        place = guess_place(objs)
        color = detect_color_hint(pil)

        nouns = list(dict.fromkeys([w for w in ws if w not in STOP]))
        attrs = {}
        if color: attrs["color"] = color

        if subj not in objs:
            objs = [subj] + objs

        # reconcile with CLIP
        objs, subj, qc = reconcile_with_clip(objs, subj, hints)

        sentiment = "neutral"
        if "rain" in objs or "cloud" in objs: sentiment = "calm"
        elif "sun" in objs: sentiment = "bright"

        return {
            "caption": cap,
            "subject": subj,
            "objects": objs,
            "action": act,
            "place": place,
            "sentiment": sentiment,
            "attrs": attrs,
            "nouns": nouns,
            "qc": qc,
        }

# ---------------- DeltaComputer ----------------
class DeltaComputer:
    @staticmethod
    def compute(prev: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
        pset = set(prev.get("objects") or [])
        cset = set(cur.get("objects") or [])
        new = sorted(list(cset - pset))
        lost = sorted(list(pset - cset))

        interaction = ""
        if {"dog","ball"}.issubset(cset): interaction = "dog_chases_ball"
        elif {"bird","apple"}.issubset(cset): interaction = "bird_pecks_apple"
        elif {"umbrella","rain"}.issubset(cset): interaction = "share_umbrella"
        elif new: interaction = f"{new[0]}_appears"

        return {"new": new, "lost": lost, "interaction": interaction}

# ---------------- NarrativePlanner ----------------
class NarrativePlanner:
    @staticmethod
    def _clean_title_terms(terms: List[str]) -> List[str]:
        out=[]
        for t in terms:
            if not t or t in COLOR_WORDS or t in TITLE_BLOCKLIST: 
                continue
            if t in PLACES and t != "tree":
                continue
            out.append(t)
        prio=lambda x: (0 if x in (PEOPLE|ANIMALS) else (1 if x in OBJECTS else 2))
        out=sorted(list(dict.fromkeys(out)), key=prio)
        return out

    @classmethod
    def title_from_scenes(cls, scenes: List[Dict[str,Any]]) -> str:
        bag=[]
        for s in scenes:
            bag.append(s.get("subject",""))
            bag.extend(s.get("objects",[]))
        bag = cls._clean_title_terms(bag)
        if len(bag)>=2: return f"THE {bag[0].upper()} AND THE {bag[1].upper()}"
        if bag: return f"THE {bag[0].upper()} ADVENTURE"
        return "A LITTLE ADVENTURE"

    @staticmethod
    def panel1(s: Dict[str,Any]) -> str:
        subj = s.get("subject","friend")
        place_txt = place_phrase(s.get("place","outside"))
        others = [o for o in s.get("objects",[]) if o not in (PLACES | {subj})]
        seen = ", ".join(others[:2])
        art = a_an(subj)
        # Specific phrasing for iconic first scenes
        if subj == "apple":
            col = (s.get("attrs", {}) or {}).get("color", "")
            if col == "red":
                return "In the first picture, a shiny red apple sits quietly on the ground."
            return "In the first picture, an apple sits quietly on the ground."

        # Person/animal: subject-first opener
        if subj in (PEOPLE | ANIMALS):
            if seen:
                return f"In the first picture, {art} {subj} is {place_txt}, with {seen}."
            return f"In the first picture, {art} {subj} is {place_txt}."

        # Object/weather: scene-first opener
        if subj in OBJECTS | WEATHER:
            if seen:
                return f"In the first picture, {art} {subj} is {place_txt}, along with {seen}."
            return f"In the first picture, {art} {subj} is {place_txt}."

        # Fallback
        if seen:
            return f"In the first picture, we see {art} {subj} {place_txt}, with {seen}."
        return f"In the first picture, we see {art} {subj} {place_txt}."

    @staticmethod
    def _new_list_words(new: List[str]) -> str:
        if not new: return ""
        if len(new)==1:
            art = a_an(new[0]); return f"{art} {new[0]}"
        if len(new)==2:
            return f"{new[0]} and {new[1]}"
        return f"{new[0]}, {new[1]} and more"

    @staticmethod
    def panel2(s: Dict[str,Any], d: Dict[str,Any]) -> str:
        subj = s.get("subject","friend")
        act  = s.get("action","look")
        inter = (d or {}).get("interaction","")
        new = (d or {}).get("new", [])
        # Special, kid-friendly beats for common classroom drawings
        # Worm popping out of an apple (Scene 2)
        if "worm" in (new or []) and "apple" in (s.get("objects", []) or []):
            return "Then, a little worm pops out of the apple, curious about the world."
        if inter == "dog_chases_ball":
            return "Then, a friendly dog chases the ball, and they start to play."
        if inter == "bird_pecks_apple":
            return "Then, a curious bird pecks the red apple softly."
        if inter == "share_umbrella":
            return "Then, rain begins and they share an umbrella."
        if new:
            nl = NarrativePlanner._new_list_words(new)
            return f"Then, {nl} appears, and the {subj} {verb_s(act)}."
        return f"Then, the {subj} {verb_s(act)}."

    @staticmethod
    def panel3(s: Dict[str,Any], d: Dict[str,Any]) -> Tuple[str,str]:
        objs = set(s.get("objects",[]))
        # Bird and worm meet (Scene 3)
        if {"bird","worm"}.issubset(objs):
            col = (s.get("attrs", {}) or {}).get("color", "")
            bird_desc = "blue bird" if col == "blue" else "bird"
            return f"Finally, a {bird_desc} spots the worm and gently leans closer, ready to make a new friend.", "Being curious can start a friendship."
        if {"ball","dog"}.issubset(objs):
            return "Finally, they take turns and play together.", "Sharing makes playtime better."
        if {"bird","apple"}.issubset(objs):
            return "Finally, they watch the bird and keep the apple safe.", "Be gentle with little friends."
        if {"rain","tree"}.issubset(objs) or ("rain" in objs and s.get("place")=="tree"):
            return "Finally, they rest under a tree until the rain slows.", "Waiting calmly can solve problems."
        return "Finally, everyone smiles and feels proud.", "Kindness makes new friends."

    @classmethod
    def plan(cls, scenes: List[Dict[str,Any]], deltas: List[Dict[str,Any]]) -> Tuple[str, List[str], str]:
        title = cls.title_from_scenes(scenes)
        p1 = cls.panel1(scenes[0])
        p2 = cls.panel2(scenes[1], deltas[0] if deltas else {})
        p3, moral = cls.panel3(scenes[2], deltas[1] if len(deltas)>1 else {})
        return title, [p1,p2,p3], moral

# ---------------- StoryGenerator ----------------
class StoryGenerator:
    def __init__(self, model_root: Path | None = None):
        self.ready = False
        self.err: Optional[str] = None
        self._pipe = None
        self._mode = "text2text"
        self._device = -1
        self.model_name = ""
        self.captioner = BLIPCaptioner(model_root)
        self.extractor = SceneExtractor(self.captioner)
        self._try_load(model_root)

    def _try_load(self, model_root: Path | None) -> None:
        try:
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            import torch
            from transformers import pipeline

            self._device = 0 if torch.cuda.is_available() else -1

            candidates = [
                {"kind": "text2text-generation", "model": "google/flan-t5-base"},
                {"kind": "text2text-generation", "model": "google/flan-t5-small"},
                {"kind": "text-generation",       "model": "roneneldan/TinyStories-33M"},
                {"kind": "text-generation",       "model": "distilgpt2"},
                {"kind": "text-generation",       "model": "gpt2"},
            ]

            last_err: Exception | None = None
            for c in candidates:
                try:
                    self._pipe = pipeline(
                        c["kind"],
                        model=c["model"],
                        device=self._device,
                        framework="pt",
                    )
                    self._mode = "text2text" if c["kind"] == "text2text-generation" else "text-generation"
                    self.model_name = c["model"]
                    self.ready = True
                    self.err = None
                    print(f"[storygen] loaded {self.model_name} ({self._mode}) device={self._device}")
                    return
                except Exception as e:
                    last_err = e
                    continue

            self.ready = False
            raise last_err or RuntimeError("no model loaded")
        except Exception as e:
            self.ready = False
            self.err = (
                "storygen_load_failed: "
                + str(e)
                + "; pip install transformers torch accelerate safetensors; "
            )
            print("[storygen] ERROR:", self.err)

    @property
    def pipe(self):
        return self._pipe

    # ---------- API helpers ----------
    def build_scenes(self, images: List[Any], clip_label_lists: Optional[List[List[str]]] = None) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
        scenes=[]
        for i, img in enumerate(images):
            labs = (clip_label_lists[i] if clip_label_lists and i < len(clip_label_lists) else None)
            scenes.append(self.extractor.extract(img, labs))
        deltas=[]
        if len(scenes)>=2:
            deltas.append(DeltaComputer.compute(scenes[0], scenes[1]))
        if len(scenes)>=3:
            deltas.append(DeltaComputer.compute(scenes[1], scenes[2]))
        # choose a persistent main subject if possible (person/animal has priority)
        def _choose_main(scs):
            pool=[]
            for s in scs:
                if s.get("subject"): pool.append(s["subject"])
                pool += [o for o in s.get("objects", [])]
            # simple priority: person > animal > object
            for cand in pool:
                if cand in PEOPLE: return cand
            for cand in pool:
                if cand in ANIMALS: return cand
            return pool[0] if pool else "friend"
        
        main_subject = _choose_main(scenes)
        for s in scenes:
            # if subject is missing/weak, fall back to the chosen main subject
            if not s.get("subject"):
                s["subject"] = main_subject
                
        return scenes, deltas


    def _prompt_from_scenes(self, scenes: List[Dict[str,Any]], deltas: List[Dict[str,Any]], mood: str="friendly") -> str:
        vocab=set()
        for s in scenes:
            vocab.update([s.get("subject","")])
            vocab.update(s.get("objects",[]))
        vocab={w for w in vocab if w and w not in PLACES and w not in COLOR_WORDS}
        req = ", ".join(sorted(vocab)) or "friend"

        return f"""
        You are writing a three-panel kids' comic from structured scene notes.
        Be vivid and concrete, avoid stock phrases, and DO NOT repeat words across panels.
        Use the same main character throughout if present. Do not invent new objects.

        SCENE 1: {scenes[0]}
        SCENE 2: {scenes[1]}  DELTA_1_2: {deltas[0] if deltas else {}}
        SCENE 3: {scenes[2]}  DELTA_2_3: {deltas[1] if len(deltas)>1 else {}}

        Required nouns (use naturally, not as a list): {", ".join(sorted({w for s in scenes for w in [s.get("subject","")] + s.get("objects",[]) if w}))}
        Tone: {mood}. End with a satisfying consequence, not a moral.

        FORMAT (exactly):
        Title: <short specific title that mentions a key object or character>
        Panel 1: <one vivid sentence that sets up the situation and place>
        Panel 2: <one sentence that shows a change or small problem>
        Panel 3: <one sentence that resolves the change and connects to panel 1/2>
        """.strip()


    def _parse_story(self, text: str) -> Tuple[str, List[str]]:
        title = ""
        panels = ["","",""]
        for line in (text or "").splitlines():
            line=line.strip()
            if not line: continue
            up=line.upper()
            if up.startswith("TITLE:"): title = line.split(":",1)[1].strip()
            elif up.startswith("PANEL 1:"): panels[0] = line.split(":",1)[1].strip()
            elif up.startswith("PANEL 2:"): panels[1] = line.split(":",1)[1].strip()
            elif up.startswith("PANEL 3:"): panels[2] = line.split(":",1)[1].strip()
        return (title or ""), [p or "" for p in panels]

    def generate_from_scenes(self, scenes: List[Dict[str,Any]], deltas: List[Dict[str,Any]], mood: str="friendly") -> Tuple[str, List[str], str]:
        # deterministic plan
        title_det, panels_det, moral_det = NarrativePlanner.plan(scenes, deltas)

        # optional LLM polish (disabled by default; enable with ENABLE_STORY_LLM=1)
        try:
            import os as _os
            if self.ready and self._pipe is not None and _os.getenv("ENABLE_STORY_LLM", "0") == "1":
                prompt = self._prompt_from_scenes(scenes, deltas, mood=mood)
                if self._mode == "text2text":
                    out = self._pipe(prompt, max_length=256, num_beams=3)
                    text = out[0]["generated_text"]
                else:
                    out = self._pipe(prompt, max_length=220, do_sample=True, temperature=0.8)
                    text = out[0]["generated_text"]
                title_llm, panels_llm = self._parse_story(text)

                # reject empty/captiony outputs
                def bad(s: str) -> bool:
                    if not s: return True
                    if len(s.split()) <= 3: return True
                    return bool(re.search(r"\b(image|drawing|picture|clip)\b", s, re.I))

                if title_llm and all(not bad(p) for p in panels_llm):
                    return title_llm, panels_llm, moral_det
        except Exception:
            pass

        return title_det, panels_det, moral_det

    # compatibility wrapper (kept for tests/legacy calls)
    def generate(self, caps: List[str]) -> Tuple[str, str, List[str]]:
        title, panels = self._compose_from_captions(caps)
        story = "\n".join(panels)
        return title, story, panels

    @staticmethod
    def _strip_medium_words(s: str) -> str:
        t = (s or "").strip()
        t = re.sub(r"\b(very|really|quite|basically|actually)\b", "", t, flags=re.I)
        t = re.sub(r"\s{2,}", " ", t)
        return t.strip()

    @staticmethod
    def _compose_from_captions(caps: List[str]) -> tuple[str, List[str]]:
        # legacy simple composition (unused by new route, but left for compatibility)
        s = [StoryGenerator._strip_medium_words(c) for c in caps]
        text = " ".join(s)
        has_sun  = "sun" in text
        has_rain = "rain" in text or "umbrella" in text
        has_tree = "tree" in text
        has_dog  = "dog" in text
        parts = []
        if has_sun:  parts.append("SUN")
        if has_rain: parts.append("RAIN")
        if has_tree: parts.append("TREE")
        if has_dog:  parts.append("DOG")
        if not parts:
            words_local = [w for w in re.findall(r"[a-z]+", s[0]) if len(w) > 2][:2]
            parts = [w.upper() for w in words_local] or ["A", "DAY"]
        title = (" ".join(parts) + " ADVENTURE")[:80]

        def clause(i: int, t: str) -> str:
            t = t.rstrip(".!?")
            t = re.sub(r"\b(two|2)\s+people\b", "two friends", t)
            t = re.sub(r"\b(man and woman|boy and girl|man and boy|woman and girl)\b", "two friends", t)
            if i == 0:
                return f"In the first picture, we see {t}."
            if i == 1:
                return f"Then, rain begins and they share an umbrella." if has_rain else f"Then, {t}."
            end = []
            if has_tree: end.append("they rest under a tree")
            if has_dog:  end.append("a friendly dog joins them")
            if has_sun and not has_rain: end.append("the bright sun warms the sky")
            return f"Finally, {(end[0] if end else t)}."
        panels = [clause(0, s[0] or "two friends outside"),
                  clause(1, s[1] or "they play together"),
                  clause(2, s[2] or "they smile and wave")]
        return title, panels
