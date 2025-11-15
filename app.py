"""
Florence-2 LoRA Dataset Builder

Gradio app that:
- Loads a Florence-2 vision-language model
- Auto-captions images
- Cleans captions into LoRA-style tags or natural sentences
- Renames and copies images + .txt files into a training dataset folder
"""

import os
import shutil
import gradio as gr
import re
from typing import List
from collections import OrderedDict

# --- Set Local Cache Directory ---
LOCAL_CACHE_DIR = os.path.join(os.getcwd(), "models_cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
os.environ["HUGGINGFACE_HUB_CACHE"] = LOCAL_CACHE_DIR
os.environ["HF_HOME"] = LOCAL_CACHE_DIR
print(f"--- Models will be downloaded to: {LOCAL_CACHE_DIR} ---")

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
)

# 🔧 GLOBAL PATCH
if not hasattr(PreTrainedModel, "_supports_sdpa"):
    PreTrainedModel._supports_sdpa = False

# --- Model Choices File Handling ---
MODELS_FILENAME = "models.txt"
DEFAULT_MODELS = {
    "Florence-2 (Large)": "microsoft/Florence-2-large",
    "Florence-2 (Base)": "microsoft/Florence-2-base",
}

# --- Output Style Choices ---
CAPTION_STYLES = [
    "Tags (for LoRA)",
    "Natural Sentence"
]

def load_model_choices(filename):
    # ... (Unchanged) ...
    if not os.path.exists(filename):
        print(f"'{filename}' not found. Creating with default models...")
        try:
            with open(filename, 'w') as f:
                for name, model_id in DEFAULT_MODELS.items():
                    f.write(f"{name},{model_id}\n")
        except Exception as e:
            print(f"Error creating {filename}: {e}. Using defaults.")
            return DEFAULT_MODELS
    choices = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ',' in line:
                    parts = line.split(',', 1) 
                    if len(parts) == 2:
                        name = parts[0].strip()
                        model_id = parts[1].strip()
                        if name and model_id:
                            choices[name] = model_id
    except Exception as e:
        print(f"Error reading {filename}: {e}. Using defaults.")
        return DEFAULT_MODELS
    if not choices:
        print(f"'{filename}' was empty or invalid. Using defaults.")
        return DEFAULT_MODELS
    print(f"Successfully loaded {len(choices)} models from {filename}.")
    return choices

MODEL_CHOICES = load_model_choices(MODELS_FILENAME)

def load_model(model_name_key):
    model_id = MODEL_CHOICES.get(model_name_key)
    if not model_id:
        return None, f"Error: Model key '{model_name_key}' not found.", gr.Button(interactive=False)

    print(f"Loading model: {model_id}...")
    log_message = f"Loading model: {model_id}..."

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        captioner = {"processor": processor, "model": model}
        log_message = f"Success! Loaded model: {model_name_key} (4-bit, auto device map)"
        print("Model loaded successfully.")
        return captioner, log_message, gr.Button(interactive=True)

    except Exception as e:
        print(f"[!] 4-bit GPU load failed: {e}")
        # CPU fallback
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            captioner = {"processor": processor, "model": model}
            log_message = (
                f"Loaded {model_name_key} without 4-bit quantization. "
                "This may be slower, especially on CPU."
            )
            return captioner, log_message, gr.Button(interactive=True)
        except Exception as e2:
            print(f"Error loading model (fallback): {e2}")
            return None, f"Error loading model: {e2}", gr.Button(interactive=False)


def generate_caption(captioner_dict, image_path):
    # ... (Unchanged) ...
    if not captioner_dict:
        raise Exception("Model is not loaded.")
    processor = captioner_dict["processor"]
    model = captioner_dict["model"]
    task_prompt = "<MORE_DETAILED_CAPTION>"
    try:
        image = Image.open(image_path)
        inputs = processor(text=task_prompt, images=image, return_tensors="pt")
        inputs = {
            k: v.to(model.device, dtype=torch.float16 if k == "pixel_values" else v.dtype)
            for k, v in inputs.items()
        }
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_text = processor.post_process_generation(
            generated_text, task=task_prompt, image_size=image.size
        )
        caption = list(parsed_text.values())[0]
        return caption
    except Exception as e:
        print(f"  Error captioning {image_path}: {e}")
        return None

def post_process_tags(tags: list[str]) -> list[str]:
    """
    Final cleanup pass on tag lists:
      - drop junk fragments (for, it appears, be sunny, etc.)
      - normalize a few synonym groups (smiling / slight smile, looking at viewer / looking directly)
      - keep order mostly intact and avoid duplicates
    """
    # Normalize whitespace
    tags = [t.strip() for t in tags if t and t.strip()]
    if not tags:
        return []

    # 1) Exact junk tags to drop completely
    junk_exact = {
        "for",
        "it appears",
        "appears to be",
        "it appears to be",
        "be sunny",
        "be naked",
        "be lively",
        "her back",
        "her body",
        "be tropical",
        "be summery",
        "behind her",
        "other tropical plants",
        "slightly turned",
        "inviting atmosphere",
    }

    # 2) Junk prefixes (if tag starts with any of these, drop it)
    junk_prefixes = (
        "it appears",
        "appears to be",
        "weather appears",
        "be ",
        "her ",
    )

    # 3) Synonym groups (we will keep only the canonical tag)
    #    Order: first element is the canonical tag
    synonym_groups = [
        ["smiling", "slight smile", "small smile"],
        ["looking at viewer", "looking directly", "looking at the camera", "looking at camera"],
    ]

    # Build a quick lookup map: synonym -> canonical
    synonym_to_canonical: dict[str, str] = {}
    for group in synonym_groups:
        canonical = group[0]
        for alt in group:
            synonym_to_canonical[alt] = canonical

    # Starting set of all tags
    original_set = set(tags)

    # Determine which canonical tags should be present
    canonical_to_add: set[str] = set()
    for group in synonym_groups:
        canonical = group[0]
        # If any member of this group exists, ensure canonical is in the final set
        if any(member in original_set for member in group):
            canonical_to_add.add(canonical)

    cleaned: list[str] = []
    already_added: set[str] = set()

    for tag in tags:
        t = tag.strip()
        if not t:
            continue

        # Drop junk exact
        if t in junk_exact:
            continue

        # Drop junk prefixes
        if any(t.startswith(prefix) for prefix in junk_prefixes):
            continue

        # Skip plain "standing" if there is a more detailed standing tag
        if t == "standing":
            if any(
                ("standing" in x) and (x != "standing")
                for x in original_set
            ):
                continue

        # Replace synonyms with canonical form
        if t in synonym_to_canonical:
            t = synonym_to_canonical[t]

        # Only add once
        if t in already_added:
            continue

        cleaned.append(t)
        already_added.add(t)

    # Ensure canonical tags are present if needed
    for canonical in canonical_to_add:
        if canonical not in already_added:
            cleaned.append(canonical)
            already_added.add(canonical)

    return cleaned


def clean_lora_tags(raw_text: str, default_subject_tag: str | None = None) -> str:
    """
    Final, aggressive conversion of natural language sentences into high-quality LoRA tags.
    - Normalizes Florence captions into short tags
    - Removes meta/filler/junk
    - Optional: injects a default subject tag like '1girl'
    - Normalizes NSFW phrases to 'topless' / 'nude'
    """
    if not raw_text:
        return ""

    # --- 0. Lowercase and capture NSFW info BEFORE stripping phrases ---
    text = raw_text.lower()

    nsfw_tag = None
    # Map various phrases to a simple nsfw tag
    if "breasts exposed" in text or "showing off her breasts" in text:
        nsfw_tag = "topless"
    if any(p in text for p in [
        "completely naked",
        "fully naked",
        "totally naked",
        "completely nude",
        "completely unclothed",
        "appears to be naked",
        "is naked"
    ]):
        nsfw_tag = "nude"

    # 1. Remove meta-phrases (about 'this image', 'this photo', etc.)
    meta_phrases = [
        r"this image shows", r"the image shows", r"this photo shows", r"the photo shows",
        r"this image is", r"the image is", r"this photo is", r"the photo is",
        r"a picture of", r"a photo of", r"an image of",
        r"this image", r"the image", r"this photo", r"the photo",
        r"in the background", r"in the foreground", r"on the left", r"on the right",
        r"a close-up of", r"a close up of", r"a full shot of"
    ]
    for phrase in meta_phrases:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", "", text, flags=re.IGNORECASE)

    # 2. Convert sentence structure to commas
    text = text.replace(".", ",").replace(";", ",")
    connectors_and_verbs = [
        # Main connectors
        " and ", " with ", " on ", " in ", " at ", " of ", " to ", " from ",
        # Verbs
        " is ", " are ", " was ", " were ", " has ", " have ", " had ",
        " wearing ", " holding ", " posing ", " sitting ", " standing ",
        # Problem phrases
        " that ", " which ", " but ", " appears to be "
    ]
    for item in connectors_and_verbs:
        text = text.replace(item, ", ")

    # 3. Handle specific problem phrases
    text = text.replace("her back to the camera", "from behind, looking at viewer")
    text = text.replace("his back to the camera", "from behind, looking at viewer")
    text = text.replace("looking at the camera", "looking at viewer")
    text = text.replace("facing the camera", "looking at viewer")

    # 3b. Extra multi-word junk removal (phrases we never want as tags)
    MULTI_JUNK_PHRASES = [
        "she is",
        "he is",
        "the woman",
        "the girl",
        "the man",
        "her face",
        "his face",
        "her head",
        "his head",
        "the camera",
        "the overall mood",
        "can be seen",
        "the distance"
    ]
    # Also strip long NSFW phrases now that we've recorded them
    MULTI_JUNK_PHRASES += [
        "her breasts exposed",
        "showing off her breasts",
        "completely naked",
        "fully naked",
        "totally naked",
        "completely nude",
        "completely unclothed",
        "appears to be naked"
    ]
    for phrase in MULTI_JUNK_PHRASES:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", "", text)

    # 4. Split by comma
    raw_tags = text.split(',')

    # 5. Clean each individual tag
    cleaned_tags = []

    # Words/phrases that are useless even if they survive splitting
    JUNK_WORDS = {
        # Verbs/Connectors
        "is", "are", "was", "were", "has", "have", "had", "be", "being", "been",
        "and", "with", "on", "in", "at", "of", "to", "from",

        # Pronouns
        "she", "he", "they", "it", "her", "his", "their", "its", "them", "whom",

        # Articles/Helpers
        "a", "an", "the", "this", "that", "these", "those",
        "but", "which",

        # Vague/Junk
        "some", "few", "other", "another", "there", "appears", "visible", "seen",
        "distance", "mood", "atmosphere", "overall", "lively", "energetic",
        "sensual", "playful", "summery", "tropical", "styled", "clapping hands",
        "also", "top", "side", "left", "right", "front", "back", "face",
        "head", "body", "hands", "surface", "painted", "placed", "setting",
        "sunny", "warm", "weather" # Added from previous polish
    }

    # Body-part tags we usually don't want for a character LoRA
    BODY_PART_PHRASES = [
        "left hand", "right hand", "hand", "hands",
        "arm", "arms",
        "hip", "hips",
        "shoulder", "shoulders",
        "chest",
        "waist",
        "side",
        "body"
    ]

    # Scene filler words (optional, but keeps tags focused)
    FILLER_WORDS = {
        "wind", "relaxed", "inviting", "creating" # 'sunny'/'warm' moved to JUNK
    }

    for tag in raw_tags:
        tag = tag.strip()

        # Remove leading articles
        if tag.startswith("a "):
            tag = tag[2:]
        elif tag.startswith("an "):
            tag = tag[3:]
        elif tag.startswith("the "):
            tag = tag[4:]

        # Remove possessives
        tag = tag.replace("'s", "")

        # Remove punctuation
        tag = re.sub(r"[^\w\s-]", "", tag)
        tag = tag.strip()

        if not tag:
            continue

        # Skip pure junk words
        if tag in JUNK_WORDS or tag in FILLER_WORDS:
            continue

        # Skip simple pronoun/article safeguards
        if tag in {"a", "an", "the", "she", "he", "they", "it", "her", "his", "there"}:
            continue

        # Skip if it contains body-part phrases
        if any(bp in tag for bp in BODY_PART_PHRASES):
            continue

        # Discard if it's too long (likely junky clause)
        if len(tag.split()) > 3:  # <-- 3-word limit
            continue

        # Skip very short, meaningless tags
        if len(tag) <= 2 and tag not in {"v", "v-neck"}:
            continue

        cleaned_tags.append(tag)

    # 6. De-duplicate while preserving order
    unique_tags = list(OrderedDict.fromkeys(cleaned_tags))

    # 7. Add default subject tag (e.g., "1girl") if provided and missing
    if default_subject_tag:
        default_tag_clean = default_subject_tag.strip().lower()
        if default_tag_clean:
            
            # First, check if the exact default tag is already present
            has_subject = default_tag_clean in unique_tags
            
            # If not, check for general subject keywords *within* any existing tag
            if not has_subject:
                SUBJECT_KEYWORDS = [
                    "girl", "woman", "boy", "man", "person", "couple",
                    "digital illustration", "glass bottle", "cat", "dog"
                ]
                has_subject = any(
                    keyword in tag for tag in unique_tags for keyword in SUBJECT_KEYWORDS
                )
            
            if not has_subject:
                unique_tags.insert(0, default_tag_clean)

    # 8. Add NSFW tag if we detected it and it's not already there
    if nsfw_tag:
        if nsfw_tag not in unique_tags:
            # If we inserted a subject tag, put nsfw right after it
            if default_subject_tag and default_subject_tag.strip():
                unique_tags.insert(1, nsfw_tag)
            else:
                unique_tags.insert(0, nsfw_tag)

    # 9. Final cleanup pass (drop fragments, merge synonyms, etc.)
    unique_tags = post_process_tags(unique_tags)

    return ", ".join(unique_tags)


def simple_clean_caption(text: str) -> str:
    """
    Take Florence's natural-language caption and:
    - remove meta phrases like 'this image', 'the photo'
    - remove only useless verbs (is, are, shows)
    - lightly turn it into a comma-separated description
    """
    if not text:
        return text
    t = text
    meta_phrases = [
        "this image", "the image", "this photo", "the photo", "this picture",
        "the picture", "in this image", "in the image", "in this photo", "in the photo",
    ]
    for p in meta_phrases:
        t = re.sub(r"\b" + re.escape(p) + r"\b", "", t, flags=re.IGNORECASE)
    bad_verbs = {
        "is", "are", "was", "were", "be", "being", "been",
        "shows", "showing",
    }
    parts = re.split(r"(\W+)", t)
    filtered_parts = []
    for part in parts:
        if part.strip().lower() in bad_verbs:
            continue
        filtered_parts.append(part)
    t = "".join(filtered_parts)
    t = re.sub(r"\s+(and|with|while|as|along|near|on|in|at)\s+",
               ", ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.")
    return t

def validate_paths(input_dir: str, dataset_dir: str) -> tuple[bool, str]:
    """Validate input/output directory paths before processing."""
    if not input_dir.strip():
        return False, "Input folder path is empty."
    if not dataset_dir.strip():
        return False, "Dataset output path is empty."

    # Must be different folders
    if os.path.abspath(input_dir) == os.path.abspath(dataset_dir):
        return False, "Input and output folders must be different."

    # Input folder must exist
    if not os.path.isdir(input_dir):
        return False, f"Input folder not found: {input_dir}"

    return True, ""

def process_and_organize_images(
    captioner_dict,
    input_dir,
    dataset_dir,
    min_res,
    prefix,
    caption_style,
    keyword_prefix,
    common_phrase,
    default_subject_tag,
):
    """
    Walk input_dir, caption each image, clean caption, copy image + create .txt in dataset_dir.
    """
    if not captioner_dict:
        return "ERROR: No model is loaded. Please select and load a model first."

    log_messages: List[str] = []
    log_messages.append(f"Starting dataset processing with style: {caption_style}")

    # Validate paths
    ok, message = validate_paths(input_dir, dataset_dir)
    if not ok:
        log_messages.append(f"ERROR: {message}")
        return "\n".join(log_messages)

    os.makedirs(dataset_dir, exist_ok=True)

    # Collect image files (sorted for deterministic order)
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"))
    ]
    total = len(image_files)
    if total == 0:
        log_messages.append("No image files found in input folder.")
        return "\n".join(log_messages)

    # --- FIX: Start file numbering from the highest existing number ---
    try:
        # 1. Look for existing files
        existing_files = [
            f for f in os.listdir(dataset_dir)
            if f.startswith(prefix + "_") and (f.lower().endswith(('.txt', '.png')))
        ]
        
        if existing_files:
            # 2. Find the highest number
            last_num = max([
                int(f.replace(prefix + "_", "").split('.')[0])
                for f in existing_files
            ])
            # 3. Start counting from the next number
            counter = last_num + 1
        else:
            # 4. Folder is empty, start at 1
            counter = 1
            
    except ValueError:
        log_messages.append("  [!] Warning: Could not parse existing filenames. Starting from 1.")
        counter = 1
    
    log_messages.append(f"Starting file numbering from: {counter:04d}")
    # --- END FIX ---
    
    processed_count = 0

    for idx, filename in enumerate(sorted(image_files), start=1):
        original_image_path = os.path.join(input_dir, filename)
        log_messages.append(f"\n--- Processing ({idx}/{total}): {filename} ---")

        try:
            with Image.open(original_image_path) as im:
                width, height = im.size
        except Exception as e:
            log_messages.append(f"  [ERROR] Unable to open image '{filename}': {e}")
            continue

        if width < min_res or height < min_res:
            log_messages.append(
                f"  [!] WARNING: '{filename}' is {width}x{height}. "
                f"Below minimum {min_res}x{min_res}."
            )

        # Generate caption
        try:
            raw_caption = generate_caption(captioner_dict, original_image_path)
            log_messages.append(f"  Raw Output: '{raw_caption}'")
        except Exception as e:
            log_messages.append(f"  [ERROR] Caption generation failed for '{filename}': {e}")
            continue

        # Clean caption
        if caption_style == "Tags (for LoRA)":
            
            # --- FIX START ---
            # 1. Generate the main tags first
            tags_string = clean_lora_tags(
                raw_caption,
                default_subject_tag=default_subject_tag
            )
            
            # 2. Manually build the prefix list
            prefix_tags = []
            if keyword_prefix and keyword_prefix.strip():
                prefix_tags.append(keyword_prefix.strip())
            if common_phrase and common_phrase.strip():
                prefix_tags.append(common_phrase.strip())

            # 3. Combine them (prefixes go first)
            if prefix_tags:
                final_caption = ", ".join(prefix_tags) + ", " + tags_string
            else:
                final_caption = tags_string
            # --- FIX END ---

        else:
            # Natural Sentence
            # --- FIX 2: Corrected function name ---
            final_caption = simple_clean_caption(raw_caption)

        log_messages.append(f"  Final Caption: '{final_caption}'")

        # Build new filename
        new_index_str = f"{counter:04d}"
        new_image_name = f"{prefix}_{new_index_str}.png"
        new_caption_name = f"{prefix}_{new_index_str}.txt"

        new_image_path = os.path.join(dataset_dir, new_image_name)
        new_caption_path = os.path.join(dataset_dir, new_caption_name)

        # Copy image as PNG
        try:
            with Image.open(original_image_path) as im:
                im.convert("RGB").save(new_image_path, format="PNG")
            log_messages.append(
                f"  Success: Copied '{filename}' -> '{new_image_name}'"
            )
        except Exception as e:
            log_messages.append(f"  [ERROR] Failed to copy '{filename}': {e}")
            continue

        # Write caption
        try:
            with open(new_caption_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(final_caption)
            log_messages.append(
                f"  Success: Created caption -> '{new_caption_name}'"
            )
        except Exception as e:
            log_messages.append(
                f"  [ERROR] Failed to write caption for '{filename}': {e}"
            )
            continue

        counter += 1
        processed_count += 1

    log_messages.append(
        f"\n--- Processing Complete ---\nProcessed and copied {processed_count} new images."
    )
    return "\n".join(log_messages)

# --- MODIFIED: Gradio UI Interface ---
with gr.Blocks() as app:
    model_state = gr.State(None) 

    gr.Markdown("# LORA Dataset Builder (Florence-2 Edition)")
    gr.Markdown(
    "This tool can caption SFW and NSFW images. "
    "**You are responsible** for complying with model licenses, platform rules, and local law."
)

    with gr.Group():
        # ... (Unchanged) ...
        gr.Markdown("### 1. Load Model (Run this first!)")
        gr.Markdown(
            f"Models are loaded from `{MODELS_FILENAME}`. "
            "You MUST `pip install timm` and `pip install bitsandbytes`."
        )
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_CHOICES.keys()),
                value=list(MODEL_CHOICES.keys())[0] if MODEL_CHOICES else None,
                label="Select Florence-2 Model"
            )
            load_model_btn = gr.Button("Load Selected Model", variant="primary")
        model_status_box = gr.Textbox(label="Model Status", interactive=False)
    
    with gr.Column():
            gr.Markdown("### 2. Set Options")
            # ... (Unchanged) ...
            gr.Markdown("Provide the **full path** to your folders.")
            input_dir_box = gr.Textbox(value="input_images", label="Input Folder Path")
            dataset_dir_box = gr.Textbox(value="my_lora_dataset", label="Dataset Output Path")
            
            gr.Markdown("### 3. Captioning Options")
            
            caption_style_dropdown = gr.Dropdown(
                choices=CAPTION_STYLES,
                value="Tags (for LoRA)",
                label="Output Style"
            )
            
            keyword_box = gr.Textbox(
                label="Keyword Prefix", 
                placeholder="e.g., my_lora_keyword"
            )
            common_phrase_box = gr.Textbox(
                label="Common Phrase", 
                placeholder="e.g., a photo of a person"
            )
            
            # --- NEW TEXTBOX ---
            default_subject_box = gr.Textbox(
                label="Default Subject Tag (for 'Tags' style)",
                placeholder="e.g., 1girl (Leave blank to add no default tag)"
            )
            
            gr.Markdown("### 4. File Options")
            # ... (Unchanged) ...
            min_res_box = gr.Number(value=562, label="Minimum Resolution (px)")
            prefix_box = gr.Textbox(value="lora_img", label="File Prefix")
            
            start_button = gr.Button(
                "Start Processing", 
                variant="primary", 
                interactive=False
            )

    with gr.Column():
        gr.Markdown("### 5. Processing Log")
        # ... (Unchanged) ...
        log_output_box = gr.Textbox(
            label="Log",
            lines=20,
            interactive=False,
            autoscroll=True
        )
            
    load_model_btn.click(
        fn=load_model,
        inputs=[model_dropdown],
        outputs=[model_state, model_status_box, start_button]
    )
    
    start_button.click(
        fn=process_and_organize_images,
        # --- MODIFIED: Added default_subject_box ---
        inputs=[
            model_state, input_dir_box, dataset_dir_box, min_res_box, 
            prefix_box, caption_style_dropdown, keyword_box, common_phrase_box,
            default_subject_box
        ],
        outputs=[log_output_box]
    )

# --- Launch the App ---
if __name__ == "__main__":
    port = int(os.environ.get("FLORENCE_LORA_PORT", "8123"))
    share = os.environ.get("FLORENCE_LORA_SHARE", "false").lower() == "true"
    app.launch(server_port=port, share=share)
