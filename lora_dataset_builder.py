import os
import shutil
from PIL import Image
from transformers import pipeline

# --- Configuration ---
INPUT_FOLDER = "input_images"
DATASET_FOLDER = "my_lora_dataset"
MIN_RESOLUTION = 562
FILE_PREFIX = "lora_img"  # The base name for your dataset files
# --- End Configuration ---

def get_image_captioner():
    """
    Initializes and returns the image captioning pipeline.
    This way, the model is only loaded once.
    """
    print("Loading captioning model... (This may take a moment)")
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def generate_caption(captioner, image_path):
    """
    Generates a caption for a given image using the pre-loaded model.
    """
    try:
        caption_results = captioner(image_path)
        if caption_results:
            return caption_results[0]['generated_text']
    except Exception as e:
        print(f"  Error captioning {image_path}: {e}")
    return None

def process_and_organize_images(input_dir, dataset_dir, min_res, prefix):
    """
    Processes all images in the input directory, checks resolution,
    generates captions, renames, and moves them to the dataset directory.
    """
    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Load the model once
    captioner = get_image_captioner()
    
    print("\nStarting dataset processing...")
    
    # Start the file numbering from the highest existing number in the dataset
    try:
        existing_files = [f for f in os.listdir(dataset_dir) if f.startswith(prefix) and (f.endswith('.txt') or f.endswith('.png'))]
        if existing_files:
            last_num = max([int(f.split('_')[-1].split('.')[0]) for f in existing_files])
            counter = last_num + 1
        else:
            counter = 1
    except ValueError:
        counter = 1
        
    print(f"Starting file numbering from: {counter:04d}")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            original_image_path = os.path.join(input_dir, filename)
            
            print(f"\n--- Processing: {filename} ---")

            # 1. Resolution Check
            try:
                with Image.open(original_image_path) as img:
                    width, height = img.size
                    if width < min_res or height < min_res:
                        print(f"  [!] WARNING: '{filename}' is {width}x{height}. "
                              f"This is below the {min_res}x{min_res} minimum.")
            except Exception as e:
                print(f"  [!] ERROR: Could not open or read image '{filename}'. Skipping. {e}")
                continue

            # 2. Generate Caption
            caption = generate_caption(captioner, original_image_path)
            if not caption:
                print(f"  [!] ERROR: Could not generate caption for '{filename}'. Skipping.")
                continue
            
            print(f"  Caption: '{caption}'")

            # 3. Rename and Organize
            file_number_str = f"{counter:04d}"  # Formats as 0001, 0002, etc.
            original_ext = os.path.splitext(filename)[1]
            
            new_image_name = f"{prefix}_{file_number_str}{original_ext}"
            new_txt_name = f"{prefix}_{file_number_str}.txt"
            
            new_image_path = os.path.join(dataset_dir, new_image_name)
            new_txt_path = os.path.join(dataset_dir, new_txt_name)
            
            # 4. Write .txt file
            try:
                with open(new_txt_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
            except Exception as e:
                print(f"  [!] ERROR: Could not write txt file '{new_txt_path}'. Skipping. {e}")
                continue

            # 5. Move and Rename Image
            try:
                # Use shutil.move to safely move the file
                shutil.move(original_image_path, new_image_path)
                print(f"  Success: Moved '{filename}' -> '{new_image_name}'")
                print(f"  Success: Created caption -> '{new_txt_name}'")
            except Exception as e:
                print(f"  [!] ERROR: Could not move image file '{original_image_path}'. {e}")
                # Clean up the orphaned .txt file
                if os.path.exists(new_txt_path):
                    os.remove(new_txt_path)
                continue
                
            # Increment counter for the next file
            counter += 1

    print("\n--- Processing Complete ---")
    print(f"All images from '{input_dir}' have been processed and moved to '{dataset_dir}'.")


if __name__ == "__main__":
    # Example Usage: Create dummy images for demonstration
    # (The script will process these the first time you run it)
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    
    # Create a "good" image
    try:
        img_good_path = os.path.join(INPUT_FOLDER, 'test_image_good.png')
        if not os.path.exists(img_good_path):
            img = Image.new('RGB', (600, 600), color = 'green')
            img.save(img_good_path)
            print(f"Created a dummy 'good' image at: {img_good_path}")
    except Exception: pass
    
    # Create a "small" image
    try:
        img_small_path = os.path.join(INPUT_FOLDER, 'test_image_small.png')
        if not os.path.exists(img_small_path):
            img = Image.new('RGB', (100, 100), color = 'red')
            img.save(img_small_path)
            print(f"Created a dummy 'small' image at: {img_small_path}")
    except Exception: pass

    # Run the main processing function
    process_and_organize_images(INPUT_FOLDER, DATASET_FOLDER, MIN_RESOLUTION, FILE_PREFIX)