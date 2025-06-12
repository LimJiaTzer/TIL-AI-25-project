import json
import os
import shutil
import random
import concurrent.futures
from ultralytics import RTDETR
from tqdm import tqdm
import torch

# --- Configuration ---
# === PATHS ===
IMAGE_SOURCE_DIR = r"../../../novice/cv/images"
JSON_ANNOTATION_FILE = r"../../../novice/cv/annotations.json"
YOLO_DATASET_BASE_DIR = r"../models"

# === MODEL & TRAINING ===
MODEL_TO_FINETUNE = "rtdetr-x.pt" 
EPOCHS = 100
BATCH_SIZE = 12
IMG_SIZE = 640

# === DATA SPLIT & OPTIMIZATION ===
VALIDATION_SPLIT_RATIO = 0.1
RANDOM_SEED = 42
# ADDED: Use a portion of CPU cores for parallel processing. None uses all available.
NUM_WORKERS = os.cpu_count() // 2  # Use half the CPU cores, adjust if needed

# --- Helper Functions ---
def convert_coco_bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, w, h = bbox
    if img_width <= 0 or img_height <= 0: return None
    x_center = max(0.0, min(1.0, (x_min + w / 2) / img_width))
    y_center = max(0.0, min(1.0, (y_min + h / 2) / img_height))
    norm_w = max(0.0, min(1.0, w / img_width))
    norm_h = max(0.0, min(1.0, h / img_height))
    return x_center, y_center, norm_w, norm_h

def process_image(img_id, image_id_to_info, annotations_by_image_id, cat_id_to_class_idx, dest_paths, split_name):
    """
    Worker function to process a single image: copy it and create its label file.
    This function is designed to be run in a separate process.
    """
    try:
        img_info = image_id_to_info.get(img_id)
        if not img_info: return "info_missing", None

        original_img_filename = img_info['file_name']
        original_img_path = os.path.join(IMAGE_SOURCE_DIR, original_img_filename)

        if not os.path.exists(original_img_path):
            return "image_missing", original_img_path

        # Copy image
        img_dest_dir = dest_paths[f"{split_name}_images"]
        shutil.copy(original_img_path, img_dest_dir)

        # Create label file
        img_w, img_h = img_info.get('width'), img_info.get('height')
        if not all([img_w, img_h, img_w > 0, img_h > 0]):
            return "invalid_dims", original_img_filename

        base_filename, _ = os.path.splitext(original_img_filename)
        label_filename = f"{base_filename}.txt"
        label_path = os.path.join(dest_paths[f"{split_name}_labels"], label_filename)

        with open(label_path, 'w') as lf:
            if img_id in annotations_by_image_id:
                for ann in annotations_by_image_id[img_id]:
                    coco_bbox, cat_id = ann.get('bbox'), ann.get('category_id')
                    if coco_bbox is None or cat_id is None: continue

                    class_idx = cat_id_to_class_idx.get(cat_id)
                    if class_idx is None: continue

                    yolo_bbox = convert_coco_bbox_to_yolo(coco_bbox, img_w, img_h)
                    if yolo_bbox is None:
                        return "bbox_error", original_img_filename
                    
                    x_c, y_c, w_n, h_n = yolo_bbox
                    lf.write(f"{class_idx} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
        return "success", None
    except Exception as e:
        return "exception", str(e)

# --- Main Script ---
def prepare_dataset_and_finetune():
    print("--- Starting Dataset Preparation and Fine-tuning ---")
    random.seed(RANDOM_SEED)
    data_yaml_path = os.path.join(YOLO_DATASET_BASE_DIR, "data.yaml")

    # --- ADDED: CACHING LOGIC ---
    # If the final yaml file and directories exist, skip straight to training
    if os.path.exists(data_yaml_path) and os.path.isdir(os.path.join(YOLO_DATASET_BASE_DIR, "images", "train")):
        print("\n[INFO] Found existing processed dataset. Skipping preparation.")
    else:
        print("\n[INFO] No cached dataset found. Starting full preparation...")
        # 1. Load JSON Annotations
        print(f"\n[1/6] Loading annotations from: {JSON_ANNOTATION_FILE}")
        if not os.path.exists(JSON_ANNOTATION_FILE):
            print(f"ERROR: Annotation file not found at {JSON_ANNOTATION_FILE}"); return
        with open(JSON_ANNOTATION_FILE, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        images_list = coco_data['images']
        annotations_list = coco_data['annotations']
        categories_list = coco_data['categories']
        print(f"   Loaded {len(images_list)} images, {len(annotations_list)} annotations, {len(categories_list)} categories.")

        # 2. Prepare Mappings
        print("\n[2/6] Preparing data mappings...")
        image_id_to_info = {img['id']: img for img in images_list}
        class_names = [cat['name'] for cat in sorted(categories_list, key=lambda x: x['id'])]
        cat_id_to_class_idx = {cat['id']: i for i, cat in enumerate(sorted(categories_list, key=lambda x: x['id']))}
        annotations_by_image_id = {}
        for ann in annotations_list:
            annotations_by_image_id.setdefault(ann['image_id'], []).append(ann)
        print("   Mappings created.")

        # 3. Prepare YOLO dataset directories
        print(f"\n[3/6] Preparing YOLO dataset directory structure...")
        if os.path.exists(YOLO_DATASET_BASE_DIR):
            shutil.rmtree(YOLO_DATASET_BASE_DIR)
        
        paths = {
            "train_images": os.path.join(YOLO_DATASET_BASE_DIR, "images", "train"),
            "val_images": os.path.join(YOLO_DATASET_BASE_DIR, "images", "val"),
            "train_labels": os.path.join(YOLO_DATASET_BASE_DIR, "labels", "train"),
            "val_labels": os.path.join(YOLO_DATASET_BASE_DIR, "labels", "val"),
        }
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
        print("   Directory structure created.")

        # 4. Split data
        print("\n[4/6] Splitting data into Training and Validation sets...")
        all_image_ids = list(image_id_to_info.keys())
        random.shuffle(all_image_ids)
        num_val = int(len(all_image_ids) * VALIDATION_SPLIT_RATIO)
        val_ids = set(all_image_ids[:num_val])
        train_ids = set(all_image_ids[num_val:])
        print(f"   Training images: {len(train_ids)}, Validation images: {len(val_ids)}")

        # 5. Process and write data in parallel
        print("\n[5/6] Processing images and creating YOLO label files (in parallel)...")
        for split_name, image_ids in [("train", train_ids), ("val", val_ids)]:
            print(f"--- Processing {split_name} data ({len(image_ids)} images) ---")
            error_counts = {"image_missing": 0, "bbox_error": 0, "exception": 0}
            with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # Create a list of futures
                futures = [executor.submit(process_image, img_id, image_id_to_info, annotations_by_image_id, cat_id_to_class_idx, paths, split_name) for img_id in image_ids]
                # Process results with tqdm progress bar
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Copying {split_name} files"):
                    status, _ = future.result()
                    if status != "success":
                        error_counts[status] = error_counts.get(status, 0) + 1
            
            for error_type, count in error_counts.items():
                if count > 0:
                    print(f"   Warning: Encountered {count} '{error_type}' errors during {split_name} processing.")

        # 6. Create data.yaml
        print("\n[6/6] Creating data.yaml configuration file...")
        data_yaml_content = f"""
train: {os.path.abspath(paths['train_images'])}
val: {os.path.abspath(paths['val_images'])}
nc: {len(class_names)}
names: {class_names}
"""
        with open(data_yaml_path, 'w') as f:
            f.write(data_yaml_content)
        print(f"   Successfully created data.yaml.")

    # 7. Fine-tune the model
    print(f"\n[INFO] Starting fine-tuning with {MODEL_TO_FINETUNE}...")
    try:
        model = RTDETR(MODEL_TO_FINETUNE)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Training on device: {device}")
        
        results = model.train(
            data=data_yaml_path, epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, device=device,
            project=os.path.join(YOLO_DATASET_BASE_DIR, "runs"),
            name=f"{os.path.splitext(MODEL_TO_FINETUNE)[0]}_finetune_augmented",
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=15.0, translate=0.1, scale=0.5,
            shear=5.0, perspective=0.0, flipud=0.5, fliplr=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.1
        )
        print("\n--- Fine-tuning complete! ---")
        print(f"   Trained model and results saved in directory: {results.save_dir}")

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

# --- Run the script ---
if __name__ == "__main__":
    prepare_dataset_and_finetune()