import json
import os
from PIL import Image
from collections import defaultdict
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm
import csv

# -------- CONFIG --------
IMAGE_DIR = './nycu-hw2-data/test/'  # Folder containing the test images
PRED_FILES = ['pred_resnet50v2.json', 'pred_resnet50.json',
              'pred_mobilenet.json', 'pred_Vgg16.json']
OUTPUT_FILE = 'wbf_ensemble.json'
iou_thr = 0.5
skip_box_thr = 0.001
# ------------------------

WEIGHTS = [2.0, 1.0, 1.0, 1.0]

# Load all predictions
all_preds = [json.load(open(os.path.join("./bagging", f))) for f in PRED_FILES]

# Gather all image_ids
image_ids = set(pred['image_id'] for preds in all_preds for pred in preds)

# Map image_id to filename (best-effort) if image_id is filename-like
image_id_to_size = {}

print("Reading image sizes...")
for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(IMAGE_DIR, filename)
        with Image.open(filepath) as img:
            width, height = img.size

        image_id = filename
        try:
            image_id = int(os.path.splitext(filename)[0])
        except Exception:
            pass

        image_id_to_size[image_id] = (width, height)

# Group predictions by image_id for all 5 models
preds_by_image = defaultdict(lambda: [[] for _ in range(len(PRED_FILES))])

for model_idx, preds in enumerate(all_preds):
    for pred in preds:
        img_id = pred['image_id']
        x, y, w, h = pred['bbox']
        width, height = image_id_to_size[img_id]
        x1 = x / width
        y1 = y / height
        x2 = (x + w) / width
        y2 = (y + h) / height
        preds_by_image[img_id][model_idx].append({
            'box': [x1, y1, x2, y2],
            'score': pred['score'],
            'label': pred['category_id']
        })

# Run WBF
print("Applying Weighted Boxes Fusion...")
fused_results = []

for img_id, model_preds in tqdm(preds_by_image.items()):
    boxes_list, scores_list, labels_list = [], [], []

    for preds in model_preds:
        boxes = [p['box'] for p in preds]
        scores = [p['score'] for p in preds]
        labels = [p['label'] for p in preds]
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=iou_thr, skip_box_thr=skip_box_thr,
        weights=WEIGHTS
    )

    width, height = image_id_to_size[img_id]
    for box, score, label in zip(boxes, scores, labels):
        x1 = box[0] * width
        y1 = box[1] * height
        x2 = box[2] * width
        y2 = box[3] * height
        fused_results.append({
            'image_id': img_id,
            'category_id': int(label),
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'score': float(score)
        })

# Save fused predictions
with open(OUTPUT_FILE, 'w') as f:
    json.dump(fused_results, f)

print(f"Done! Fused predictions saved to {OUTPUT_FILE}")

# record the digit in each image in csv file
# 1. read the json file and concatenate the digits from left to right
# 2. save the result in a csv file

# Load WBF-ensemble JSON
with open('wbf_ensemble.json') as f:
    preds = json.load(f)

# Group predictions by image_id
image_preds = defaultdict(list)
for pred in preds:
    image_id = pred['image_id']
    image_preds[image_id].append(pred)

# Create CSV content
rows = []
for image_id in range(1, 13069):  # 1 to 13068 inclusive
    preds_for_image = image_preds.get(image_id, [])
    if not preds_for_image:
        digits = -1
    else:
        # Sort by x-coordinate (left to right)
        preds_sorted = sorted(preds_for_image, key=lambda x: x['bbox'][0])
        digits = ''.join(str(p['category_id'] - 1) for p in preds_sorted)
    rows.append((image_id, digits))

# Save to CSV
with open('pred_digits.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_id', 'pred_label'])
    writer.writerows(rows)

print("Saved predictions to pred_digits.csv")
