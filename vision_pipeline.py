import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from collections import Counter
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# config
VIDEO_PATH = 'input_video.mp4'
OUTPUT_DIR = 'extracted_frames'
REQUIRED_FPS = 0.25
JPEG_QUALITY = 95

# Load models
model = YOLO('yolov8n.pt')
model.fuse()

# BLIP for captions
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

vid = cv2.VideoCapture(VIDEO_PATH)

if not vid.isOpened():
    raise ValueError(f"Can't open {VIDEO_PATH}")

# Video properties
fps = vid.get(cv2.CAP_PROP_FPS)
total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    raise ValueError("Invalid FPS")

# Frame interval
frame_interval = max(1, int(fps / REQUIRED_FPS))

frame_summary = {}
video_knowledge_base = []

# Frames to process
frames_to_process = list(range(0, total_frames, frame_interval))
total_to_process = len(frames_to_process)

print(f"Processing {total_to_process} frames...")

# Process frames
for idx, frame_idx in enumerate(frames_to_process, 1):
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = vid.read()

    if not ret:
        print(f"Can't read frame {frame_idx}")
        break

    # Timestamp
    timestamp = frame_idx / fps
    frame_filename = os.path.join(OUTPUT_DIR, f'frame_{int(timestamp)}s.jpg')
    
    # YOLO detection
    results = model(frame, verbose=False, imgsz=640, conf=0.25)

    # Extract detections
    if len(results[0].boxes) > 0:
        cls_ids = results[0].boxes.cls.tolist()
        counts = Counter(cls_ids)
        class_count_dict = {
            results[0].names[int(cls_id)]: count
            for cls_id, count in counts.items()
        }
    else:
        class_count_dict = {}

    # Generate caption
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    prompt_text = "a photo of"
    inputs = blip_processor(pil_image,text=prompt_text, return_tensors="pt")
    caption_ids = blip_model.generate(**inputs, max_new_tokens=50, num_beams=5, temperature=0.7)
    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)

    # Build summary
    if class_count_dict:
        detection_summary = "Objects detected: " + ", ".join(
            [f"{count} {obj_class}" for obj_class, count in class_count_dict.items()]
        )
    else:
        detection_summary = "Objects detected: None"
    
    # Combine content
    content = f"{caption}. {detection_summary}."
    
    # Store summary
    frame_summary[int(timestamp)] = {"counts": class_count_dict, "caption": caption}
    
    # Add to knowledge base
    knowledge_entry = {
        "content": content,
        "metadata": {
            "source": VIDEO_PATH,
            "timestamp": round(timestamp, 2),
            "frame_id": frame_idx,
            "objects_found": list(class_count_dict.keys()) if class_count_dict else []
        }
    }
    video_knowledge_base.append(knowledge_entry)

    # Optional: save frame
    #cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    # Progress
    if idx % max(1, total_to_process // 10) == 0 or idx == total_to_process:
        print(f"{idx}/{total_to_process} ({idx*100//total_to_process}%)")

vid.release()

# Save files
summary_file = os.path.join(OUTPUT_DIR, 'detection_summary.json')
with open(summary_file, 'w') as f:
    json.dump(frame_summary, f, indent=2)

knowledge_base_file = os.path.join(OUTPUT_DIR, 'video_knowledge_base.json')
with open(knowledge_base_file, 'w') as f:
    json.dump(video_knowledge_base, f, indent=2)

# Stats
total_detections = sum(sum(counts["counts"].values()) for counts in frame_summary.values())
all_classes = set()
for counts in frame_summary.values():
    all_classes.update(counts["counts"].keys())

print(f"\nDone! Processed {len(frame_summary)} frames")
print(f"Saved to {OUTPUT_DIR}")
print(f"Objects found: {total_detections}")
print(f"Classes: {sorted(all_classes)}")

