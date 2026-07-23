import json
import os
import re

# --- Configuration ---
DATA_DIR = "/home/ara/ros2_ws/src/robust_lidar_inertial/log/splat_data"
INPUT_JSON = os.path.join(DATA_DIR, "transforms.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "transforms_clean.json")

def rescue_slam_data(input_path, output_path):
    print(f"Reading corrupted file: {input_path}")
    try:
        with open(input_path, 'r') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print("File not found.")
        return

    # 1. Extract Global Parameters (camera intrinsics)
    clean_dict = {}
    header_part = raw_text.split('"frames"')[0]
    for line in header_part.split('\n'):
        if ':' in line and not '{' in line:
            parts = line.split(':')
            key = parts[0].replace('"', '').strip()
            val_str = parts[1].replace(',', '').strip()
            try:
                clean_dict[key] = float(val_str) if '.' in val_str else int(val_str)
            except ValueError:
                pass
                
    print(f"Extracted camera intrinsics: {clean_dict}")

    # 2. Extract Frames manually
    frames = []
    chunks = raw_text.split('"file_path":')
    last_frame_num = -1
    
    print("Extracting frames...")
    for chunk in chunks[1:]:  # Skip the first chunk (header)
        # Find the end of this specific frame's object
        end_idx = chunk.find('}')
        if end_idx != -1:
            # Reconstruct a valid mini-JSON for just this frame
            frame_str = '{"file_path":' + chunk[:end_idx+1]
            
            try:
                frame_data = json.loads(frame_str)
                
                # Check for a restart (if frame number goes back to 0)
                filepath = frame_data.get('file_path', '')
                num_match = re.search(r'frame_(\d+)', filepath)
                if num_match:
                    frame_num = int(num_match.group(1))
                    if frame_num <= last_frame_num:
                        print(f"--> Detected SLAM restart at frame {frame_num}. Stopping extraction.")
                        break
                    last_frame_num = frame_num
                
                frames.append(frame_data)
            except Exception:
                # If a frame is half-written, just skip it
                pass

    print(f"Successfully rescued {len(frames)} frames!")
    clean_dict['frames'] = frames

    # 3. Save as a perfectly formatted JSON
    with open(output_path, 'w') as f:
        json.dump(clean_dict, f, indent=2)
    print(f"Saved clean data to: {output_path}")

if __name__ == "__main__":
    rescue_slam_data(INPUT_JSON, OUTPUT_JSON)