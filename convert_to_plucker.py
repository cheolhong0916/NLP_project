import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Use a non-interactive backend to save figures without displaying windows
plt.switch_backend('Agg') 

def get_plucker_rays(H, W, K, c2w):
    """
    Generates Plucker Rays given image resolution, intrinsics, and extrinsics.
    
    Args:
        H, W: Image height and width
        K: (3, 3) Intrinsic Matrix
        c2w: (4, 4) Camera-to-World Matrix (Pose)
        
    Returns:
        rays_d: (H*W, 3) Direction vectors
        rays_m: (H*W, 3) Moment vectors
        rays_o: (H*W, 3) Camera origins
    """
    # 1. Create pixel grid (u, v)
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), 
                       np.arange(H, dtype=np.float32), 
                       indexing='xy')
    
    # 2. Convert pixels to camera coordinates (z=1 plane)
    # OpenCV convention: x-right, y-down, z-forward
    dirs = np.stack([(i - K[0, 2]) / K[0, 0], 
                     (j - K[1, 2]) / K[1, 1], 
                     np.ones_like(i)], -1) # (H, W, 3)

    # 3. Transform from Camera to World coordinates
    # rays_d = R * dirs
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)
    
    # Normalize direction vectors
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    
    # 4. Camera origin (World space)
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape) # (H, W, 3)
    
    # 5. Calculate Plucker Coordinates
    # Moment m = o x d
    rays_m = np.cross(rays_o, rays_d)
    
    return rays_d.reshape(-1, 3), rays_m.reshape(-1, 3), rays_o.reshape(-1, 3)

def save_plucker_visualization(save_path, rays_d, rays_m, rays_o, num_vis=50, scale=1.0):
    """
    Visualizes the rays using Matplotlib and saves the figure to a file.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample rays for visualization to avoid clutter
    indices = np.linspace(0, len(rays_d)-1, num_vis, dtype=int)
    
    # Draw camera origin
    cam_origin = rays_o[0]
    ax.scatter(cam_origin[0], cam_origin[1], cam_origin[2], c='red', s=50, marker='^', label='Cam')

    for idx in indices:
        d = rays_d[idx]
        start = cam_origin
        end = start + d * scale
        
        # Plot ray line
        ax.plot([start[0], end[0]], 
                [start[1], end[1]], 
                [start[2], end[2]], 
                c='blue', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Plucker Rays: {os.path.basename(save_path)}')
    
    # Set axis limits to keep aspect ratio somewhat consistent locally
    axis_limits = [np.min(cam_origin) - scale, np.max(cam_origin) + scale]
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_zlim(axis_limits)
    
    # Save the figure using relative path
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig) # Close memory to prevent leaks during loop

def parse_camera_info(text_content):
    """
    Parses camera intrinsics and extrinsics from the user prompt text.
    
    Args:
        text_content: The string containing 'fx=...', 'rotation quaternion [...]', etc.
        
    Returns:
        K: (3, 3) Intrinsic matrix
        poses: Dictionary mapping image ID (str) to (4, 4) c2w matrix
    """
    # 1. Parse Intrinsics
    # Pattern: fx=1438.070, fy=1439.220, cx=959.841, cy=719.327
    intrinsics_pattern = r"fx=([\d\.]+),\s*fy=([\d\.]+),\s*cx=([\d\.]+),\s*cy=([\d\.]+)"
    match = re.search(intrinsics_pattern, text_content)
    
    if match:
        fx, fy, cx, cy = map(float, match.groups())
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        # Fallback or error if intrinsics are missing
        print("[Warning] Could not parse intrinsics. Using identity.")
        K = np.eye(3, dtype=np.float32)

    # 2. Parse Extrinsics (Poses)
    # Pattern: Image 2360: rotation quaternion [-0.4..., ...] and translation vector [4.4..., ...]
    pose_pattern = r"Image\s+(\d+):\s+rotation quaternion\s+\[(.*?)\]\s+and\s+translation vector\s+\[(.*?)\]"
    matches = re.findall(pose_pattern, text_content)
    
    poses = {}
    for img_id, quat_str, trans_str in matches:
        # Parse quaternion [x, y, z, w] or [w, x, y, z] depending on data source convention.
        # Scipy expects [x, y, z, w]. Assuming the input follows standard convention for this dataset.
        q = np.fromstring(quat_str, sep=',')
        t = np.fromstring(trans_str, sep=',')
        
        # Construct rotation matrix
        # Note: If quaternions are [w, x, y, z], use specific conversion. 
        # Here we assume standard [x, y, z, w] for scipy. 
        # If the visualization looks wrong, check quaternion order.
        rot_mat = R.from_quat(q).as_matrix()
        
        # Construct 4x4 C2W matrix
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rot_mat
        c2w[:3, 3] = t
        
        poses[img_id] = c2w
        
    return K, poses

def process_dataset_files(json_file_list):
    """
    Iterates through a list of JSON files, processes images mentioned in them, 
    and generates Plucker visualizations.
    """
    for json_file in json_file_list:
        print(f"--- Processing JSON: {json_file} ---")
        
        if not os.path.exists(json_file):
            print(f"[Error] File not found: {json_file}")
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Iterate through each sample in the JSON
        for sample_idx, entry in enumerate(data):
            # 1. Extract Camera Info from the first user message
            user_msg = next((m['content'] for m in entry['messages'] if m['role'] == 'user'), None)
            if not user_msg:
                continue
                
            K, poses = parse_camera_info(user_msg)
            
            # 2. Process RGB images listed in 'images'
            if 'images' not in entry:
                continue
                
            for img_path_str in entry['images']:
                # Only process RGB images (usually in 'image_color' folder)
                if "image_color" not in img_path_str:
                    continue
                
                # 3. Setup Paths
                # Ensure relative path usage
                if os.path.isabs(img_path_str):
                     img_path_str = os.path.relpath(img_path_str)
                
                # Extract Image ID from filename (e.g., '2360.jpg' -> '2360')
                img_filename = os.path.basename(img_path_str)
                img_id = os.path.splitext(img_filename)[0]
                
                # Check if we have pose data for this image
                if img_id not in poses:
                    # Some images might not be the primary ones with pose info in text, skip if so
                    # print(f"[Skip] No pose found for {img_id}")
                    continue
                
                c2w = poses[img_id]
                
                # Construct output path
                output_path_str = img_path_str.replace("image_color", "plucker_c2w")
                output_path_str = os.path.splitext(output_path_str)[0] + ".png"
                
                # Skip if already exists (optional, currently overwrites)
                # if os.path.exists(output_path_str): continue

                os.makedirs(os.path.dirname(output_path_str), exist_ok=True)
                
                # 4. Load Image
                try:
                    with Image.open(img_path_str) as img:
                        W, H = img.size
                except FileNotFoundError:
                    print(f"[Error] Image not found: {img_path_str}")
                    continue

                # 5. Generate and Save
                try:
                    rays_d, rays_m, rays_o = get_plucker_rays(H, W, K, c2w)
                    save_plucker_visualization(output_path_str, rays_d, rays_m, rays_o, num_vis=30, scale=2.0)
                    print(f"[Saved] {output_path_str}")
                except Exception as e:
                    print(f"[Error] Failed to process {img_path_str}: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # List of JSON files to process
    target_files = [
        "train/obj_spatial_relation_oo_mv_both_train.json",
        "train/obj_spatial_relation_oo_mv_plucker_train.json",
        "test/obj_spatial_relation_oo_mv_both_test.json",
        "test/obj_spatial_relation_oo_mv_plucker_test.json"
    ]
    
    # Ensure dependencies are installed:
    # pip install numpy matplotlib pillow scipy
    
    process_dataset_files(target_files)