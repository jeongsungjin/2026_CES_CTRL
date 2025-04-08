
import numpy as np
import os
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def track_vehicles(sequence, max_distance=5.0):
    num_frames, H, W = sequence.shape
    next_id = 1
    tracks = {}
    active_ids = {}

    for t in range(num_frames):
        frame = sequence[t]
        binary = (frame == 0.0).astype(np.uint8)
        labeled, num_features = label(binary)
        centers = center_of_mass(binary, labeled, range(1, num_features + 1))

        frame_ids = {}

        if t == 0:
            for c in centers:
                tracks[next_id] = [c]
                frame_ids[next_id] = c
                next_id += 1
        else:
            prev_centers = np.array([active_ids[vid] for vid in active_ids])
            prev_ids = list(active_ids.keys())
            matched = set()

            if len(prev_centers) > 0 and len(centers) > 0:
                dists = cdist(prev_centers, centers)
                for i, row in enumerate(dists):
                    j = np.argmin(row)
                    if row[j] < max_distance and j not in matched:
                        vid = prev_ids[i]
                        tracks[vid].append(centers[j])
                        frame_ids[vid] = centers[j]
                        matched.add(j)
                    else:
                        tracks[prev_ids[i]].append((None, None))

            unmatched = set(range(len(centers))) - matched
            for j in unmatched:
                tracks[next_id] = [(None, None)] * t + [centers[j]]
                frame_ids[next_id] = centers[j]
                next_id += 1

            for vid in active_ids:
                if vid not in frame_ids:
                    tracks[vid].append((None, None))

        active_ids = frame_ids

    return tracks

def extract_recent_waypoints(gt_dict, num_waypoints=5):
    waypoints = []
    for vid, track in gt_dict.items():
        valid = [pt for pt in track if pt[0] is not None]
        if len(valid) >= num_waypoints:
            waypoints.append(valid[-num_waypoints:])
    return np.array(waypoints, dtype=np.float32)

def process_all(sequence_dir, gt_out_dir, wp_out_dir, visualize=False, num_waypoints=5):
    os.makedirs(gt_out_dir, exist_ok=True)
    os.makedirs(wp_out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(sequence_dir) if f.endswith(".npy") and not f.endswith("_gt.npy")])

    for f in files:
        seq_path = os.path.join(sequence_dir, f)
        sequence = np.load(seq_path)
        seq_id = int(f.split("_")[1].split(".")[0])

        # 트래킹 기반 GT 생성
        tracked_gt = track_vehicles(sequence)
        gt_path = os.path.join(gt_out_dir, f.replace(".npy", "_tracked_gt.npy"))
        np.save(gt_path, tracked_gt)

        # 웨이포인트 생성
        waypoints = extract_recent_waypoints(tracked_gt, num_waypoints)
        if len(waypoints) == 0:
            print(f"⚠️ {f}: 유효 웨이포인트 없음 → 건너뜀")
            continue

        wp_path = os.path.join(wp_out_dir, f.replace(".npy", "_tracked_waypoints.npy"))
        np.save(wp_path, waypoints)
        print(f"✅ 저장: {gt_path}, {wp_path}")

        if visualize:
            frame = sequence[-1]
            plt.imshow(frame, cmap='gray')
            for traj in waypoints:
                traj = np.array(traj)
                plt.plot(traj[:, 1], traj[:, 0], 'o-')
            plt.title(f"Sequence {seq_id} - Tracked Waypoints")
            plt.show()

if __name__ == "__main__":
    sequence_dir = r"C:\Users\user\Desktop\2025CES\sequences"
    gt_out_dir = r"C:\Users\user\Desktop\2025CES\gt_tracked"
    wp_out_dir = r"C:\Users\user\Desktop\2025CES\waypoints_tracked"
    process_all(sequence_dir, gt_out_dir, wp_out_dir, visualize=False)
