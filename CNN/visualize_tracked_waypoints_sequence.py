
import numpy as np
import os
import matplotlib.pyplot as plt

def visualize_sequence_waypoints(sequence, waypoints_dict, seq_id, start_frame=90, num_frames=10):
    frames = sequence[start_frame:start_frame + num_frames]
    waypoints_dict = {int(k): v for k, v in waypoints_dict.items()}

    for i, frame in enumerate(frames):
        frame_id = start_frame + i
        plt.figure(figsize=(6,6))
        plt.title(f"Sequence {seq_id} - Frame {frame_id}")
        plt.imshow(frame, cmap='gray')

        for vid, traj in waypoints_dict.items():
            if len(traj) > frame_id:
                point = traj[frame_id]
                if point[0] is not None:
                    plt.plot(point[1], point[0], 'o', label=f"ID {vid}")
        plt.legend()
        plt.grid(True)
        plt.show()

def run(seq_path, gt_path, seq_id=0, start_frame=90, num_frames=10):
    sequence = np.load(seq_path)
    waypoints_dict = np.load(gt_path, allow_pickle=True).item()
    visualize_sequence_waypoints(sequence, waypoints_dict, seq_id, start_frame, num_frames)

if __name__ == "__main__":
    seq_path = r"C:\Users\user\Desktop\2025CES\sequences\seq_0000.npy"
    gt_path  = r"C:\Users\user\Desktop\2025CES\gt_tracked\seq_0000_tracked_gt.npy"
    run(seq_path, gt_path, seq_id=0, start_frame=90, num_frames=10)
