import numpy as np
import matplotlib.pyplot as plt


def generate_bw_blobs_sequence_no_overlap(num_frames=300, grid_size=64, num_objects=6, object_radius=2):
    sequence = []
    margin = object_radius + 1

    # 초기 위치 설정 (겹치지 않게)
    positions = []
    while len(positions) < num_objects:
        candidate = np.array([np.random.randint(margin, grid_size - margin),
                              np.random.randint(margin, grid_size - margin)])
        if all(np.linalg.norm(candidate - p) >= 2 * object_radius + 1 for p in positions):
            positions.append(candidate)
    positions = np.array(positions)

    # 초기 방향과 색상
    directions = np.random.randint(-1, 2, size=(num_objects, 2))
    colors = np.array([0.0]*3 + [1.0]*3)
    np.random.shuffle(colors)  # 섞어서 위치 무작위

    for frame_idx in range(num_frames):
        grid = np.ones((grid_size, grid_size)) * 0.5  # 회색 배경

        # 그리기
        for i, (x, y) in enumerate(positions):
            for dx in range(-object_radius, object_radius + 1):
                for dy in range(-object_radius, object_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        grid[nx, ny] = colors[i]

        noise = np.random.normal(loc=0, scale=0.05, size=grid.shape)
        grid += noise
        grid = np.clip(grid, 0, 1)
        
        # 위치 업데이트
        new_positions = positions.copy()
        for i in range(num_objects):
            candidate = positions[i] + directions[i]
            candidate = np.clip(candidate, margin, grid_size - margin)

            # 다른 블록들과 거리 체크
            conflict = False
            for j in range(num_objects):
                if i != j:
                    if np.linalg.norm(candidate - positions[j]) < 2 * object_radius + 1:
                        conflict = True
                        break

            if not conflict:
                new_positions[i] = candidate
            else:
                # 방향 변경
                directions[i] = np.random.randint(-1, 2, size=2)

            # 방향 조정 확률
            if np.random.rand() < 0.3:
                directions[i] += np.random.randint(-1, 2, size=2)
                directions[i] = np.clip(directions[i], -1, 1)

        positions = new_positions
        sequence.append(grid.copy())

    return np.array(sequence)

def plot_sequence(sequence, interval=50):
    plt.figure(figsize=(5, 5))
    for i, frame in enumerate(sequence):
        plt.imshow(frame, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Frame {i}")
        plt.axis('off')
        plt.pause(interval / 1000)
        plt.clf()
    plt.close()

if __name__ == "__main__":
    seq = generate_bw_blobs_sequence_no_overlap()
    plot_sequence(seq)
    np.save("bw_grid_sequence_300_no_overlap.npy", seq)
