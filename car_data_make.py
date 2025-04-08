import numpy as np
import matplotlib.pyplot as plt
import random

import numpy as np
import random


def get_intersection_mask(road_mask):
    """도로 마스크에서 교차로가 될 수 있는 영역만 추출"""
    intersection_mask = np.zeros_like(road_mask)

    for x in [32, 64, 96]:
        for y in [32, 64, 96]:
            # 교차로 영역을 중심 (x, y) 기준으로 정사각형 범위로 지정
            intersection_mask[x-3:x+4, y-3:y+4] = 1.0  # 크기는 조절 가능

    return intersection_mask


def is_near_intersection(pos, intersection_mask):
    x, y = int(pos[0]), int(pos[1])
    if 0 <= x < intersection_mask.shape[0] and 0 <= y < intersection_mask.shape[1]:
        return intersection_mask[x, y] == 1.0
    return False


def generate_vehicle_sequence_with_black_white_blocks(
    num_frames=3000,
    grid_size=128,
    vehicle_width=3,
    vehicle_height=5
):
    sequence = []
    grid = np.ones((grid_size, grid_size)) * 0.2  # 배경

    # 도로 생성
    road_mask = np.zeros((grid_size, grid_size))
    for x in [32, 64, 96]:
        road_mask[x-3:x+4, :] = 1.0
    for y in [32, 64, 96]:
        road_mask[:, y-3:y+4] = 1.0
    road_positions = np.argwhere(road_mask == 1.0)

    intersection_mask = get_intersection_mask(road_mask)

    # 차량 6대 (3 검정, 3 흰색)
    num_objects = 6
    vehicle_colors = [0.0]*3 + [1.0]*3
    random.shuffle(vehicle_colors)

    directions_dict = {
        "up": np.array([0, -1]),
        "down": np.array([0, 1]),
        "left": np.array([-1, 0]),
        "right": np.array([1, 0]),
    }
    direction_keys = list(directions_dict.keys())

    positions = []
    directions = []

    while len(positions) < num_objects:
        candidate = road_positions[np.random.choice(len(road_positions))]
        if all(np.linalg.norm(candidate - p) >= max(vehicle_width, vehicle_height) * 2 for p in positions):
            positions.append(candidate)
            directions.append(directions_dict[random.choice(direction_keys)])

    positions = np.array(positions)
    directions = np.array(directions)
    stuck_counter = np.zeros(num_objects, dtype=int)  # 정체 프레임 수

    for frame_idx in range(num_frames):
        frame = grid.copy()
        frame[road_mask == 1.0] = 0.5  # 도로는 회색

        for i, (x, y) in enumerate(positions):
            dir_vec = directions[i]
            h, w = (vehicle_height, vehicle_width) if abs(dir_vec[1]) else (vehicle_width, vehicle_height)

            for dx in range(-w//2, w//2 + 1):
                for dy in range(-h//2, h//2 + 1):
                    nx, ny = int(x + dx), int(y + dy)
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        frame[nx, ny] = vehicle_colors[i]

        sequence.append(frame)

        new_positions = positions.copy()
        new_directions = directions.copy()

        for i in range(num_objects):
            current_dir = directions[i]
            candidate_forward = positions[i] + current_dir * 2
            cx, cy = int(candidate_forward[0]), int(candidate_forward[1])

            can_go_forward = (
                0 <= cx < grid_size and 0 <= cy < grid_size and
                road_mask[cx, cy] == 1.0
            )

            moved = False

            # ✅ 교차로에 가까우면 확률적으로 회전 시도 (우선)
            if is_near_intersection(positions[i], intersection_mask) and random.random() < 0.3:
                left_dir = np.array([-current_dir[1], current_dir[0]])
                right_dir = np.array([current_dir[1], -current_dir[0]])
                turn_choices = [left_dir, right_dir]
                random.shuffle(turn_choices)

                for turn_dir in turn_choices:
                    candidate = positions[i] + turn_dir * 2
                    tx, ty = int(candidate[0]), int(candidate[1])
                    if (
                        0 <= tx < grid_size and 0 <= ty < grid_size and
                        road_mask[tx, ty] == 1.0 and
                        all(np.linalg.norm(candidate - p) > max(vehicle_width, vehicle_height) * 0.6
                            for j, p in enumerate(positions) if i != j)
                    ):
                        new_positions[i] = candidate
                        new_directions[i] = turn_dir
                        stuck_counter[i] = 0
                        moved = True
                        break

            # 직진 가능하면 직진
            if not moved and can_go_forward and all(np.linalg.norm(candidate_forward - p) > max(vehicle_width, vehicle_height) * 0.8
                                                    for j, p in enumerate(positions) if i != j):
                new_positions[i] = candidate_forward
                new_directions[i] = current_dir
                stuck_counter[i] = 0
                moved = True

            # 그 외의 경우 기존처럼 좌우 회전 시도
            if not moved:
                left_dir = np.array([-current_dir[1], current_dir[0]])
                right_dir = np.array([current_dir[1], -current_dir[0]])
                turn_choices = [left_dir, right_dir]
                random.shuffle(turn_choices)

                for turn_dir in turn_choices:
                    candidate = positions[i] + turn_dir * 2
                    tx, ty = int(candidate[0]), int(candidate[1])
                    if (
                        0 <= tx < grid_size and 0 <= ty < grid_size and
                        road_mask[tx, ty] == 1.0 and
                        all(np.linalg.norm(candidate - p) > max(vehicle_width, vehicle_height) * 0.6
                            for j, p in enumerate(positions) if i != j)
                    ):
                        new_positions[i] = candidate
                        new_directions[i] = turn_dir
                        stuck_counter[i] = 0
                        moved = True
                        break

            # 여전히 이동 못했으면 stuck 처리
            if not moved:
                stuck_counter[i] += 1

            # 정체 탈출 로직 (그대로 유지)
            if stuck_counter[i] >= 5:
                recovery_dirs = [
                    -current_dir,
                    np.array([-current_dir[1], current_dir[0]]),
                    np.array([current_dir[1], -current_dir[0]])
                ]
                random.shuffle(recovery_dirs)
                for recovery_dir in recovery_dirs:
                    candidate = positions[i] + recovery_dir * 2
                    tx, ty = int(candidate[0]), int(candidate[1])
                    if (
                        0 <= tx < grid_size and 0 <= ty < grid_size and
                        road_mask[tx, ty] == 1.0 and
                        all(np.linalg.norm(candidate - p) > max(vehicle_width, vehicle_height) * 0.8
                            for j, p in enumerate(positions) if i != j)
                    ):
                        new_positions[i] = candidate
                        new_directions[i] = recovery_dir
                        stuck_counter[i] = 0
                        break


        positions = new_positions
        directions = new_directions

    return np.array(sequence)

def plot_sequence(sequence, interval=50):
    plt.figure(figsize=(5, 5))
    for i, frame in enumerate(sequence):
        plt.imshow(frame.T, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Frame {i}")
        plt.axis('off')
        plt.pause(interval / 1000)
        plt.clf()
    plt.close()

if __name__ == "__main__":
    seq = generate_vehicle_sequence_with_black_white_blocks()
    plot_sequence(seq)
    np.save("vehicle_black_white_sequence.npy", seq)
