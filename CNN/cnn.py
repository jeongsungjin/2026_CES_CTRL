# simulation_data.py
import numpy as np
import random
import os

# === 시뮬레이션 설정 ===
IMAGE_SIZE = 64
SEQ_LEN = 100
NUM_EGO = 8
NUM_OTHERS = 3

CAR_HEIGHT = 3
CAR_WIDTH = 4
TURN_PROB = 0.1

LANE_WIDTH = 8
NUM_LANES_ONE_WAY = 4
NUM_TOTAL_LANES = NUM_LANES_ONE_WAY * 2

ROAD_START = (IMAGE_SIZE - LANE_WIDTH * NUM_TOTAL_LANES) // 2
LANE_CENTERS = [ROAD_START + LANE_WIDTH * i + LANE_WIDTH // 2 for i in range(NUM_TOTAL_LANES)]

def turn_direction(dir, turn):
    if turn == 'left':
        return [-dir[1], dir[0]]
    elif turn == 'right':
        return [dir[1], -dir[0]]
    return dir

def draw_car(img, y, x, h, w, color):
    if 0 <= y < IMAGE_SIZE and 0 <= x < IMAGE_SIZE:
        y2 = min(y + h, IMAGE_SIZE)
        x2 = min(x + w, IMAGE_SIZE)
        img[y:y2, x:x2] = color

def check_collision(y1, x1, h1, w1, y2, x2, h2, w2):
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

def create_car(car_type, cars):
    for _ in range(100):
        dir = random.choice([[0,1], [1,0], [0,-1], [-1,0]])
        if dir == [0,1]:
            y = random.choice(LANE_CENTERS[:NUM_LANES_ONE_WAY])
            x = random.randint(0, IMAGE_SIZE//2)
        elif dir == [0,-1]:
            y = random.choice(LANE_CENTERS[NUM_LANES_ONE_WAY:])
            x = random.randint(IMAGE_SIZE//2, IMAGE_SIZE - CAR_WIDTH)
        elif dir == [1,0]:
            x = random.choice(LANE_CENTERS[:NUM_LANES_ONE_WAY])
            y = random.randint(0, IMAGE_SIZE//2)
        else:
            x = random.choice(LANE_CENTERS[NUM_LANES_ONE_WAY:])
            y = random.randint(IMAGE_SIZE//2, IMAGE_SIZE - CAR_HEIGHT)

        h, w = (CAR_WIDTH, CAR_HEIGHT) if dir[0] != 0 else (CAR_HEIGHT, CAR_WIDTH)

        valid = True
        for c in cars:
            cy, cx = c['pos']
            ch, cw = c['size']
            if check_collision(y, x, h, w, cy, cx, ch, cw):
                valid = False
                break
        if valid:
            return {'type': car_type, 'pos': [y, x], 'dir': dir, 'size': [h, w], 'turning': False}
    return None

def generate_sequence(seq_len=SEQ_LEN, num_ego=NUM_EGO, num_others=NUM_OTHERS):
    cars = []
    for _ in range(num_ego):
        car = create_car('ego', cars)
        if car: cars.append(car)
    for _ in range(num_others):
        car = create_car('other', cars)
        if car: cars.append(car)

    frames = []
    for _ in range(seq_len):
        img = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        new_positions = []

        for car in cars:
            y, x = car['pos']
            dy, dx = car['dir']

            if not car['turning'] and random.random() < TURN_PROB:
                car['dir'] = turn_direction(car['dir'], random.choice(['left', 'right']))
                car['turning'] = True
                continue

            dy, dx = car['dir']
            h, w = (CAR_WIDTH, CAR_HEIGHT) if dy != 0 else (CAR_HEIGHT, CAR_WIDTH)
            new_y, new_x = y + dy, x + dx

            if not (0 <= new_y <= IMAGE_SIZE - h and 0 <= new_x <= IMAGE_SIZE - w):
                new_y, new_x = y, x

            new_positions.append((car, [new_y, new_x], [h, w]))

        for i, (car_i, pos_i, size_i) in enumerate(new_positions):
            y1, x1 = pos_i
            h1, w1 = size_i
            collision = False
            for j, (car_j, pos_j, size_j) in enumerate(new_positions):
                if i == j: continue
                y2, x2 = pos_j
                h2, w2 = size_j
                if check_collision(y1, x1, h1, w1, y2, x2, h2, w2):
                    collision = True
                    break

            if not collision:
                car_i['pos'] = pos_i
                car_i['size'] = size_i
                car_i['turning'] = False
            else:
                if car_i['type'] == 'ego':
                    for turn_dir in ['left', 'right']:
                        new_dir = turn_direction(car_i['dir'], turn_dir)
                        dy, dx = new_dir
                        h, w = (CAR_WIDTH, CAR_HEIGHT) if dy != 0 else (CAR_HEIGHT, CAR_WIDTH)
                        new_y, new_x = car_i['pos'][0] + dy, car_i['pos'][1] + dx

                        if not (0 <= new_y <= IMAGE_SIZE - h and 0 <= new_x <= IMAGE_SIZE - w):
                            continue

                        rotated_collision = False
                        for j2, (car_j2, pos_j2, size_j2) in enumerate(new_positions):
                            if i == j2: continue
                            y2, x2 = pos_j2
                            h2, w2 = size_j2
                            if check_collision(new_y, new_x, h, w, y2, x2, h2, w2):
                                rotated_collision = True
                                break
                        if not rotated_collision:
                            car_i['dir'] = new_dir
                            car_i['pos'] = [new_y, new_x]
                            car_i['size'] = [h, w]
                            car_i['turning'] = True
                            break

        for car in cars:
            y, x = car['pos']
            h, w = car['size']
            color = 0.0 if car['type'] == 'ego' else 0.5
            draw_car(img, y, x, h, w, color)

        frames.append(img)

    return np.array(frames, dtype=np.float32)

def generate_dataset(num_sequences=1):
    return np.array([generate_sequence() for _ in range(num_sequences)], dtype=np.float32)

# === 실행 및 저장 (수정된 부분) ===
if __name__ == "__main__":
    output_dir = r"C:\Users\user\Desktop\2025CES\sequences"  # 저장 폴더 경로
    num_sequences = 1000  # 생성할 시퀀스 수
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_sequences):
        sequence = generate_sequence()
        filename = os.path.join(output_dir, f"seq_{i:04d}.npy")
        np.save(filename, sequence)

        if i % 100 == 0:
            print(f"✅ 저장됨: {filename} (진행률: {i}/{num_sequences})")

    print(f"✅ 전체 저장 완료: {output_dir}")

