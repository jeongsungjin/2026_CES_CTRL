import numpy as np
import random
import os

# ===================== 시뮬레이션 전역 설정 =====================
MAP_SIZE = 64
BACKGROUND_VALUE = 0
ROAD_VALUE = 50

ROAD_THICKNESS = 4
NUM_FRAMES = 3000  # ✅ 프레임 수 변경
NUM_SEQUENCES = 100  # ✅ 시퀀스 수 추가
SAVE_DIR = 'C://home//ctrl1//2026_CES_CTRL//sequence_data'

VEHICLE_SIZE_LONG = 3
VEHICLE_SIZE_SHORT = 2

INTERSECTIONS = [(16, 16), (16, 48), (48, 16), (48, 48), (32, 32)]
INTERSECTION_AREA = set()

def create_map():
    global INTERSECTION_AREA
    INTERSECTION_AREA.clear()

    game_map = np.full((MAP_SIZE, MAP_SIZE), BACKGROUND_VALUE, dtype=np.uint8)
    half_t = ROAD_THICKNESS // 2
    half_t_ceil = (ROAD_THICKNESS + 1) // 2

    for (r, c) in INTERSECTIONS:
        game_map[r - half_t : r + half_t_ceil, :] = ROAD_VALUE
        game_map[:, c - half_t : c + half_t_ceil] = ROAD_VALUE

        for rr in range(r - half_t, r + half_t_ceil):
            for cc in range(c - half_t, c + half_t_ceil):
                INTERSECTION_AREA.add((rr, cc))

    return game_map

def in_bounds(r, c):
    return 0 <= r < MAP_SIZE and 0 <= c < MAP_SIZE

def get_vehicle_pixels(r, c, height, width):
    coords = []
    for i in range(height):
        for j in range(width):
            rr = r + i
            cc = c + j
            if in_bounds(rr, cc):
                coords.append((rr, cc))
    return coords

def is_road(pixels, road_map):
    for (rr, cc) in pixels:
        if road_map[rr, cc] != ROAD_VALUE:
            return False
    return True

class Vehicle:
    def __init__(self, r, c, direction, v_type, vehicle_id):
        self.r = r
        self.c = c
        self.direction = direction
        self.v_type = v_type
        self.vehicle_id = vehicle_id

        self.set_size_by_direction()
        self.stuck = 0
        self.snap_to_lane()

    def set_size_by_direction(self):
        if self.direction in ['up', 'down']:
            self.height = VEHICLE_SIZE_LONG
            self.width = VEHICLE_SIZE_SHORT
        else:
            self.height = VEHICLE_SIZE_SHORT
            self.width = VEHICLE_SIZE_LONG

    def get_pixels(self):
        return get_vehicle_pixels(self.r, self.c, self.height, self.width)

    def center_position(self):
        return (self.r + self.height // 2, self.c + self.width // 2)

    def set_position_centered(self, center_r, center_c):
        self.r = center_r - self.height // 2
        self.c = center_c - self.width // 2

    def snap_to_lane(self):
        center_r, center_c = self.center_position()
        rows = [ic[0] for ic in INTERSECTIONS]
        cols = [ic[1] for ic in INTERSECTIONS]
        near_r = min(rows, key=lambda x: abs(x - center_r))
        near_c = min(cols, key=lambda x: abs(x - center_c))

        if self.direction == 'up':
            self.set_position_centered(center_r, near_c - 1)
        elif self.direction == 'down':
            self.set_position_centered(center_r, near_c + 1)
        elif self.direction == 'left':
            self.set_position_centered(near_r - 1, center_c)
        else:
            self.set_position_centered(near_r + 1, center_c)

    def is_valid(self, road_map, others):
        my_pixels = self.get_pixels()
        if not my_pixels or not is_road(my_pixels, road_map):
            return False
        s = set(my_pixels)
        for o in others:
            if s & set(o.get_pixels()):
                return False
        return True

    def at_intersection(self):
        return any(px in INTERSECTION_AREA for px in self.get_pixels())

    def move(self, road_map, others):
        for _ in range(2 if self.v_type == 'gray' else 1):
            self._move_once(road_map, others)

    def _move_once(self, road_map, others):
        vec = {
            'up': (-1, 0), 'down': (1, 0),
            'left': (0, -1), 'right': (0, 1)
        }
        turn_left = {
            'up': 'left', 'left': 'down',
            'down': 'right', 'right': 'up'
        }
        turn_right = {
            'up': 'right', 'right': 'down',
            'down': 'left', 'left': 'up'
        }

        old_center = self.center_position()
        self.snap_to_lane()

        dx, dy = vec[self.direction]

        if self.v_type == 'white':
            for step in range(1, 3):
                check_r = old_center[0] + dx * step
                check_c = old_center[1] + dy * step
                check_pixels = get_vehicle_pixels(check_r - self.height // 2, check_c - self.width // 2, self.height, self.width)
                for o in others:
                    if set(check_pixels) & set(o.get_pixels()):
                        self.stuck += 1
                        return

        new_center = (self.center_position()[0] + dx, self.center_position()[1] + dy)
        self.set_position_centered(*new_center)

        if self.is_valid(road_map, others):
            self.stuck = 0
            return
        else:
            self.set_position_centered(*old_center)

        if self.at_intersection():
            new_dir = turn_left[self.direction] if random.random() < 0.5 else turn_right[self.direction]
            old_dir = self.direction
            self.direction = new_dir
            self.set_size_by_direction()

            old_center2 = self.center_position()
            self.snap_to_lane()
            dx2, dy2 = vec[self.direction]
            new_center2 = (self.center_position()[0] + dx2, self.center_position()[1] + dy2)
            self.set_position_centered(*new_center2)

            if self.is_valid(road_map, others):
                self.stuck = 0
                return
            else:
                self.set_position_centered(*old_center2)
                self.direction = old_dir
                self.set_size_by_direction()
                self.set_position_centered(*old_center)

        if self.v_type in ('gray', 'white'):
            self.stuck += 1
            old_dir = self.direction
            old_cpos = self.center_position()

            rows = [ic[0] for ic in INTERSECTIONS]
            cols = [ic[1] for ic in INTERSECTIONS]
            near_r = min(rows, key=lambda x: abs(x - old_cpos[0]))
            near_c = min(cols, key=lambda x: abs(x - old_cpos[1]))

            if self.direction == 'up':
                self.set_position_centered(old_cpos[0], near_c + 1)
            elif self.direction == 'down':
                self.set_position_centered(old_cpos[0], near_c - 1)
            elif self.direction == 'left':
                self.set_position_centered(near_r - 1, old_cpos[1])
            else:
                self.set_position_centered(near_r + 1, old_cpos[1])

            dx2, dy2 = vec[self.direction]
            new_c2 = (self.center_position()[0] + dx2, self.center_position()[1] + dy2)
            self.set_position_centered(*new_c2)

            if self.is_valid(road_map, others):
                self.stuck = 0
                return
            else:
                self.set_position_centered(*old_cpos)
                self.direction = old_dir
                self.set_size_by_direction()
                self.set_position_centered(*old_center)

            if self.stuck > 3:
                reverse_dir = {
                    'up': 'down', 'down': 'up',
                    'left': 'right', 'right': 'left'
                }
                old_dir = self.direction
                self.direction = reverse_dir[old_dir]
                self.set_size_by_direction()

                old_center2 = self.center_position()
                self.snap_to_lane()
                dx2, dy2 = vec[self.direction]
                new_c2 = (self.center_position()[0] + dx2, self.center_position()[1] + dy2)
                self.set_position_centered(*new_c2)

                if self.is_valid(road_map, others):
                    self.stuck = 0
                    return
                else:
                    self.set_position_centered(*old_center2)
                    self.direction = old_dir
                    self.set_size_by_direction()
                    self.set_position_centered(*old_center)

def run_simulation(seq_idx):
    road_map = create_map()

    WHITE_VEHICLES = 5
    GRAY_VEHICLES = 10
    NUM_VEHICLES = WHITE_VEHICLES + GRAY_VEHICLES
    DIRECTIONS = ['up', 'down', 'left', 'right']

    vehicle_targets = (['white'] * WHITE_VEHICLES) + (['gray'] * GRAY_VEHICLES)
    random.shuffle(vehicle_targets)

    vehicles = []
    tries = 0
    max_tries = NUM_VEHICLES * 10
    next_id = 1

    while vehicle_targets and tries < max_tries:
        direction = random.choice(DIRECTIONS)
        v_type = vehicle_targets[0]
        r = random.randint(0, MAP_SIZE - VEHICLE_SIZE_LONG)
        c = random.randint(0, MAP_SIZE - VEHICLE_SIZE_LONG)
        temp_vehicle = Vehicle(r, c, direction, v_type, next_id)

        if temp_vehicle.is_valid(road_map, vehicles):
            vehicles.append(temp_vehicle)
            vehicle_targets.pop(0)
            next_id += 1

        tries += 1

    frames = []
    for _ in range(NUM_FRAMES):
        frame = road_map.copy()
        for v in vehicles:
            others = [o for o in vehicles if o != v]
            v.move(road_map, others)
        for v in vehicles:
            for (rr, cc) in v.get_pixels():
                frame[rr, cc] = v.vehicle_id
        frames.append(frame)

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_file = os.path.join(SAVE_DIR, f"seq_{seq_idx:03d}.npy")
    np.save(save_file, np.array(frames, dtype=np.uint8))
    print(f"✅ 시퀀스 {seq_idx+1}/{NUM_SEQUENCES} 저장 완료: {save_file}")

if __name__ == "__main__":
    for seq_idx in range(NUM_SEQUENCES):
        run_simulation(seq_idx)
