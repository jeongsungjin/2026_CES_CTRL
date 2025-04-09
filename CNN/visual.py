import numpy as np
import matplotlib.pyplot as plt
import time
import os

def visualize_sequence(npy_path, interval=0.1):
    if not os.path.exists(npy_path):
        print(f"파일이 존재하지 않습니다: {npy_path}")
        return

    sequence = np.load(npy_path)  # shape: (num_frames, 64, 64)
    num_frames = sequence.shape[0]

    plt.ion()  # interactive 모드
    fig, ax = plt.subplots()

    # 컬러맵 변경: 차량 ID가 서로 다른 색으로 표시되도록
    img = ax.imshow(sequence[0], cmap='tab20', vmin=0, vmax=np.max(sequence))
    ax.set_title("Frame 0")

    for i in range(num_frames):
        img.set_data(sequence[i])
        ax.set_title(f"Frame {i}")
        plt.draw()
        plt.pause(interval)  # interval 초 대기 (예: 0.1초 = 100ms)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    # npy_file = "/home/ctrl1/2026_CES_CTRL/sequence_data/sequence.npy"
    npy_file = 'C://Users//user//Desktop//2025CES//sequence_data.npy'
    visualize_sequence(npy_file)
