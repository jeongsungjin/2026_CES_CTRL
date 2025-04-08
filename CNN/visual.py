# visualize_sequence.py
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_sequence(frames, pause_time=0.1):
    fig, ax = plt.subplots()
    img_plot = ax.imshow(frames[0], cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    title = ax.set_title("Frame 0")

    for i, frame in enumerate(frames):
        img_plot.set_data(frame)
        title.set_text(f"Frame {i}")
        plt.pause(pause_time)

    plt.show()

if __name__ == "__main__":
    folder_path = r"C:\Users\user\Desktop\2025CES\sequences"  # 시퀀스 폴더
    file_index = 49  # 몇 번째 시퀀스를 볼지 선택 (0부터 시작)

    # 파일 리스트 불러오기
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    
    if not npy_files:
        print(f"❌ .npy 파일을 찾을 수 없습니다: {folder_path}")
    else:
        selected_file = npy_files[file_index]
        input_path = os.path.join(folder_path, selected_file)

        dataset = np.load(input_path)
        print(f"✅ 시퀀스 로드 완료: {selected_file}")
        print(f"✅ shape: {dataset.shape}")  # (100, 64, 64)

        visualize_sequence(dataset)
