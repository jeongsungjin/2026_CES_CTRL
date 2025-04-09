import numpy as np
import matplotlib.pyplot as plt
import os

def draw_grid(ax, size=64, interval=8, color='gray'):
    for x in range(0, size, interval):
        ax.axhline(x, color=color, linewidth=0.3)
        ax.axvline(x, color=color, linewidth=0.3)

def convert_to_rgb(frame, threshold=0.5):
    """3채널 Occupancy Grid → RGB 시각화"""
    bg = frame[0]
    road = frame[1]
    vehicle = frame[2]
    h, w = bg.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[(road > threshold)] = [128, 128, 128]     # 도로: 회색
    rgb[(vehicle > threshold)] = [255, 0, 0]       # 차량: 빨강
    return rgb

def visualize_gt_vs_prediction(gt_path, pred_path, interval=0.2):
    if not os.path.exists(gt_path):
        print(f"❌ GT 파일 없음: {gt_path}")
        return
    if not os.path.exists(pred_path):
        print(f"❌ 예측 파일 없음: {pred_path}")
        return

    gt_data = np.load(gt_path)     # (T, 3, H, W)
    pred_data = np.load(pred_path) # (T, 3, H, W)
    assert gt_data.shape == pred_data.shape, "GT와 예측의 shape이 다릅니다."

    T = gt_data.shape[0]

    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # 첫 프레임 초기화
    gt_img = axs[0].imshow(convert_to_rgb(gt_data[0]))
    pred_img = axs[1].imshow(convert_to_rgb(pred_data[0]))

    axs[0].set_title("Ground Truth")
    axs[1].set_title("Prediction")

    draw_grid(axs[0], size=gt_data.shape[2])
    draw_grid(axs[1], size=gt_data.shape[2])

    for t in range(T):
        gt_rgb = convert_to_rgb(gt_data[t])
        pred_rgb = convert_to_rgb(pred_data[t])
        gt_img.set_data(gt_rgb)
        pred_img.set_data(pred_rgb)
        axs[0].set_title(f"GT Frame {t}")
        axs[1].set_title(f"Pred Frame {t}")
        plt.draw()
        plt.pause(interval)

    plt.ioff()
    plt.show()

# ==========================
# 실행 예시
# ==========================
if __name__ == "__main__":
    epoch = 30  # 보고 싶은 에폭
    pred_path = f"predictions/pred_epoch_{epoch}.npy"
    gt_dataset = np.load("/home/ctrl1/2026_CES_CTRL/seq_000.npy")  # raw GT
    from pathlib import Path

    # GT split
    def split_channels(frames):
        bg = (frames == 0).astype(np.float32)
        road = (frames == 50).astype(np.float32)
        vehicle = ((frames > 0) & (frames != 50)).astype(np.float32)
        return np.stack([bg, road, vehicle], axis=1)

    # 같은 프레임 수만큼 슬라이싱 (기준: input=10, pred=5)
    gt_seq = split_channels(gt_dataset[30:35])  # 예: index 10~14
    gt_temp_path = "temp_gt.npy"
    np.save(gt_temp_path, gt_seq)

    visualize_gt_vs_prediction(gt_temp_path, pred_path, interval=1.2)

    # 임시 GT 파일 삭제하고 싶으면:
    # os.remove(gt_temp_path)
