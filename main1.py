import cv2
import numpy as np
import time
import os

# --- 非線形分離型シグモイドフィルタ ---
def sigmoid_separable_blur(img_gray: np.ndarray, a: float = 0.5, N: int = 15) -> np.ndarray:
    # カーネル事前計算
    i = np.arange(-N, N+1, dtype=np.float32)
    S = 1 / (1 + np.exp(-a * i))
    K = S * (1 - S)          # 導関数形状
    K /= np.sum(K)           # 正規化
    # sepFilter2D で横・縦方向に適用
    blurred = cv2.sepFilter2D(img_gray, -1, K, K)
    return blurred

# --- Retinex 関連関数（非線形シグモイド版） ---
def fastretinex_sigmoid(rgb: np.ndarray, a: float = 0.5, N: int = 15, eps: float = 1e-6,
                        clip_scale=(0.5, 1.5), gamma: float = 0.8,
                        bright_thresh: float = 0.8, bright_k: float = 0.08):
    """
    修正版 Fast Retinex (シグモイド分離ブラーを使用)。
    bright_thresh: ハイライト閾値 t (0..1)。これより明るい領域は補正を小さくする。
    bright_k: 閾値のスムーズネス κ。小さいほど急峻に切り替わる。
    """
    t0 = time.perf_counter()
    img = rgb.astype(np.float32) / 255.0
    # 輝度（Y）
    Y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    t1 = time.perf_counter()
    # 照明推定（分離シグモイドブラー）
    L_sep = sigmoid_separable_blur(Y, a=a, N=N)
    t2 = time.perf_counter()

    # 安全化
    L = np.clip(L_sep, eps, 1.0)

    # 反射成分 R = Y / L
    R = Y / L

    # 正しいスケール（Rをそのまま or クリップ）
    s = np.clip(R, clip_scale[0], clip_scale[1])  # s(x,y)

    # ハイライト保護用ウェイト w(L) = 1 / (1 + exp((L - t)/k))
    # 明るい領域(L>t) で w -> 0, 暗い領域で w -> 1
    w = 1.0 / (1.0 + np.exp((L - bright_thresh) / (bright_k + 1e-12)))

    # 最終スケール: 1 + w * (s - 1)
    S_final = 1.0 + w * (s - 1.0)
    # チャネルに合わせて展開
    out = img * S_final[:, :, None]
    out = np.clip(out, 0.0, 1.0)

    # ガンマ補正
    out = np.power(out, gamma)
    out_uint8 = (out * 255.0).astype(np.uint8)

    t3 = time.perf_counter()
    timings = {"sep_time": t2 - t1, "full_time": t3 - t2, "total_time": t3 - t0}

    return out_uint8, L_sep.astype(np.float32), timings

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))

# --- メイン ---
def run_camera_benchmark_with_hands():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return

    a = 0.5    # シグモイド勾配
    N = 15     # カーネル半径
    frame_count = 0
    os.makedirs("screenshots", exist_ok=True)

    t_prev = time.perf_counter()
    fps_list, sep_time_list, full_time_list, mse_list = [], [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 補正前（BGR）を保持
        frame_before = frame.copy()

        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out, L_sep, timings = fastretinex_sigmoid(rgb, a=a, N=N)

        # 補正後（RGB → BGR）
        frame_after = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        # MSE（元画像のRチャネルと補正後グレースケール）
        gray_out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        frame_mse = mse(rgb[:, :, 0], gray_out)

        # FPS計算
        t_now = time.perf_counter()
        fps = 1.0 / (t_now - t_prev) if (t_now - t_prev) > 0 else 0.0
        t_prev = t_now

        fps_list.append(fps)
        sep_time_list.append(timings['sep_time'])
        full_time_list.append(timings['full_time'])
        mse_list.append(frame_mse)

        frame_count += 1

        # Before / After を横並びで表示
        disp = np.hstack((frame_before, frame_after))
        cv2.imshow("FastRetinex Sigmoid (Before | After)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            filename = f"screenshots/frame_{frame_count:04d}.png"
            cv2.imwrite(filename, disp)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

    # 結果表示
    avg_fps = np.mean(fps_list) if fps_list else 0
    avg_sep = np.mean(sep_time_list) if sep_time_list else 0
    avg_full = np.mean(full_time_list) if full_time_list else 0
    avg_mse = np.mean(mse_list) if mse_list else 0
    print("\n--- CAMERA SESSION SUMMARY ---")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average sep time: {avg_sep*1000:.2f} ms")
    print(f"Average full time: {avg_full*1000:.2f} ms")
    print(f"Average MSE(L): {avg_mse:.6f}")
    print(f"Total frames processed: {frame_count}")

if __name__ == "__main__":
    run_camera_benchmark_with_hands()
