import cv2
import numpy as np
import math
import glob
import os
import time  # ★ 処理時間計測用

# =========================
#  共通: 指標 MSE / PSNR / SSIM
# =========================
def mse(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))

def psnr(a, b, max_val=255.0):
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / m)

def ssim(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    L = 255.0
    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)

    return float(np.mean(num / (den + 1e-12)))


# =========================
#  XCR
# =========================
def xcr_kernel_1d(a=0.5, N=15):
    u = np.arange(-N, N+1, dtype=np.float64)
    theta = a * u
    X = np.cos(theta) - 1j * np.sin(theta)
    D = 1.0 + 2.0 * X + X**2
    K = np.abs(X / D)
    K = np.clip(K, 0.0, None)
    K /= K.sum()
    return K.astype(np.float32)

def xcr_blur_Y(Y, a=0.5, N=15):
    K = xcr_kernel_1d(a, N)
    return cv2.sepFilter2D(Y, -1, K, K)

def xcr_enhance_bgr(bgr,
                    a=0.5, N=15,
                    eps=1e-6,
                    smin=0.5, smax=1.5,
                    t=0.8, kappa=0.08,
                    gamma=0.8):
    img = bgr.astype(np.float32) / 255.0
    Rc = img[:, :, 2]
    Gc = img[:, :, 1]
    Bc = img[:, :, 0]

    Y = 0.299 * Rc + 0.587 * Gc + 0.114 * Bc

    L = xcr_blur_Y(Y, a, N)
    L = np.clip(L, eps, 1.0)

    logY = np.log(Y + eps)
    logL = np.log(L + eps)
    logR = logY - logL
    RY = np.exp(logR)

    s0 = np.clip(RY / (Y + eps), smin, smax)

    w_L = 1.0 / (1.0 + np.exp((L - t) / (kappa + 1e-12)))

    S = 1.0 + w_L * (s0 - 1.0)

    out = img * S[:, :, None]
    out = np.clip(out, 0.0, 1.0)
    out = np.power(out, gamma)
    out_bgr = (out * 255.0).astype(np.uint8)

    return out_bgr


# =========================
#  MSR (Multi-Scale Retinex)
# =========================
def msr_enhance_bgr(bgr,
                    sigmas=(15, 80, 250),
                    eps=1e-6):
    img = bgr.astype(np.float32) / 255.0
    retinex = np.zeros_like(img, dtype=np.float32)

    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += np.log(img + eps) - np.log(blur + eps)

    retinex /= float(len(sigmas))

    # 0〜1 に正規化
    r_min = retinex.min()
    r_max = retinex.max()
    retinex = (retinex - r_min) / (r_max - r_min + 1e-12)
    out = (retinex * 255.0).astype(np.uint8)
    return out


# =========================
#  LIME 風 (簡易版)
# =========================
def lime_like_enhance_bgr(bgr,
                          ksize=15,
                          sigma=3.0,
                          eps=1e-6,
                          gamma=0.8):
    img = bgr.astype(np.float32) / 255.0

    # 照明マップ T(x,y) = max(R,G,B)
    T = np.max(img, axis=2)

    # 平滑化して T' に（本家はもっと凝った構造保持）
    T_smooth = cv2.GaussianBlur(T, (ksize, ksize), sigma)

    # 増幅
    J = img / (T_smooth[:, :, None] + eps)

    # 0〜1正規化
    J = np.clip(J, 0.0, 1.0)
    J = np.power(J, gamma)
    out = (J * 255.0).astype(np.uint8)
    return out


# =========================
#  データセット評価ループ（処理時間付き）
# =========================
def evaluate_method(low_dir, gt_dir, enhance_fn, name="METHOD"):
    low_paths = sorted(glob.glob(os.path.join(low_dir, "*.*")))
    if not low_paths:
        print(f"{name}: No files in {low_dir}")
        return

    mse_list, psnr_list, ssim_list = [], [], []

    print(f"\n=== Evaluating {name} ===")

    t_start_all = time.perf_counter()  # ★ この手法全体の開始時間

    for low_path in low_paths:
        fname = os.path.basename(low_path)
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            print(f"  [skip] GT not found for {fname}")
            continue

        low_bgr = cv2.imread(low_path, cv2.IMREAD_COLOR)
        gt_bgr = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        if low_bgr is None or gt_bgr is None:
            print(f"  [skip] load error: {fname}")
            continue

        h, w = low_bgr.shape[:2]
        gt_bgr = cv2.resize(gt_bgr, (w, h), interpolation=cv2.INTER_AREA)

        # ---- 強調（ここが計測対象のメイン処理） ----
        enh_bgr = enhance_fn(low_bgr)
        # ------------------------------------------

        # 指標（ここではグレースケール）
        gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2GRAY)

        m = mse(gt_gray, enh_gray)
        p = psnr(gt_gray, enh_gray)
        s = ssim(gt_gray, enh_gray)

        mse_list.append(m)
        psnr_list.append(p)
        ssim_list.append(s)

        print(f"  {fname}: PSNR={p:.2f} dB, SSIM={s:.4f}")

    t_end_all = time.perf_counter()
    total_time = t_end_all - t_start_all  # ★ 合計処理時間

    if mse_list:
        n = len(mse_list)
        print(f"\n--- {name} SUMMARY ---")
        print(f"  N         : {n}")
        print(f"  MSE       : {np.mean(mse_list):.2f}")
        print(f"  PSNR      : {np.mean(psnr_list):.2f} dB")
        print(f"  SSIM      : {np.mean(ssim_list):.4f}")
        print(f"  TotalTime : {total_time:.3f} s")
        print(f"  Time/img  : {total_time / n:.3f} s")
    else:
        print(f"{name}: no valid pairs.")


# =========================
#  1枚ぶんの比較画像を保存
# =========================
def save_example_comparisons(low_dir, gt_dir, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    low_paths = sorted(glob.glob(os.path.join(low_dir, "*.*")))
    if not low_paths:
        print("save_example_comparisons: no files in low_dir")
        return

    for low_path in low_paths:
        fname = os.path.basename(low_path)
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            continue

        low_bgr = cv2.imread(low_path, cv2.IMREAD_COLOR)
        gt_bgr = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        if low_bgr is None or gt_bgr is None:
            continue

        h, w = low_bgr.shape[:2]
        gt_bgr = cv2.resize(gt_bgr, (w, h), interpolation=cv2.INTER_AREA)

        # 各手法で補正
        xcr_bgr  = xcr_enhance_bgr(low_bgr)
        msr_bgr  = msr_enhance_bgr(low_bgr)
        lime_bgr = lime_like_enhance_bgr(low_bgr)

        # 個別保存
        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_dir, f"{base}_low.png"),  low_bgr)
        cv2.imwrite(os.path.join(out_dir, f"{base}_gt.png"),   gt_bgr)
        cv2.imwrite(os.path.join(out_dir, f"{base}_xcr.png"),  xcr_bgr)
        cv2.imwrite(os.path.join(out_dir, f"{base}_msr.png"),  msr_bgr)
        cv2.imwrite(os.path.join(out_dir, f"{base}_lime.png"), lime_bgr)

        # 横並び比較 (low | XCR | MSR | LIME | GT)
        row = np.hstack([low_bgr, xcr_bgr, msr_bgr, lime_bgr, gt_bgr])
        cv2.imwrite(os.path.join(out_dir, f"{base}_comparison.png"), row)

        print(f"Saved comparison images for {fname} in '{out_dir}'")
        break  # 1枚分だけでOKなので最初の1枚で抜ける


if __name__ == "__main__":
    low_dir = "dataset/low"
    gt_dir  = "dataset/gt"

    # 評価（+処理時間）
    evaluate_method(low_dir, gt_dir, xcr_enhance_bgr,       name="XCR")
    evaluate_method(low_dir, gt_dir, msr_enhance_bgr,       name="MSR")
    evaluate_method(low_dir, gt_dir, lime_like_enhance_bgr, name="LIME-like")

    # 比較用画像を1枚だけ保存
    save_example_comparisons(low_dir, gt_dir, out_dir="results")
    