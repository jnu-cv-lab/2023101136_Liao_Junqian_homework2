import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# -------------------------- FFT频谱分析 --------------------------
def compute_fft_spectrum(img):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(f_shift) + 1)
    return magnitude

# -------------------------- 生成测试图--------------------------
def generate_checkerboard(size=512, block_size=16):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i // block_size + j // block_size) % 2 == 0:
                img[i, j] = 255
            else:
                img[i, j] = 0
    return img

def generate_chirp(size=512):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    chirp = np.sin(np.pi * size * r**2 / 2)
    chirp = (chirp - chirp.min()) / (chirp.max() - chirp.min()) * 255
    return chirp.astype(np.uint8)

# -------------------------- 直接下采样 --------------------------
def downsample_direct(img, scale=4):
    return img[::scale, ::scale]

# -------------------------- 高斯滤波下采样 --------------------------
def gaussian_blur_downsample(img, scale=4, sigma=1.8):
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return blurred[::scale, ::scale]

# -------------------------- 生成测试图 --------------------------
checker = generate_checkerboard()
chirp = generate_chirp()

checker_direct = downsample_direct(checker, 4)
chirp_direct = downsample_direct(chirp, 4)

checker_gauss = gaussian_blur_downsample(checker, 4, 1.8)
chirp_gauss = gaussian_blur_downsample(chirp, 4, 1.8)

# -------------------------- 不同σ对比 --------------------------
sigmas = [0.5, 1.0, 1.8, 2.0, 4.0]
checker_sigma_imgs, checker_sigma_ffts = [], []
chirp_sigma_imgs, chirp_sigma_ffts = [], []

for s in sigmas:
    checker_sigma_imgs.append(gaussian_blur_downsample(checker, 4, s))
    checker_sigma_ffts.append(compute_fft_spectrum(checker_sigma_imgs[-1]))
    chirp_sigma_imgs.append(gaussian_blur_downsample(chirp, 4, s))
    chirp_sigma_ffts.append(compute_fft_spectrum(chirp_sigma_imgs[-1]))

# -------------------------- 基础频谱 --------------------------
checker_fft = compute_fft_spectrum(checker)
checker_direct_fft = compute_fft_spectrum(checker_direct)
checker_gauss_fft = compute_fft_spectrum(checker_gauss)

chirp_fft = compute_fft_spectrum(chirp)
chirp_direct_fft = compute_fft_spectrum(chirp_direct)
chirp_gauss_fft = compute_fft_spectrum(chirp_gauss)

# -------------------------- 画图1：棋盘格 --------------------------
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
axes1[0,0].imshow(checker, cmap='gray'); axes1[0,0].set_title('Original Checkerboard'); axes1[0,0].axis('off')
axes1[0,1].imshow(checker_direct, cmap='gray'); axes1[0,1].set_title('Direct Downsampling M=4'); axes1[0,1].axis('off')
axes1[0,2].imshow(checker_gauss, cmap='gray'); axes1[0,2].set_title('Gaussian Downsampling M=4'); axes1[0,2].axis('off')
axes1[1,0].imshow(checker_fft, cmap='gray'); axes1[1,0].set_title('Original Spectrum'); axes1[1,0].axis('off')
axes1[1,1].imshow(checker_direct_fft, cmap='gray'); axes1[1,1].set_title('Direct Spectrum'); axes1[1,1].axis('off')
axes1[1,2].imshow(checker_gauss_fft, cmap='gray'); axes1[1,2].set_title('Gaussian Spectrum'); axes1[1,2].axis('off')
plt.tight_layout()
plt.savefig("01_checkerboard_comparison.jpg", dpi=300, bbox_inches='tight')  

# -------------------------- 画图2：Chirp --------------------------
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
axes2[0,0].imshow(chirp, cmap='gray'); axes2[0,0].set_title('Original Chirp'); axes2[0,0].axis('off')
axes2[0,1].imshow(chirp_direct, cmap='gray'); axes2[0,1].set_title('Direct Downsampling M=4'); axes2[0,1].axis('off')
axes2[0,2].imshow(chirp_gauss, cmap='gray'); axes2[0,2].set_title('Gaussian Downsampling M=4'); axes2[0,2].axis('off')
axes2[1,0].imshow(chirp_fft, cmap='gray'); axes2[1,0].set_title('Original Spectrum'); axes2[1,0].axis('off')
axes2[1,1].imshow(chirp_direct_fft, cmap='gray'); axes2[1,1].set_title('Direct Spectrum'); axes2[1,1].axis('off')
axes2[1,2].imshow(chirp_gauss_fft, cmap='gray'); axes2[1,2].set_title('Gaussian Spectrum'); axes2[1,2].axis('off')
plt.tight_layout()
plt.savefig("02_chirp_comparison.jpg", dpi=300, bbox_inches='tight')  # 保存

# -------------------------- 画图3：棋盘σ对比 --------------------------
fig3, axes3 = plt.subplots(2, 5, figsize=(20, 8))
fig3.suptitle('Checkerboard Comparison with Different σ (M=4)', fontsize=16)
for i in range(5):
    axes3[0,i].imshow(checker_sigma_imgs[i], cmap='gray'); axes3[0,i].set_title(f'σ={sigmas[i]}'); axes3[0,i].axis('off')
    axes3[1,i].imshow(checker_sigma_ffts[i], cmap='gray'); axes3[1,i].set_title(f'Spectrum σ={sigmas[i]}'); axes3[1,i].axis('off')
plt.tight_layout()
plt.savefig("03_checkerboard_sigma.jpg", dpi=300, bbox_inches='tight')  # 保存

# -------------------------- 画图4：Chirp σ对比 --------------------------
fig4, axes4 = plt.subplots(2, 5, figsize=(20, 8))
fig4.suptitle('Chirp Comparison with Different σ (M=4)', fontsize=16)
for i in range(5):
    axes4[0,i].imshow(chirp_sigma_imgs[i], cmap='gray'); axes4[0,i].set_title(f'σ={sigmas[i]}'); axes4[0,i].axis('off')
    axes4[1,i].imshow(chirp_sigma_ffts[i], cmap='gray'); axes4[1,i].set_title(f'Spectrum σ={sigmas[i]}'); axes4[1,i].axis('off')
plt.tight_layout()
plt.savefig("04_chirp_sigma.jpg", dpi=300, bbox_inches='tight')  # 保存

# -------------------------- 自适应函数 --------------------------
def compute_gradient_magnitude(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return grad_mag

def estimate_local_M(grad_mag, block_size=8, M_min=3, M_max=5.5):
    h, w = grad_mag.shape
    M_map = np.zeros_like(grad_mag, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = grad_mag[i:i+block_size, j:j+block_size]
            avg_grad = np.mean(block)
            M = M_max - (avg_grad / 255.0) * (M_max - M_min)
            M = np.clip(M, M_min, M_max)
            M_map[i:i+block_size, j:j+block_size] = M
    M_map = cv2.GaussianBlur(M_map, (3,3), 0.6)
    return M_map

def adaptive_downsample(img, M_map, block_size=8, target_scale=4):
    h, w = img.shape
    th, tw = h // target_scale, w // target_scale
    out = np.zeros((th, tw), dtype=np.float32)

    M_map_big = cv2.resize(M_map, (w, h), interpolation=cv2.INTER_CUBIC)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            bi, bj = i, j
            ei, ej = min(i+block_size, h), min(j+block_size, w)
            ti, tj = bi // target_scale, bj // target_scale
            tih, tjw = (ei - bi)//target_scale, (ej - bj)//target_scale

            block = img[bi:ei, bj:ej].astype(np.float32)
            M = float(M_map_big[(bi+ei)//2, (bj+ej)//2])
            sigma = 0.45 * M
            ksize = int(round(6 * sigma))
            if ksize < 3:
                ksize = 3
            if ksize % 2 == 0:
                ksize += 1

            blurred = cv2.GaussianBlur(block, (ksize, ksize), sigma)
            small = cv2.resize(blurred, (tjw, tih), interpolation=cv2.INTER_AREA)
            out[ti:ti+tih, tj:tj+tjw] = small

    return np.clip(out, 0, 255).astype(np.uint8)

def uniform_downsample(img, M=4):
    sigma = 0.45 * M
    ksize = int(6 * sigma + 1) | 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return blurred[::4, ::4]

def upsample(img, h, w):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

# -------------------------- 棋盘图 --------------------------
img1 = checker
g1 = compute_gradient_magnitude(img1)
m1 = estimate_local_M(g1)
a1 = adaptive_downsample(img1, m1)
u1 = uniform_downsample(img1)
ar1 = upsample(a1, 512, 512)
ur1 = upsample(u1, 512, 512)
e1 = np.abs(img1 - ar1)
eu1 = np.abs(img1 - ur1)
ps1 = psnr(img1, ar1)
pu1 = psnr(img1, ur1)

plt.figure(figsize=(18, 12))
plt.subplot(3,3,1); plt.imshow(img1, cmap='gray'); plt.title('Checkerboard Original'); plt.axis('off')
plt.subplot(3,3,2); plt.imshow(g1, cmap='gray'); plt.title('Gradient'); plt.axis('off')
plt.subplot(3,3,3); plt.imshow(m1, cmap='viridis'); plt.title('Local M Map'); plt.axis('off')
plt.subplot(3,3,4); plt.imshow(a1, cmap='gray'); plt.title('Adaptive Downsampling'); plt.axis('off')
plt.subplot(3,3,5); plt.imshow(u1, cmap='gray'); plt.title('Uniform Downsampling'); plt.axis('off')
plt.subplot(3,3,7); plt.imshow(ar1, cmap='gray'); plt.title(f'Adaptive Restore\nPSNR={ps1:.1f}'); plt.axis('off')
plt.subplot(3,3,8); plt.imshow(ur1, cmap='gray'); plt.title(f'Uniform Restore\nPSNR={pu1:.1f}'); plt.axis('off')
plt.subplot(3,3,6); plt.imshow(e1, cmap='hot'); plt.title('Adaptive Error'); plt.axis('off')
plt.subplot(3,3,9); plt.imshow(eu1, cmap='hot'); plt.title('Uniform Error'); plt.axis('off')
plt.suptitle('Adaptive Downsampling - Checkerboard', fontsize=18)
plt.tight_layout()
plt.savefig("05_adaptive_checkerboard.jpg", dpi=300, bbox_inches='tight')  

# -------------------------- Chirp图 --------------------------
img2 = chirp
g2 = compute_gradient_magnitude(img2)
m2 = estimate_local_M(g2)
a2 = adaptive_downsample(img2, m2)
u2 = uniform_downsample(img2)
ar2 = upsample(a2, 512, 512)
ur2 = upsample(u2, 512, 512)
e2 = np.abs(img2 - ar2)
eu2 = np.abs(img2 - ur2)
ps2 = psnr(img2, ar2)
pu2 = psnr(img2, ur2)

plt.figure(figsize=(18, 12))
plt.subplot(3,3,1); plt.imshow(img2, cmap='gray'); plt.title('Chirp Original'); plt.axis('off')
plt.subplot(3,3,2); plt.imshow(g2, cmap='gray'); plt.title('Gradient'); plt.axis('off')
plt.subplot(3,3,3); plt.imshow(m2, cmap='viridis'); plt.title('Local M Map'); plt.axis('off')
plt.subplot(3,3,4); plt.imshow(a2, cmap='gray'); plt.title('Adaptive Downsampling'); plt.axis('off')
plt.subplot(3,3,5); plt.imshow(u2, cmap='gray'); plt.title('Uniform Downsampling'); plt.axis('off')
plt.subplot(3,3,7); plt.imshow(ar2, cmap='gray'); plt.title(f'Adaptive Restore\nPSNR={ps2:.1f}'); plt.axis('off')
plt.subplot(3,3,8); plt.imshow(ur2, cmap='gray'); plt.title(f'Uniform Restore\nPSNR={pu2:.1f}'); plt.axis('off')
plt.subplot(3,3,6); plt.imshow(e2, cmap='hot'); plt.title('Adaptive Error'); plt.axis('off')
plt.subplot(3,3,9); plt.imshow(eu2, cmap='hot'); plt.title('Uniform Error'); plt.axis('off')
plt.suptitle('Adaptive Downsampling - Chirp', fontsize=18)
plt.tight_layout()
plt.savefig("06_adaptive_chirp.jpg", dpi=300, bbox_inches='tight')  # 保存

# -------------------------- 3. 人脸图 固定 M=4，自适应 σ --------------------------
img3 = cv2.imread("pic.jpg", cv2.IMREAD_GRAYSCALE)
img3 = cv2.resize(img3, (512, 512))

g3 = compute_gradient_magnitude(img3)

def get_adaptive_sigma_map(grad_mag, block_size=8, sigma_min=0.8, sigma_max=3.0):
    h, w = grad_mag.shape
    sigma_map = np.zeros_like(grad_mag, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = grad_mag[i:i+block_size, j:j+block_size]
            avg_grad = np.mean(block)
            sigma = sigma_max - (avg_grad / 255.0) * (sigma_max - sigma_min)
            sigma = np.clip(sigma, sigma_min, sigma_max)
            sigma_map[i:i+block_size, j:j+block_size] = sigma
    sigma_map = cv2.GaussianBlur(sigma_map, (3, 3), 0.5)
    return sigma_map

sigma_map = get_adaptive_sigma_map(g3, block_size=8, sigma_min=0.8, sigma_max=3.0)

def adaptive_sigma_downsample(img, sigma_map, block_size=8, scale=4):
    h, w = img.shape
    th, tw = h // scale, w // scale
    out = np.zeros((th, tw), dtype=np.float32)
    sigma_map_big = cv2.resize(sigma_map, (w, h), interpolation=cv2.INTER_CUBIC)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            bi, bj = i, j
            ei, ej = min(i+block_size, h), min(j+block_size, w)
            ti, tj = bi // scale, bj // scale
            tih, tjw = (ei - bi)//scale, (ej - bj)//scale

            block = img[bi:ei, bj:ej].astype(np.float32)
            sigma = float(sigma_map_big[(bi+ei)//2, (bj+ej)//2])

            ksize = int(round(6 * sigma))
            if ksize < 3:
                ksize = 3
            if ksize % 2 == 0:
                ksize += 1

            blurred = cv2.GaussianBlur(block, (ksize, ksize), sigma)
            small = cv2.resize(blurred, (tjw, tih), tjw, interpolation=cv2.INTER_AREA)
            out[ti:ti+tih, tj:tj+tjw] = small

    return np.clip(out, 0, 255).astype(np.uint8)

# 均匀下采样
def uniform_downsample(img, M=4):
    sigma = 0.45 * M
    ksize = int(6 * sigma + 1) | 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return blurred[::4, ::4]

a3 = adaptive_sigma_downsample(img3, sigma_map)    
u3 = uniform_downsample(img3)                    

ar3 = upsample(a3, 512, 512)
ur3 = upsample(u3, 512, 512)

e3 = np.abs(img3 - ar3)
eu3 = np.abs(img3 - ur3)

ps3 = psnr(img3, ar3)
pu3 = psnr(img3, ur3)

plt.figure(figsize=(18, 12))
plt.subplot(3,3,1); plt.imshow(img3, cmap='gray'); plt.title('Face Original'); plt.axis('off')
plt.subplot(3,3,2); plt.imshow(g3, cmap='gray'); plt.title('Gradient'); plt.axis('off')
plt.subplot(3,3,3); plt.imshow(sigma_map, cmap='viridis'); plt.title('Adaptive σ Map'); plt.axis('off')
plt.subplot(3,3,4); plt.imshow(a3, cmap='gray'); plt.title('Adaptive σ Downsample M=4'); plt.axis('off')
plt.subplot(3,3,5); plt.imshow(u3, cmap='gray'); plt.title('Uniform Downsample M=4'); plt.axis('off')
plt.subplot(3,3,7); plt.imshow(ar3, cmap='gray'); plt.title(f'Adaptive Restore\nPSNR={ps3:.1f}');plt.title(f'Adaptive Restore\nPSNR={ps3:.1f}'); plt.axis('off')
plt.subplot(3,3,8); plt.imshow(ur3, cmap='gray'); plt.title(f'Uniform Restore\nPSNR={pu3:.1f}'); plt.axis('off')
plt.subplot(3,3,6); plt.imshow(e3, cmap='hot'); plt.title('Adaptive Error'); plt.axis('off')
plt.subplot(3,3,9); plt.imshow(eu3, cmap='hot'); plt.title('Uniform Error'); plt.axis('off')

plt.suptitle('Adaptive σ Downsample (Fixed M=4)', fontsize=18)
plt.tight_layout()
plt.savefig("07_adaptive_face.jpg", dpi=300, bbox_inches='tight')

plt.show()