import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# 1. 创建输出目录

output_dir = "transform_results"
os.makedirs(output_dir, exist_ok=True)

# 2. 生成测试图像

def create_test_image(size=600):
    img = np.ones((size, size, 3), dtype=np.uint8) * 255  
    
    # 绘制矩形
    cv2.rectangle(img, (150, 150), (350, 350), (0, 0, 255), 2)
    
    # 绘制圆
    cv2.circle(img, (480, 300), 80, (0, 255, 0), 2)
    
    # 绘制平行线
    cv2.line(img, (120, 20), (120, 580), (255, 0, 0), 2)
    cv2.line(img, (320, 20), (320, 580), (255, 0, 0), 2)
    
    # 绘制垂直线
    cv2.line(img, (300, 50), (300, 550), (0, 0, 0), 2)
    cv2.line(img, (50, 300), (550, 300), (0, 0, 0), 2)
    
    return img

# 生成原始测试图
original = create_test_image()
cv2.imwrite(os.path.join(output_dir, "original.png"), original)

# 计算变换后画布边界

def get_transformed_corners(img, M, is_perspective=False):
    rows, cols = img.shape[:2]
    # 原图像四个角点
    corners = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]]).reshape(-1, 1, 2)
    if is_perspective:
        transformed_corners = cv2.perspectiveTransform(corners, M)
    else:
        transformed_corners = cv2.transform(corners, M)
    return transformed_corners


# 3. 实现三种变换

# 相似变换
def similarity_transform(img):

    M = np.array([
        [ 0.69282032,  0.40000000,  107.84609691],
        [-0.40000000,  0.69282032,  157.84609691]
    ], dtype=np.float32)
    # 计算变换后四个角点，确定新画布大小
    transformed_corners = get_transformed_corners(img, M)
    min_x, max_x = transformed_corners[:, 0, 0].min(), transformed_corners[:, 0, 0].max()
    min_y, max_y = transformed_corners[:, 0, 1].min(), transformed_corners[:, 0, 1].max()
    
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))
    
    M[0, 2] -= min_x
    M[1, 2] -= min_y
    
    transformed = cv2.warpAffine(img, M, (new_width, new_height), borderValue=(255,255,255))
    return transformed

# 仿射变换
def affine_transform(img):
    # 生成仿射变换矩阵
    M = np.array([
        [1.2667, 0.6000, -83.3333],
        [-0.3333,  1.0000,  66.6667]
    ], dtype=np.float32)
    
    transformed_corners = get_transformed_corners(img, M)
    min_x, max_x = transformed_corners[:, 0, 0].min(), transformed_corners[:, 0, 0].max()
    min_y, max_y = transformed_corners[:, 0, 1].min(), transformed_corners[:, 0, 1].max()
    
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))
    
    M[0, 2] -= min_x
    M[1, 2] -= min_y
    
    transformed = cv2.warpAffine(img, M, (new_width, new_height), borderValue=(255,255,255))
    return transformed

# 透视变换
def perspective_transform(img):
    rows, cols = img.shape[:2]
    pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    
    pts2 = np.float32([
        [cols*0.25, rows*0.15],   # 左上点大幅向中心收缩
        [cols*0.75, rows*0.15],   # 右上点大幅向中心收缩
        [0, rows],                # 左下点保持在底部
        [cols, rows]              # 右下点保持在底部
    ])
    # 生成透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    # 计算变换后四个角点，确定新画布大小
    transformed_corners = get_transformed_corners(img, M, is_perspective=True)
    min_x, max_x = transformed_corners[:, 0, 0].min(), transformed_corners[:, 0, 0].max()
    min_y, max_y = transformed_corners[:, 0, 1].min(), transformed_corners[:, 0, 1].max()
    
    # 新画布尺寸
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))
    
    # 构建平移矩阵，把偏移量加进去
    translate_M = np.array([[1, 0, -min_x],
                            [0, 1, -min_y],
                            [0, 0, 1]], dtype=np.float32)
    # 合并变换矩阵：先透视，再平移
    M = translate_M @ M
    # 执行透视变换，白色背景
    transformed = cv2.warpPerspective(img, M, (new_width, new_height), borderValue=(255,255,255))
    return transformed


# 4. 执行所有变换
sim_img = similarity_transform(original)
aff_img = affine_transform(original)
pers_img = perspective_transform(original)

# 保存变换后的图像
cv2.imwrite(os.path.join(output_dir, "similarity.png"), sim_img)
cv2.imwrite(os.path.join(output_dir, "affine.png"), aff_img)
cv2.imwrite(os.path.join(output_dir, "perspective.png"), pers_img)


plt.figure(figsize=(24, 6))

# 原图
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title('Original Image', fontsize=14, fontweight='bold')
plt.axis('off')

# 相似变换
plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB))
plt.title('Similarity Transform', fontsize=14, fontweight='bold')
plt.axis('off')

# 仿射变换
plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(aff_img, cv2.COLOR_BGR2RGB))
plt.title('Affine Transform', fontsize=14, fontweight='bold')
plt.axis('off')

# 透视变换
plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(pers_img, cv2.COLOR_BGR2RGB))
plt.title('Perspective Transform', fontsize=14, fontweight='bold')
plt.axis('off')

# 显示窗口
plt.tight_layout()
plt.show()

print(f"All images saved to: {output_dir}")