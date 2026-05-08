import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 任务1：数据准备 
print("任务1：数据准备 ")
digits = load_digits()

image_count = digits.images.shape[0]
print(f"数据集中总图像数量：{image_count} 张")

image_shape = digits.images.shape[1:]
print(f"单张图像大小：{image_shape[0]} × {image_shape[1]} 像素")

feature_dim = digits.data.shape[1]
print(f"图像展开后的特征向量维度：{feature_dim} 维")

labels = np.unique(digits.target)
print(f"数据集中的类别标签：{labels}")

label_counts = np.bincount(digits.target)
print("\n每个类别对应的样本数量：")
for num, count in enumerate(label_counts):
    print(f"数字 {num}：{count} 张")

# 显示并保存样本图像
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r)
    plt.title(f"{digits.target[i]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("homework7/digits_samples.png", dpi=300, bbox_inches='tight')
plt.show()

#  任务2：数据划分
print("任务2：数据集划分（训练集 / 测试集）")

X_train, X_test, y_train, y_test = train_test_split(
    digits.data,
    digits.target,
    test_size=0.25,
    random_state=42
)

print(f"训练集样本数量：{X_train.shape[0]} 张")
print(f"测试集样本数量：{X_test.shape[0]} 张")
print(f"总样本数量：{X_train.shape[0] + X_test.shape[0]} 张")

# 任务3：特征表示 
print("任务3：特征表示 - 图像转换为特征向量")

sample_image = digits.images[0]
sample_vector = digits.data[0]

print(f"原始图像形状: {sample_image.shape}")
print(f"展开后向量形状: {sample_vector.shape}")

# 任务4：模型训练 
print("任务4：传统机器学习模型训练与分类")

models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

accuracy_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = acc
    print(f"【{name}】")
    print(f"  测试集准确率: {acc:.4f} ({acc*100:.2f}%)\n")

# 任务5：结果比较
print("任务5：结果比较与分析")

# 1. 整理成表格输出
print("\n 不同模型测试准确率汇总表：")
print("-" * 35)
print(f"{'模型':<20} | {'测试准确率':<10}")
print("-" * 35)
for name in ["KNN", "Naive Bayes", "Logistic Regression", "SVM", "Decision Tree", "Random Forest"]:
    acc = accuracy_results[name]
    print(f"{name:<20} | {acc:.4f} ({acc*100:.2f}%)")

# 任务6：错误样本分析
print("任务6：错误样本分析（基于最优模型 KNN）")

# 选择 KNN 模型
best_model = KNeighborsClassifier()
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# 计算并绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))

disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix (KNN)")
plt.tight_layout()
plt.savefig("homework7/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# 找出错误分类的样本
error_indices = np.where(y_pred != y_test)[0]
print(f"\n【错误分类样本总数】：{len(error_indices)} 个")
print(f"测试集总数：{len(y_test)} 个")

# 可视化展示错误样本
if len(error_indices) > 0:
    plt.figure(figsize=(12, 6))
    n_show = min(6, len(error_indices))  # 显示6个错误样本
    for i in range(n_show):
        idx = error_indices[i]
        plt.subplot(2, 3, i + 1)
        img = X_test[idx].reshape(8, 8)
        plt.imshow(img, cmap=plt.cm.gray_r)
        plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}", color='red')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("homework7/error_samples.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(" 错误分类样本已保存为：homework7/error_samples.png")

# 分析最容易混淆的数字
print("\n【最容易被混淆的数字对】")
confusion_pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append((i, j, cm[i, j]))

# 取混淆最多的前5个
confusion_pairs.sort(key=lambda x: x[2], reverse=True)
for true_digit, pred_digit, count in confusion_pairs[:5]:
    print(f"数字 {true_digit} 被误判为 {pred_digit} → {count} 次")

