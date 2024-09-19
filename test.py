import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 18,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


# 给定的混淆矩阵
conf_matrix = np.array([[1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                        [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                        [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                        [0.00, 0.00, 0.00, 0.97, 0.00, 0.00, 0.03, 0.00, 0.00, 0.00, 0.00],
                        [0.00, 0.00, 0.00, 0.00, 0.98, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02],
                        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                        [0.00, 0.00, 0.00, 0.01, 0.02, 0.00, 0.97, 0.00, 0.00, 0.00, 0.00],
                        [0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.00, 0.98, 0.00, 0.00, 0.00],
                        [0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.93, 0.07, 0.00],
                        [0.00, 0.00, 0.00, 0.03, 0.12, 0.00, 0.00, 0.00, 0.00, 0.85, 0.00],
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00]])

# 根冠腐烂（Crown Leaf RustRot）、叶锈病（Leaf Rust）、小麦散斑，又名小麦黑穗病（Wheat Loose Smut）、小麦白粉病（Powdery Mildew）、小麦孢囊线虫病（Wheat cyst nematode）、小麦赤霉病（Wheat scab）、小麦红蜘蛛（Wheat Red Spider）、小麦茎基腐（Wheat stalk rot）、小麦全蚀病（Wheat Take-all）、小麦纹枯病（wheat sharp eyespot）、小麦蚜虫病（Wheat Aphids）。

labels = ['叶锈病', '小麦黑穗病', '小麦红蜘蛛', '小麦赤霉病', '小麦茎基腐',
          '根冠腐烂', '小麦白粉病', '小麦蚜虫病', '小麦孢囊线虫病',
          '小麦纹枯病', '小麦全蚀病']

# 设置图像大小
plt.figure(figsize=(10, 10))

# 显示混淆矩阵
plt.imshow(conf_matrix, cmap=plt.cm.Blues)

# 在混淆矩阵中标注数量信息
thresh = conf_matrix.max() / 2
for x in range(conf_matrix.shape[0]):
    for y in range(conf_matrix.shape[1]):
        plt.text(y, x, f'{conf_matrix[x, y]:.2f}',  # 显示小数点后两位
                 verticalalignment='center',
                 horizontalalignment='center',
                 color="white" if conf_matrix[x, y] > thresh else "black",
                 fontdict={'family': 'SimSun', 'size': 14})  # 使用Times New Roman字体

# 设置坐标轴标签
plt.xticks(range(len(labels)), labels, rotation=90, fontsize=12, fontname='SimSun')
plt.yticks(range(len(labels)), labels, fontsize=12, fontname='SimSun')

# 在图的左右和上方添加True Label和Predict Label的说明
plt.xlabel('Predict Label', fontsize=14, fontname='Times New Roman')  # 横坐标添加说明
plt.ylabel('True Label', fontsize=14, fontname='Times New Roman')     # 纵坐标添加说明

# 保证图像布局美观
plt.tight_layout()

# 保存图像
plt.savefig('conf_matrix_with_labels.jpg', bbox_inches='tight')

# 显示图像
plt.show()

