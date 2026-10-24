智慧农业-荔枝成熟度检测数据集】YOLO txt格式｜4类生长阶段｜1005张高清果园图片
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2b8d5a237fa441b6ab5e41c7563c7ce5.png)

数据集亮点:
✅ 1005张荔枝高清图片，真实果园场景采集
✅ txt格式标注，兼容YOLO等框架
✅ 4类生长阶段检测，标注框总数 ​​超1,800个​​
✅ 专业农学标注：成熟度特征精准标注
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/210a6a4d75a04fff8036ec8cb639723c.png)

数据集规格
1. 标注格式：txt
2. 类别标签及中文意思：
• 0: blossom —— 花期
• 1: green —— 青果期
• 2: ripe —— 成熟期
• 3: young —— 幼果期
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3625279ff5a84861a6bacf02c55e60f7.png)


3. 数据划分：
• 训练集：804张（80%）
• 验证集：100张（10%）
• 测试集：101张（10%）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cfb83a1d27db414cbd575fe55dd49b25.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/068f5ada1f864176aa13bc5d00f702d3.png)
根据你提供的荔枝成熟度检测数据集信息，我为你整理了详细的数据规格表，并编写了基于最新 **YOLOv11** 框架的训练代码。

### 数据集规格描述表

该数据集专为荔枝全生长周期的智能监测设计，涵盖了从开花到成熟的四个关键阶段。

| 属性 | 详细说明 |
| :--- | :--- |
| **数据集名称** | 荔枝成熟度检测数据集 |
| **图片总数** | **1005 张** |
| **图像质量** | 高清，真实果园复杂场景采集 |
| **标注格式** | **YOLO txt** 格式 (归一化坐标) |
| **标注总量** | 超过 **1,800 个** 标注框 |
| **检测类别** | **4 类** (花期、幼果、青果、成熟) |
| **数据划分** | 训练集 (804张, 80%)验证集 (100张, 10%)测试集 (101张, 10%) |

#### 类别标签映射

在训练和推理时，请确保类别索引与含义对应一致：

| 类别 ID | 英文标签 | 中文含义 | 特征描述 |
| :---: | :--- | :--- | :--- |
| **0** | **blossom** | **花期** | 荔枝花朵开放阶段 |
| **1** | **green** | **青果期** | 果实已成形但呈青色，未成熟 |
| **2** | **ripe** | **成熟期** | 果皮转红，达到采摘标准 |
| **3** | **young** | **幼果期** | 刚结出的小果实 |

### YOLOv11 训练代码

以下代码基于 Ultralytics 最新的 **YOLOv11** 模型。请确保你的数据集目录结构如下：

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── litchi.yaml  <-- 下面的配置文件
```

#### 配置文件 (litchi.yaml)

首先，在代码同级目录下创建 `litchi.yaml` 文件：

```yaml
path: ./dataset  # 数据集根目录
train: images/train  # 训练集图片路径 (相对于path)
val: images/val      # 验证集图片路径
test: images/test    # 测试集图片路径

# 类别数量
nc: 4

# 类别名称
names: ['blossom', 'green', 'ripe', 'young']
```

#### 训练脚本 (train.py)

```python
from ultralytics import YOLO

def train_litchi_model():
    # 1. 加载模型
    # 使用 YOLOv11 nano 版本 (速度快，适合果园实时监测)
    # 如果需要更高精度，可改为 'yolo11m.pt' 或 'yolo11l.pt'
    model = YOLO('yolo11n.pt')

    # 2. 开始训练
    results = model.train(
        data='litchi.yaml',    # 指定数据配置文件
        epochs=100,            # 训练轮数，根据收敛情况可调整
        imgsz=640,             # 输入图片尺寸
        batch=16,              # 每批次图片数量，根据显存调整
        device=0,              # 使用 GPU (device=0) 或 CPU (device='cpu')
        workers=4,             # 数据加载线程数
        project='Litchi_Detect', # 保存项目的根目录
        name='exp_v11',        # 实验名称
        patience=10,           # 早停策略，10轮无提升则停止
        optimizer='AdamW',     # 优化器选择
        lr0=0.001,             # 初始学习率
        hsv_h=0.015,           # 数据增强：色调
        hsv_s=0.7,             # 数据增强：饱和度
        hsv_v=0.4,             # 数据增强：明度
        degrees=10.0,          # 数据增强：旋转角度
        flipud=0.5             # 数据增强：上下翻转概率 (果园场景适用)
    )

    # 3. 评估模型 (在验证集上)
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}") # 打印 mAP

if __name__ == '__main__':
    train_litchi_model()
```

### 代码关键点解析

1. **模型选择 (`yolo11n.pt`)**：考虑到果园监测可能部署在边缘设备（如Jetson Nano或树莓派），代码默认使用了 **Nano** 版本，它在保持较高精度的同时速度最快。如果你有高性能服务器，建议换成 `yolo11m.pt` 以获得更高的 mAP。
2. **数据增强**：
   - `hsv_h/s/v`：果园光照变化大（早晨、正午、阴天），调整色彩空间增强有助于模型适应不同光线。
   - `flipud=0.5`：荔枝生长在树上，上下翻转不会改变物体的物理属性，增加此增强可有效扩充数据集。
3. **早停机制 (`patience=10`)**：防止过拟合。如果验证集指标在10个epoch内没有提升，训练会自动停止。

通过上述配置与脚本，即可高效完成针对荔枝生长周期的模型训练任务。
