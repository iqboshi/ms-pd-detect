import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import json
from tqdm import tqdm
import random
import cv2
from scipy.stats import entropy
import warnings

# 全局抑制GDAL_NODATA相关警告
warnings.filterwarnings("ignore", message=".*GDAL_NODATA.*")
warnings.filterwarnings("ignore", message=".*is not castable to float32.*")
warnings.filterwarnings("ignore", message=".*parsing GDAL_NODATA tag raised ValueError.*")
warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")

try:
    # 在导入tifffile时抑制所有警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tifffile
except ImportError:
    tifffile = None
    print("警告: tifffile库未安装，将尝试使用其他方法读取TIFF文件")

class SeverityDataset(Dataset):
    """用于疾病严重程度分级的数据集类"""
    
    def __init__(self, image_paths, labels, transform=None, img_size=(64, 64)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        try:
            # 处理tif文件
            if image_path.lower().endswith(('.tif', '.tiff')):
                if tifffile is not None:
                    # 使用tifffile读取多通道tif，完全抑制GDAL_NODATA警告
                    import sys
                    import io
                    import contextlib
                    
                    # 临时重定向stderr来抑制tifffile内部警告
                    old_stderr = sys.stderr
                    sys.stderr = io.StringIO()
                    try:
                        image_array = tifffile.imread(image_path)
                    finally:
                        sys.stderr = old_stderr
                    
                    # 处理多通道数据，使用所有5个波段（蓝、绿、红、红边、近红外）
                    if len(image_array.shape) == 3 and image_array.shape[2] >= 5:
                        # Use all 5 bands: 蓝、绿、红、红边、近红外
                        image_array = image_array[:, :, :5]
                    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                        # If only 4 channels, duplicate the last channel to make 5
                        last_channel = image_array[:, :, -1:]
                        image_array = np.concatenate([image_array, last_channel], axis=2)
                    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
                        # If only 3 channels (RGB), duplicate red and green to make 5 bands
                        red_edge = image_array[:, :, 0:1]  # Use red as red edge
                        nir = image_array[:, :, 1:2]       # Use green as NIR
                        image_array = np.concatenate([image_array, red_edge, nir], axis=2)
                    elif len(image_array.shape) == 2:
                        # 灰度图复制为5通道
                        image_array = np.stack([image_array] * 5, axis=-1)
                    elif len(image_array.shape) == 3 and image_array.shape[2] < 5:
                        # If less than 5 channels, pad by repeating the last channel
                        while image_array.shape[2] < 5:
                            last_channel = image_array[:, :, -1:]
                            image_array = np.concatenate([image_array, last_channel], axis=2)
                    
                    # 确保数据类型正确
                    if image_array.dtype != np.uint8:
                        # 归一化到0-255
                        image_array = ((image_array - image_array.min()) / 
                                     (image_array.max() - image_array.min() + 1e-8) * 255).astype(np.uint8)
                    
                    # 调整大小
                    image_array = cv2.resize(image_array, self.img_size)
                    # 对于5通道数据，保持为numpy数组，不转换为PIL
                    if image_array.shape[2] == 5:
                        image = image_array  # 保持为numpy数组
                    else:
                        # 转换为PIL图像（仅用于3通道或更少）
                        image = Image.fromarray(image_array)
                else:
                    # 回退到PIL，但可能失败
                    try:
                        image = Image.open(image_path).convert('RGB')
                        image = image.resize(self.img_size)
                        # Convert RGB to 5 channels
                        image_array = np.array(image)
                        red_edge = image_array[:, :, 0:1]  # Use red as red edge
                        nir = image_array[:, :, 1:2]       # Use green as NIR
                        image_array = np.concatenate([image_array, red_edge, nir], axis=2)
                        # 保持为5通道numpy数组
                        image = image_array
                    except:
                        # 如果PIL也失败，创建随机图像
                        print(f"警告: 无法读取 {image_path}，使用随机图像")
                        image_array = np.random.randint(0, 256, (*self.img_size, 5), dtype=np.uint8)
                        # 保持为5通道numpy数组
                        image = image_array
            else:
                # 使用PIL读取其他格式
                image = Image.open(image_path).convert('RGB')
                image = image.resize(self.img_size)
                # Convert RGB to 5 channels
                image_array = np.array(image)
                red_edge = image_array[:, :, 0:1]  # Use red as red edge
                nir = image_array[:, :, 1:2]       # Use green as NIR
                image_array = np.concatenate([image_array, red_edge, nir], axis=2)
                # 保持为5通道numpy数组，不转换为PIL
                image = image_array
            
            # 如果使用了tifffile并且有5通道数据，直接处理numpy数组
            if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 5:
                # 直接处理5通道numpy数组
                if self.transform and hasattr(self.transform, 'transforms'):
                    # 对于5通道数据，只应用归一化，跳过PIL相关的变换
                    image = image.astype(np.float32) / 255.0
                    image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
                    # 应用归一化（如果有的话）
                    for t in self.transform.transforms:
                        if isinstance(t, transforms.Normalize):
                            # 对5通道数据使用前3通道的归一化参数，后2通道使用相同参数
                            mean = list(t.mean) + [t.mean[0], t.mean[1]]  # 扩展到5通道
                            std = list(t.std) + [t.std[0], t.std[1]]      # 扩展到5通道
                            image = transforms.Normalize(mean=mean, std=std)(image)
                            break
                else:
                    # 没有变换，直接转换
                    image = image.astype(np.float32) / 255.0
                    image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
            else:
                # 对于PIL图像（3通道），使用原有的变换流程
                if self.transform:
                    image = self.transform(image)
                else:
                    # 默认转换为tensor
                    image = transforms.ToTensor()(image)
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回空白图像（5通道）
            image = torch.zeros(5, *self.img_size)
            
        return image, label

class SeverityClassifier:
    """疾病严重程度分级器"""
    
    def __init__(self, data_dir='data', n_levels=6, img_size=(64, 64), device=None, output_dir=None):
        self.data_dir = data_dir
        self.n_levels = n_levels
        self.img_size = img_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = os.path.join('outputs', f'severity_run_{timestamp}')
        else:
            self.output_dir = output_dir
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        
        print(f"输出目录已创建: {self.output_dir}")
        print(f"设备: {self.device}")
        print(f"严重程度级别数: {n_levels}")
        
        # 设置随机种子
        self.set_random_seeds(42)
        
        self.model = None
        self.severity_levels = [f"Level_{i+1}" for i in range(n_levels)]
        
    def set_random_seeds(self, seed=42):
        """设置随机种子以确保结果可重现"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def load_diseased_images(self):
        """加载diseased文件夹中的所有tif图像"""
        diseased_dir = os.path.join(self.data_dir, 'diseased')
        
        if not os.path.exists(diseased_dir):
            raise FileNotFoundError(f"找不到diseased文件夹: {diseased_dir}")
            
        image_paths = []
        for file in os.listdir(diseased_dir):
            if file.lower().endswith(('.tif', '.tiff')):
                image_paths.append(os.path.join(diseased_dir, file))
                
        if len(image_paths) == 0:
            raise ValueError(f"在{diseased_dir}中没有找到tif图像文件")
            
        print(f"找到 {len(image_paths)} 张diseased图像")
        return image_paths
    
    def analyze_image_features(self, image_paths):
        """分析图像特征用于自动分级"""
        print("正在分析图像特征...")
        
        features = []
        valid_paths = []
        
        for path in tqdm(image_paths, desc="分析图像"):
            try:
                # 处理tif文件
                if path.lower().endswith(('.tif', '.tiff')):
                    if tifffile is not None:
                        # 使用tifffile读取多通道tif，完全抑制GDAL_NODATA警告
                        import sys
                        import io
                        
                        # 临时重定向stderr来抑制tifffile内部警告
                        old_stderr = sys.stderr
                        sys.stderr = io.StringIO()
                        try:
                            img_array = tifffile.imread(path)
                        finally:
                            sys.stderr = old_stderr
                        # 如果是多通道，取前3个通道或转换为RGB
                        if len(img_array.shape) == 3 and img_array.shape[2] > 3:
                            img_array = img_array[:, :, :3]  # 取前3个通道
                        elif len(img_array.shape) == 2:
                            # 灰度图转RGB
                            img_array = np.stack([img_array] * 3, axis=-1)
                        
                        # 确保数据类型正确
                        if img_array.dtype != np.uint8:
                            # 归一化到0-255
                            img_array = ((img_array - img_array.min()) / 
                                       (img_array.max() - img_array.min() + 1e-8) * 255).astype(np.uint8)
                        
                        # 调整大小
                        img_array = cv2.resize(img_array, self.img_size)
                    else:
                        # 回退到PIL，但可能失败
                        try:
                            image = Image.open(path).convert('RGB')
                            image = image.resize(self.img_size)
                            img_array = np.array(image)
                        except:
                            # 如果PIL也失败，跳过这个图像
                            print(f"警告: 无法读取 {path}，跳过")
                            continue
                else:
                    # 使用PIL读取其他格式
                    image = Image.open(path).convert('RGB')
                    image = image.resize(self.img_size)
                    img_array = np.array(image)
                
                # 计算多种特征
                # 1. 平均亮度
                brightness = np.mean(img_array)
                
                # 2. 对比度（标准差）
                contrast = np.std(img_array)
                
                # 3. 绿色通道强度（植物健康指标）
                green_intensity = np.mean(img_array[:, :, 1])
                
                # 4. 红绿比值（疾病指标）
                red_mean = np.mean(img_array[:, :, 0])
                green_mean = np.mean(img_array[:, :, 1])
                rg_ratio = red_mean / (green_mean + 1e-8)
                
                # 5. 图像熵（复杂度）
                hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
                img_entropy = entropy(hist + 1e-8)
                
                # 综合特征分数（可以根据需要调整权重）
                feature_score = (
                    0.2 * (255 - brightness) +  # 较暗的图像可能病情更重
                    0.3 * rg_ratio +            # 红绿比值高可能表示病情重
                    0.2 * contrast +            # 对比度
                    0.2 * img_entropy +         # 复杂度
                    0.1 * (255 - green_intensity)  # 绿色强度低可能表示病情重
                )
                
                features.append(feature_score)
                valid_paths.append(path)
                
            except Exception as e:
                print(f"分析图像 {path} 时出错: {e}")
                continue
                
        return np.array(features), valid_paths
    
    def assign_severity_levels(self, features, image_paths):
        """根据特征分数分配严重程度级别"""
        if len(features) == 0:
            raise ValueError("没有有效的图像特征数据，无法进行分级")
            
        features = np.array(features)
        print(f"\n特征统计:")
        print(f"有效图像数量: {len(features)}")
        print(f"特征分数范围: {features.min():.2f} - {features.max():.2f}")
        print(f"特征分数均值: {features.mean():.2f}")
        
        # 使用分位数方法分配级别
        percentiles = np.linspace(0, 100, self.n_levels + 1)
        thresholds = np.percentile(features, percentiles)
        
        labels = []
        level_counts = [0] * self.n_levels
        
        for feature_score in features:
            # 找到对应的级别
            level = 0
            for i in range(len(thresholds) - 1):
                if thresholds[i] <= feature_score < thresholds[i + 1]:
                    level = i
                    break
            if feature_score >= thresholds[-1]:
                level = self.n_levels - 1
                
            labels.append(level)
            level_counts[level] += 1
            
        print("\n严重程度分布:")
        for i, count in enumerate(level_counts):
            print(f"Level {i+1}: {count} 张图像 ({count/len(labels)*100:.1f}%)")
            
        return labels, thresholds
    
    def create_model(self):
        """创建CNN模型用于严重程度分类"""
        class SeverityCNN(nn.Module):
            def __init__(self, num_classes, img_size):
                super(SeverityCNN, self).__init__()
                
                # 卷积层
                self.conv1 = nn.Conv2d(5, 32, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(128)
                
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)
                
                # 计算全连接层输入大小
                conv_output_size = (img_size[0] // 8) * (img_size[1] // 8) * 128
                
                # 全连接层
                self.fc1 = nn.Linear(conv_output_size, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, num_classes)
                
                self.relu = nn.ReLU()
                
            def forward(self, x):
                # 卷积块1
                x = self.pool(self.relu(self.bn1(self.conv1(x))))
                # 卷积块2
                x = self.pool(self.relu(self.bn2(self.conv2(x))))
                # 卷积块3
                x = self.pool(self.relu(self.bn3(self.conv3(x))))
                
                # 展平
                x = x.view(x.size(0), -1)
                
                # 全连接层
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.dropout(self.relu(self.fc2(x)))
                x = self.fc3(x)
                
                return x
        
        self.model = SeverityCNN(self.n_levels, self.img_size).to(self.device)
        print(f"\n模型结构:")
        print(self.model)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
        
    def prepare_data(self, image_paths, labels, train_ratio=0.8):
        """准备训练和验证数据"""
        # 数据增强
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 分割数据
        indices = list(range(len(image_paths)))
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_paths = [image_paths[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_paths = [image_paths[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        # 创建数据集
        train_dataset = SeverityDataset(train_paths, train_labels, train_transform, self.img_size)
        val_dataset = SeverityDataset(val_paths, val_labels, val_transform, self.img_size)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        print(f"\n数据分割:")
        print(f"训练集: {len(train_dataset)} 张图像")
        print(f"验证集: {len(val_dataset)} 张图像")
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, epochs=50, learning_rate=0.001):
        """训练模型"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        print(f"\n开始训练 {epochs} 个epoch...")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for images, labels in train_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct_train/total_train:.2f}%'
                })
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct_train / total_train
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for images, labels in val_pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*correct_val/total_val:.2f}%'
                    })
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * correct_val / total_val
            
            # 记录历史
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # 学习率调度
            scheduler.step(val_acc)
            
            print(f'Epoch [{epoch+1}/{epochs}]:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_path = os.path.join(self.output_dir, 'models', 'best_severity_model.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'n_levels': self.n_levels,
                    'img_size': self.img_size,
                    'severity_levels': self.severity_levels
                }, best_model_path)
                print(f'  新的最佳模型已保存! 验证准确率: {val_acc:.2f}%')
            else:
                patience_counter += 1
                
            # 早停
            if patience_counter >= patience:
                print(f'\n早停触发! {patience} 个epoch没有改善')
                break
                
            print('-' * 60)
        
        # 保存训练历史
        history = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
        
        return history
    
    def plot_training_history(self, history):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(history['train_losses'], label='Training Loss', color='blue')
        ax1.plot(history['val_losses'], label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(history['train_accuracies'], label='Training Accuracy', color='blue')
        ax2.plot(history['val_accuracies'], label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        history_path = os.path.join(self.output_dir, 'plots', 'severity_training_history.png')
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存到: {history_path}")
        plt.show()
    
    def evaluate_model(self, val_loader):
        """评估模型性能"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="评估模型"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算准确率
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        print(f"\n验证集准确率: {accuracy:.4f}")
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=self.severity_levels))
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.severity_levels, 
                   yticklabels=self.severity_levels,
                   cbar_kws={'label': 'Number of Images'})
        plt.title('Confusion Matrix for Severity Classification\n(All Diseased Images)', fontsize=16, pad=20)
        plt.xlabel('Predicted Severity Level', fontsize=12)
        plt.ylabel('True Severity Level', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存混淆矩阵
        cm_path = os.path.join(self.output_dir, 'plots', 'severity_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {cm_path}")
        plt.show()
        
        return accuracy
    
    def predict_single_image(self, img_path):
        """预测单张图像的严重程度级别"""
        if self.model is None:
            print("请先训练或加载模型")
            return None, 0.0
        
        try:
            # 使用SeverityDataset的图像加载逻辑
            dataset = SeverityDataset([img_path], [0], transform=None, img_size=self.img_size)
            image, _ = dataset[0]
            
            # 添加批次维度
            image = image.unsqueeze(0).to(self.device)
            
            # 预测
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                predicted_class = predicted.item()
                confidence = probabilities[0][predicted_class].item()
                
                result = self.severity_levels[predicted_class]
            
            return result, confidence
            
        except Exception as e:
            print(f"预测图像 {img_path} 时出错: {e}")
            return "Level_1", 0.0
    
    def generate_full_confusion_matrix(self, image_paths, true_labels):
        """使用所有diseased数据生成混淆矩阵"""
        print("\n生成使用所有diseased数据的混淆矩阵...")
        
        # 设置随机种子确保结果可重现
        self.set_random_seeds(42)
        
        self.model.eval()
        predicted_labels = []
        
        # 数据转换（不使用数据增强）
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 对所有图像进行预测
        for i, image_path in enumerate(tqdm(image_paths, desc="预测所有图像")):
            try:
                # 加载和预处理图像
                if image_path.lower().endswith(('.tif', '.tiff')):
                    if tifffile is not None:
                        # 使用tifffile读取多通道tif，完全抑制GDAL_NODATA警告
                        import sys
                        import io
                        
                        # 临时重定向stderr来抑制tifffile内部警告
                        old_stderr = sys.stderr
                        sys.stderr = io.StringIO()
                        try:
                            img_array = tifffile.imread(image_path)
                        finally:
                            sys.stderr = old_stderr
                        
                        # 如果是多通道，取前3个通道或转换为RGB
                        if len(img_array.shape) == 3 and img_array.shape[2] > 3:
                            img_array = img_array[:, :, :3]  # 取前3个通道
                        elif len(img_array.shape) == 2:
                            # 灰度图转RGB
                            img_array = np.stack([img_array] * 3, axis=-1)
                        
                        # 确保数据类型正确
                        if img_array.dtype != np.uint8:
                            # 归一化到0-255
                            img_array = ((img_array - img_array.min()) / 
                                       (img_array.max() - img_array.min() + 1e-8) * 255).astype(np.uint8)
                        
                        # 调整大小
                        img_array = cv2.resize(img_array, self.img_size)
                        # 转换为PIL图像
                        image = Image.fromarray(img_array)
                    else:
                        # 回退到PIL
                        image = Image.open(image_path).convert('RGB')
                        image = image.resize(self.img_size)
                else:
                    # 使用PIL读取其他格式
                    image = Image.open(image_path).convert('RGB')
                    image = image.resize(self.img_size)
                
                # 转换为tensor并预测
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    _, predicted = torch.max(outputs, 1)
                    predicted_labels.append(predicted.item())
                    
            except Exception as e:
                print(f"预测图像 {image_path} 时出错: {e}")
                # 如果预测失败，使用默认标签
                predicted_labels.append(0)
        
        # 生成混淆矩阵
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # 计算准确率
        accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
        print(f"\n所有数据准确率: {accuracy:.4f}")
        
        # 分类报告
        print("\n所有数据分类报告:")
        print(classification_report(true_labels, predicted_labels, 
                                  target_names=self.severity_levels, zero_division=0))
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.severity_levels, 
                   yticklabels=self.severity_levels,
                   cbar_kws={'label': 'Number of Images'})
        plt.title('Confusion Matrix for Severity Classification\n(All Diseased Images)', fontsize=16, pad=20)
        plt.xlabel('Predicted Severity Level', fontsize=12)
        plt.ylabel('True Severity Level', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存混淆矩阵（覆盖之前的文件）
        cm_path = os.path.join(self.output_dir, 'plots', 'severity_confusion_matrix_all_data.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"\n完整数据混淆矩阵已保存到: {cm_path}")
        plt.show()
        
        return accuracy
    
    def save_training_log(self, history, accuracy, image_count, thresholds):
        """保存训练日志"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'n_levels': self.n_levels,
                'img_size': self.img_size,
                'device': str(self.device),
                'image_count': image_count
            },
            'severity_thresholds': thresholds.tolist(),
            'severity_levels': self.severity_levels,
            'training_history': history,
            'final_accuracy': accuracy,
            'best_validation_accuracy': history['best_val_acc']
        }
        
        log_path = os.path.join(self.output_dir, 'logs', 'severity_training_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"训练日志已保存到: {log_path}")
    
    def run_severity_classification(self, epochs=50, learning_rate=0.001):
        """运行完整的严重程度分类流程"""
        print("=" * 60)
        print("疾病严重程度分级系统")
        print("=" * 60)
        
        try:
            # 1. 加载diseased图像
            image_paths = self.load_diseased_images()
            
            # 2. 分析图像特征
            features, valid_paths = self.analyze_image_features(image_paths)
            
            # 3. 分配严重程度级别
            labels, thresholds = self.assign_severity_levels(features, valid_paths)
            
            # 4. 创建模型
            self.create_model()
            
            # 5. 准备数据
            train_loader, val_loader = self.prepare_data(valid_paths, labels)
            
            # 6. 训练模型
            history = self.train_model(train_loader, val_loader, epochs, learning_rate)
            
            # 7. 绘制训练历史
            self.plot_training_history(history)
            
            # 8. 评估模型
            accuracy = self.evaluate_model(val_loader)
            
            # 8.5. 生成使用所有数据的混淆矩阵
            self.generate_full_confusion_matrix(valid_paths, labels)
            
            # 9. 保存最终模型
            final_model_path = os.path.join(self.output_dir, 'models', 'severity_classifier_model.pth')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'n_levels': self.n_levels,
                'img_size': self.img_size,
                'severity_levels': self.severity_levels,
                'thresholds': thresholds
            }, final_model_path)
            print(f"最终模型已保存到: {final_model_path}")
            
            # 10. 保存训练日志
            self.save_training_log(history, accuracy, len(valid_paths), thresholds)
            
            print("\n=" * 2)
            print("严重程度分级训练完成!")
            print(f"最佳验证准确率: {history['best_val_acc']:.2f}%")
            print(f"最终验证准确率: {accuracy:.4f}")
            print(f"输出目录: {self.output_dir}")
            print("=" * 60)
            
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # 创建严重程度分类器
    classifier = SeverityClassifier(
        data_dir='data',
        n_levels=6,  # 默认6级
        img_size=(64, 64)
    )
    
    # 运行分类训练
    classifier.run_severity_classification(epochs=50, learning_rate=0.001)