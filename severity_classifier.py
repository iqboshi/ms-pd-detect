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
                    if str(image_array.dtype) != 'uint8':
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
    
    def __init__(self, data_dir='data', n_levels=6, img_size=(64, 64), device=None, output_dir=None, lazy_init=False):
        self.data_dir = data_dir
        self.n_levels = n_levels
        self.img_size = img_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lazy_init = lazy_init
        
        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = os.path.join('outputs', f'severity_run_{timestamp}')
        else:
            self.output_dir = output_dir
        
        # 根据lazy_init决定是否立即创建目录
        if not lazy_init:
            self._ensure_output_dirs()
        else:
            # 只创建主输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"输出目录已创建: {self.output_dir} (lazy initialization)")
        
        print(f"设备: {self.device}")
        print(f"严重程度级别数: {n_levels}")
        
        # 设置随机种子
        self.set_random_seeds(42)
        
        self.model = None
        self.severity_levels = [f"Level_{i+1}" for i in range(n_levels)]
    
    def _ensure_output_dirs(self):
        """确保输出目录存在（按需创建）"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        print(f"输出目录已创建: {self.output_dir}")
        
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
                        if str(img_array.dtype) != 'uint8':
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
        """创建基于对比学习的特征提取模型（优化版，减少土壤阴影等干扰）"""
        class SpatialAttention(nn.Module):
            """空间注意力机制，帮助模型关注重要区域"""
            def __init__(self, in_channels):
                super(SpatialAttention, self).__init__()
                self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                attention = self.conv(x)
                attention = self.sigmoid(attention)
                return x * attention
        
        class ChannelAttention(nn.Module):
            """通道注意力机制，强调重要的光谱通道"""
            def __init__(self, in_channels, reduction=16):
                super(ChannelAttention, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.max_pool = nn.AdaptiveMaxPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(in_channels, in_channels // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_channels // reduction, in_channels)
                )
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                b, c, h, w = x.size()
                avg_out = self.fc(self.avg_pool(x).view(b, c))
                max_out = self.fc(self.max_pool(x).view(b, c))
                attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
                return x * attention
        
        class EnhancedContrastiveEncoder(nn.Module):
            def __init__(self, img_size, feature_dim=128):
                super(EnhancedContrastiveEncoder, self).__init__()
                
                # 多尺度特征提取
                # 第一个卷积块 - 保留细节
                self.conv1 = nn.Sequential(
                    nn.Conv2d(5, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    ChannelAttention(64),
                    SpatialAttention(64)
                )
                
                # 第二个卷积块
                self.conv2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    ChannelAttention(128),
                    SpatialAttention(128)
                )
                
                # 第三个卷积块
                self.conv3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    ChannelAttention(256),
                    SpatialAttention(256)
                )
                
                # 第四个卷积块
                self.conv4 = nn.Sequential(
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    ChannelAttention(512),
                    SpatialAttention(512)
                )
                
                self.pool = nn.MaxPool2d(2, 2)
                self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
                
                # 多尺度特征融合
                self.feature_fusion = nn.Sequential(
                    nn.Linear(512 + 256 + 128, 512),  # 融合多层特征
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3)
                )
                
                # 投影头用于对比学习
                self.projection_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
            nn.Linear(256, feature_dim)
            # L2归一化在forward中实现
                )
                
            def forward(self, x):
                # 多尺度特征提取
                x1 = self.conv1(x)
                x1_pool = self.pool(x1)
                
                x2 = self.conv2(x1_pool)
                x2_pool = self.pool(x2)
                
                x3 = self.conv3(x2_pool)
                x3_pool = self.pool(x3)
                
                x4 = self.conv4(x3_pool)
                
                # 全局平均池化
                feat4 = self.adaptive_pool(x4).view(x4.size(0), -1)
                feat3 = self.adaptive_pool(x3).view(x3.size(0), -1)
                feat2 = self.adaptive_pool(x2).view(x2.size(0), -1)
                
                # 多尺度特征融合
                fused_features = torch.cat([feat4, feat3, feat2], dim=1)
                features = self.feature_fusion(fused_features)
                
                # 投影到对比学习空间
                projections = self.projection_head(features)
                # L2归一化
                projections = nn.functional.normalize(projections, p=2, dim=1)
                
                return features, projections
        
        # 添加L2Norm层
        class L2Norm(nn.Module):
            def __init__(self, dim=1):
                super(L2Norm, self).__init__()
                self.dim = dim
                
            def forward(self, x):
                return nn.functional.normalize(x, p=2, dim=self.dim)
        
        self.model = EnhancedContrastiveEncoder(self.img_size, feature_dim=128).to(self.device)
        print(f"\n增强对比学习模型结构（抗干扰优化版）:")
        print(self.model)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
        print(f"模型优化特性: 空间注意力 + 通道注意力 + 多尺度特征融合")
        
    def create_vegetation_mask(self, image_array):
        """创建高级植被掩码，使用多光谱指数和自适应阈值"""
        if image_array.shape[2] >= 5:
            # 获取各波段（蓝、绿、红、红边、近红外）
            blue = image_array[:, :, 0].astype(np.float32)
            green = image_array[:, :, 1].astype(np.float32)
            red = image_array[:, :, 2].astype(np.float32)
            red_edge = image_array[:, :, 3].astype(np.float32)
            nir = image_array[:, :, 4].astype(np.float32)
            
            # 计算多种植被指数
            # 1. NDVI (归一化植被指数)
            ndvi = (nir - red) / (nir + red + 1e-8)
            
            # 2. EVI (增强植被指数) - 减少土壤和大气影响
            evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
            
            # 3. SAVI (土壤调节植被指数) - 专门处理土壤背景
            L = 0.5  # 土壤亮度校正因子
            savi = ((nir - red) / (nir + red + L)) * (1 + L)
            
            # 4. NDRE (归一化红边指数) - 对叶绿素敏感
            ndre = (nir - red_edge) / (nir + red_edge + 1e-8)
            
            # 5. GNDVI (绿色归一化植被指数)
            gndvi = (nir - green) / (nir + green + 1e-8)
            
            # 综合植被分数（加权组合）
            vegetation_score = (
                0.3 * np.clip(ndvi, 0, 1) +
                0.25 * np.clip(evi, 0, 1) +
                0.2 * np.clip(savi, 0, 1) +
                0.15 * np.clip(ndre, 0, 1) +
                0.1 * np.clip(gndvi, 0, 1)
            )
            
            # 使用自适应阈值（Otsu方法的简化版）
            hist, bins = np.histogram(vegetation_score.flatten(), bins=50, range=(0, 1))
            hist = hist.astype(np.float32)
            
            # 计算类间方差最大的阈值
            total = vegetation_score.size
            current_max = 0
            threshold = 0
            sum_total = np.sum(np.arange(len(hist)) * hist)
            sum_background = 0
            weight_background = 0
            
            for i in range(len(hist)):
                weight_background += hist[i]
                if weight_background == 0:
                    continue
                    
                weight_foreground = total - weight_background
                if weight_foreground == 0:
                    break
                    
                sum_background += i * hist[i]
                mean_background = sum_background / weight_background
                mean_foreground = (sum_total - sum_background) / weight_foreground
                
                # 类间方差
                variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
                
                if variance_between > current_max:
                    current_max = variance_between
                    threshold = bins[i]
            
            # 如果自适应阈值失败，使用固定阈值
            if threshold <= 0:
                threshold = 0.4
            
            vegetation_mask = vegetation_score > threshold
            
            # 亮度和纹理过滤
            brightness = np.mean(image_array[:, :, :3], axis=2)
            brightness_threshold = np.percentile(brightness, 15)
            brightness_mask = brightness > brightness_threshold
            
            # 组合掩码
            combined_mask = vegetation_mask & brightness_mask
            
            return combined_mask.astype(np.uint8) * 255
        else:
            # 简化版本：仅使用RGB通道
            if image_array.shape[2] >= 3:
                red = image_array[:, :, 0].astype(np.float32)
                green = image_array[:, :, 1].astype(np.float32)
                blue = image_array[:, :, 2].astype(np.float32)
                
                # 使用绿色比例和亮度
                green_ratio = green / (red + green + blue + 1e-8)
                brightness = (red + green + blue) / 3
                
                # 植被通常有较高的绿色比例
                green_mask = green_ratio > 0.35
                brightness_mask = brightness > np.percentile(brightness, 25)
                
                combined_mask = green_mask & brightness_mask
            else:
                # 灰度图像：仅使用亮度阈值
                brightness = image_array.squeeze() if len(image_array.shape) == 3 else image_array
                threshold = np.percentile(brightness, 30)
                combined_mask = brightness > threshold
            
            return combined_mask.astype(np.uint8) * 255
    
    def spectral_augmentation(self, image_array):
        """光谱增强，强调植被特征"""
        if image_array.shape[2] >= 5:
            # 增强红边和近红外通道对比
            enhanced = image_array.copy().astype(np.float32)
            
            # 轻微增强红边通道（索引3）
            enhanced[:, :, 3] = np.clip(enhanced[:, :, 3] * 1.1, 0, 255)
            
            # 轻微增强近红外通道（索引4）
            enhanced[:, :, 4] = np.clip(enhanced[:, :, 4] * 1.05, 0, 255)
            
            # 轻微减弱蓝色通道（减少土壤反射）
            enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 0.95, 0, 255)
            
            return enhanced.astype(np.uint8)
        return image_array
    
    def create_contrastive_augmentations(self):
        """创建对比学习的数据增强策略（优化版，减少干扰）"""
        # 强增强：用于创建正样本对，加入植被关注机制
        strong_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # 减少旋转角度
            transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1),  # 减少颜色变化
            transforms.RandomResizedCrop(self.img_size, scale=(0.85, 1.0)),  # 减少裁剪范围
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # 减少模糊程度
        ])
        
        # 弱增强：用于基础变换，更保守
        weak_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),  # 非常轻微的颜色变化
        ])
        
        # 标准化（针对5通道优化）
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456], 
                               std=[0.229, 0.224, 0.225, 0.229, 0.224])
        ])
        
        return strong_augment, weak_augment, normalize
    
    def prepare_contrastive_data(self, image_paths):
        """准备对比学习数据（无需标签）"""
        strong_aug, weak_aug, normalize = self.create_contrastive_augmentations()
        
        # 对比学习数据集
        class ContrastiveDataset(Dataset):
            def __init__(self, image_paths, strong_transform, weak_transform, normalize_transform, img_size):
                self.image_paths = image_paths
                self.strong_transform = strong_transform
                self.weak_transform = weak_transform
                self.normalize = normalize_transform
                self.img_size = img_size
                
            def __len__(self):
                return len(self.image_paths)
                
            def __getitem__(self, idx):
                image_path = self.image_paths[idx]
                
                # 加载图像（复用原有的5通道加载逻辑）
                try:
                    if image_path.lower().endswith(('.tif', '.tiff')):
                        if tifffile is not None:
                            import sys, io
                            old_stderr = sys.stderr
                            sys.stderr = io.StringIO()
                            try:
                                image_array = tifffile.imread(image_path)
                            finally:
                                sys.stderr = old_stderr
                            
                            # 处理为5通道
                            if len(image_array.shape) == 3 and image_array.shape[2] >= 5:
                                image_array = image_array[:, :, :5]
                            elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
                                red_edge = image_array[:, :, 0:1]
                                nir = image_array[:, :, 1:2]
                                image_array = np.concatenate([image_array, red_edge, nir], axis=2)
                            elif len(image_array.shape) == 2:
                                image_array = np.stack([image_array] * 5, axis=-1)
                            
                            if str(image_array.dtype) != 'uint8':
                                image_array = ((image_array - image_array.min()) / 
                                             (image_array.max() - image_array.min() + 1e-8) * 255).astype(np.uint8)
                            
                            image_array = cv2.resize(image_array, self.img_size)
                            
                            # 转换为PIL图像用于数据增强（仅使用前3通道）
                            pil_image = Image.fromarray(image_array[:, :, :3])
                        else:
                            pil_image = Image.open(image_path).convert('RGB')
                            pil_image = pil_image.resize(self.img_size)
                    else:
                        pil_image = Image.open(image_path).convert('RGB')
                        pil_image = pil_image.resize(self.img_size)
                    
                    # 应用光谱增强（在PIL变换之前）
                    if hasattr(self, 'parent_classifier'):
                        image_array = self.parent_classifier.spectral_augmentation(image_array)
                        pil_image = Image.fromarray(image_array[:, :, :3])
                    
                    # 生成两个不同的增强版本（正样本对）
                    aug1 = self.strong_transform(pil_image)
                    aug2 = self.weak_transform(pil_image)
                    
                    # 转换为5通道tensor（改进版）
                    def to_5channel_tensor_enhanced(pil_img, original_5ch=None):
                        img_array = np.array(pil_img)
                        
                        if original_5ch is not None:
                            # 使用原始5通道数据的后两个通道
                            red_edge = original_5ch[:, :, 3:4]
                            nir = original_5ch[:, :, 4:5]
                        else:
                            # 回退方案：从RGB估算
                            red_edge = img_array[:, :, 0:1]  # 使用红色通道
                            nir = img_array[:, :, 1:2]       # 使用绿色通道
                        
                        img_5ch = np.concatenate([img_array, red_edge, nir], axis=2)
                        
                        # 创建植被掩码（如果可能）
                        if hasattr(self, 'parent_classifier') and img_5ch.shape[2] >= 5:
                            mask = self.parent_classifier.create_vegetation_mask(img_5ch)
                            # 将掩码应用为权重（软掩码）
                            mask_weight = mask.astype(np.float32) / 255.0
                            mask_weight = np.expand_dims(mask_weight, axis=2)
                            # 增强植被区域，减弱非植被区域
                            img_5ch = img_5ch * (0.3 + 0.7 * mask_weight)
                        
                        img_tensor = torch.from_numpy(img_5ch).permute(2, 0, 1).float() / 255.0
                        # 应用归一化
                        mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456]).view(5, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224]).view(5, 1, 1)
                        return (img_tensor - mean) / std
                    
                    # 传递原始5通道数据用于更好的特征保持
                    original_5ch = image_array if image_array.shape[2] >= 5 else None
                    aug1_tensor = to_5channel_tensor_enhanced(aug1, original_5ch)
                    aug2_tensor = to_5channel_tensor_enhanced(aug2, original_5ch)
                    
                    return aug1_tensor, aug2_tensor, idx
                    
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    # 返回随机tensor
                    dummy = torch.randn(5, *self.img_size)
                    return dummy, dummy, idx
        
        # 创建对比学习数据集
        strong_aug, weak_aug, normalize = self.create_contrastive_augmentations()
        contrastive_dataset = ContrastiveDataset(image_paths, strong_aug, weak_aug, normalize, self.img_size)
        # 添加parent_classifier引用以使用植被掩码和光谱增强
        contrastive_dataset.parent_classifier = self
        contrastive_loader = DataLoader(contrastive_dataset, batch_size=32, shuffle=True, num_workers=0)
        
        return contrastive_loader
        
        print(f"\n数据分割:")
        print(f"训练集: {len(train_dataset)} 张图像")
        print(f"验证集: {len(val_dataset)} 张图像")
        
        return train_loader, val_loader
    
    def contrastive_loss(self, features1, features2, temperature=0.07, hard_negative_weight=1.0):
        """计算改进的InfoNCE对比学习损失（抗干扰优化版）"""
        batch_size = features1.size(0)
        
        # L2归一化
        features1 = nn.functional.normalize(features1, dim=1)
        features2 = nn.functional.normalize(features2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features1, features2.T) / temperature
        
        # 创建标签（对角线为正样本）
        labels = torch.arange(batch_size).to(self.device)
        
        # 基础InfoNCE损失
        base_loss = nn.functional.cross_entropy(similarity_matrix, labels)
        
        # 困难负样本挖掘：增加对困难负样本的关注
        if hard_negative_weight > 1.0:
            # 获取负样本相似度（非对角线元素）
            mask = torch.eye(batch_size, device=self.device).bool()
            negative_similarities = similarity_matrix.masked_fill(mask, float('-inf'))
            
            # 找到最困难的负样本（相似度最高的负样本）
            hard_negatives, _ = torch.max(negative_similarities, dim=1)
            
            # 计算困难负样本损失
            positive_similarities = torch.diag(similarity_matrix)
            hard_negative_loss = torch.mean(torch.relu(hard_negatives - positive_similarities + 0.2))
            
            # 组合损失
            total_loss = base_loss + hard_negative_weight * hard_negative_loss
        else:
            total_loss = base_loss
        
        return total_loss
    
    def train_contrastive_model(self, contrastive_loader, epochs=100, learning_rate=0.001):
        """训练对比学习模型"""
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        history = {
            'contrastive_loss': [],
            'lr': []
        }
        
        print(f"\n开始对比学习训练，共{epochs}个epoch")
        print(f"学习率: {learning_rate}")
        print(f"设备: {self.device}")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            train_pbar = tqdm(contrastive_loader, desc=f'Epoch {epoch+1}/{epochs} [Contrastive]')
            for batch_idx, (aug1, aug2, _) in enumerate(train_pbar):
                aug1, aug2 = aug1.to(self.device), aug2.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                features1, projections1 = self.model(aug1)
                features2, projections2 = self.model(aug2)
                
                # 计算改进的对比损失（包含困难负样本挖掘）
                # 在训练初期使用较低的困难负样本权重，后期逐渐增加
                hard_neg_weight = 1.0 + 0.5 * (epoch / epochs)  # 从1.0增加到1.5
                loss = self.contrastive_loss(projections1, projections2, 
                                            temperature=0.07, 
                                            hard_negative_weight=hard_neg_weight)
                
                loss.backward()
                
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss/num_batches:.4f}'
                })
            
            # 计算平均损失
            avg_loss = total_loss / num_batches
            
            # 记录历史
            history['contrastive_loss'].append(avg_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            print(f'\nEpoch {epoch+1}/{epochs}:')
            print(f'Contrastive Loss: {avg_loss:.4f}')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # 每10个epoch保存一次模型
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }, os.path.join(self.output_dir, 'models', f'contrastive_model_epoch_{epoch+1}.pth'))
                print(f'模型已保存: epoch_{epoch+1}')
                
            scheduler.step()
            print('-' * 60)
        
        # 保存最终模型
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_loss': avg_loss
        }, os.path.join(self.output_dir, 'models', 'final_contrastive_model.pth'))
        
        print(f'\n对比学习训练完成! 最终损失: {avg_loss:.4f}')
        return history
    
    def extract_features(self, image_paths):
        """提取所有图像的特征用于聚类"""
        self.model.eval()
        features_list = []
        
        # 创建简单的数据加载器用于特征提取
        class FeatureDataset(Dataset):
            def __init__(self, image_paths, img_size):
                self.image_paths = image_paths
                self.img_size = img_size
                
            def __len__(self):
                return len(self.image_paths)
                
            def __getitem__(self, idx):
                image_path = self.image_paths[idx]
                
                try:
                    if image_path.lower().endswith(('.tif', '.tiff')):
                        if tifffile is not None:
                            import sys, io
                            old_stderr = sys.stderr
                            sys.stderr = io.StringIO()
                            try:
                                image_array = tifffile.imread(image_path)
                            finally:
                                sys.stderr = old_stderr
                            
                            # 处理为5通道
                            if len(image_array.shape) == 3 and image_array.shape[2] >= 5:
                                image_array = image_array[:, :, :5]
                            elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
                                red_edge = image_array[:, :, 0:1]
                                nir = image_array[:, :, 1:2]
                                image_array = np.concatenate([image_array, red_edge, nir], axis=2)
                            elif len(image_array.shape) == 2:
                                image_array = np.stack([image_array] * 5, axis=-1)
                            
                            if str(image_array.dtype) != 'uint8':
                                image_array = ((image_array - image_array.min()) / 
                                             (image_array.max() - image_array.min() + 1e-8) * 255).astype(np.uint8)
                            
                            image_array = cv2.resize(image_array, self.img_size)
                        else:
                            pil_image = Image.open(image_path).convert('RGB')
                            pil_image = pil_image.resize(self.img_size)
                            image_array = np.array(pil_image)
                            red_edge = image_array[:, :, 0:1]
                            nir = image_array[:, :, 1:2]
                            image_array = np.concatenate([image_array, red_edge, nir], axis=2)
                    else:
                        pil_image = Image.open(image_path).convert('RGB')
                        pil_image = pil_image.resize(self.img_size)
                        image_array = np.array(pil_image)
                        red_edge = image_array[:, :, 0:1]
                        nir = image_array[:, :, 1:2]
                        image_array = np.concatenate([image_array, red_edge, nir], axis=2)
                    
                    # 转换为tensor并归一化
                    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
                    mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456]).view(5, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224]).view(5, 1, 1)
                    image_tensor = (image_tensor - mean) / std
                    
                    return image_tensor, idx
                    
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    return torch.randn(5, *self.img_size), idx
        
        feature_dataset = FeatureDataset(image_paths, self.img_size)
        feature_loader = DataLoader(feature_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        print("\n提取图像特征用于聚类...")
        with torch.no_grad():
            for images, indices in tqdm(feature_loader, desc="提取特征"):
                images = images.to(self.device)
                features, _ = self.model(images)
                features_list.append(features.cpu().numpy())
        
        # 合并所有特征
        all_features = np.concatenate(features_list, axis=0)
        print(f"提取完成，特征维度: {all_features.shape}")
        
        return all_features
    
    def cluster_severity_levels(self, features, image_paths):
        """使用K-means聚类将图像分为6个严重程度级别"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        print(f"\n开始聚类分析，目标级别数: {self.n_levels}")
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 可选：使用PCA降维以提高聚类效果
        if features.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            features_scaled = pca.fit_transform(features_scaled)
            print(f"PCA降维后特征维度: {features_scaled.shape}")
        
        # K-means聚类
        kmeans = KMeans(n_clusters=self.n_levels, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # 计算每个聚类中心到原点的距离，用于排序严重程度
        cluster_centers = kmeans.cluster_centers_
        center_distances = np.linalg.norm(cluster_centers, axis=1)
        
        # 根据距离排序聚类（距离越大，严重程度越高）
        sorted_indices = np.argsort(center_distances)
        
        # 重新映射标签，使其按严重程度排序
        severity_labels = np.zeros_like(cluster_labels)
        for new_label, old_label in enumerate(sorted_indices):
            severity_labels[cluster_labels == old_label] = new_label
        
        # 统计每个级别的数量
        level_counts = np.bincount(severity_labels, minlength=self.n_levels)
        
        print("\n聚类结果统计:")
        for i, count in enumerate(level_counts):
            percentage = count / len(severity_labels) * 100
            print(f"Level {i+1}: {count} 张图像 ({percentage:.1f}%)")
        
        # 保存聚类结果
        clustering_results = {
            'image_paths': image_paths,
            'severity_labels': severity_labels.tolist(),
            'level_counts': level_counts.tolist(),
            'cluster_centers': cluster_centers.tolist(),
            'scaler_params': {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
        }
        
        # 保存到文件
        results_path = os.path.join(self.output_dir, 'clustering_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(clustering_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n聚类结果已保存到: {results_path}")
        
        return severity_labels, level_counts
    
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
                        if str(img_array.dtype) != 'uint8':
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
    
    def run_contrastive_severity_classification(self, epochs=100, learning_rate=0.001):
        """运行基于对比学习的无监督严重程度分级流程"""
        try:
            print("="*60)
            print("开始基于对比学习的无监督疾病严重程度分级")
            print("="*60)
            
            # 1. 加载diseased图像
            image_paths = self.load_diseased_images()
            
            # 2. 创建对比学习模型
            self.create_model()
            
            # 3. 准备对比学习数据（无需标签）
            contrastive_loader = self.prepare_contrastive_data(image_paths)
            
            # 4. 训练对比学习模型
            print("\n第一阶段：对比学习特征提取训练")
            history = self.train_contrastive_model(contrastive_loader, epochs, learning_rate)
            
            # 5. 提取所有图像的特征
            print("\n第二阶段：特征提取")
            features = self.extract_features(image_paths)
            
            # 6. 使用聚类进行无监督分级
            print("\n第三阶段：无监督聚类分级")
            severity_labels, level_counts = self.cluster_severity_levels(features, image_paths)
            
            # 7. 绘制训练历史
            self.plot_contrastive_history(history)
            
            # 8. 可视化聚类结果
            self.visualize_clustering_results(features, severity_labels, image_paths)
            
            # 9. 保存训练日志
            self.save_contrastive_log(history, level_counts, len(image_paths))
            
            print("\n" + "="*60)
            print("基于对比学习的无监督严重程度分级完成!")
            print(f"总图像数量: {len(image_paths)}")
            print(f"聚类级别数: {self.n_levels}")
            print(f"输出目录: {self.output_dir}")
            print("="*60)
            
            return severity_labels, level_counts
            
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def plot_contrastive_history(self, history):
        """绘制对比学习训练历史"""
        plt.figure(figsize=(12, 4))
        
        # 对比学习损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['contrastive_loss'], label='Contrastive Loss', color='blue')
        plt.title('Contrastive Learning Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 学习率曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['lr'], label='Learning Rate', color='green')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = os.path.join(self.output_dir, 'plots', 'contrastive_training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"对比学习训练历史图已保存到: {plot_path}")
    
    def visualize_clustering_results(self, features, labels, image_paths):
        """可视化聚类结果"""
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        print("\n生成聚类结果可视化...")
        
        # 使用t-SNE降维到2D用于可视化
        if features.shape[1] > 50:
            # 先用PCA降维到50维，再用t-SNE
            pca = PCA(n_components=50, random_state=42)
            features_pca = pca.fit_transform(features)
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
            features_2d = tsne.fit_transform(features_pca)
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
            features_2d = tsne.fit_transform(features)
        
        # 创建颜色映射
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_levels))
        
        plt.figure(figsize=(12, 8))
        
        # 绘制散点图
        for i in range(self.n_levels):
            mask = (labels == i)
            if mask.any():  # 使用.any()方法而不是np.any()
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=[colors[i]], label=f'Level {i+1}', alpha=0.7, s=50)
        
        plt.title('Clustering Results Visualization (t-SNE)', fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = os.path.join(self.output_dir, 'plots', 'clustering_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"聚类可视化图已保存到: {plot_path}")
    
    def save_contrastive_log(self, history, level_counts, total_images):
        """保存对比学习训练日志"""
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': 'Contrastive Learning + K-means Clustering',
            'total_images': total_images,
            'n_levels': self.n_levels,
            'img_size': self.img_size,
            'device': str(self.device),
            'final_contrastive_loss': history['contrastive_loss'][-1] if history['contrastive_loss'] else 0,
            'level_distribution': {
                f'Level_{i+1}': int(count) for i, count in enumerate(level_counts)
            },
            'level_percentages': {
                f'Level_{i+1}': f"{count/total_images*100:.1f}%" for i, count in enumerate(level_counts)
            },
            'training_epochs': len(history['contrastive_loss']),
            'output_directory': self.output_dir
        }
        
        # 保存到JSON文件
        log_path = os.path.join(self.output_dir, 'logs', 'contrastive_training_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n对比学习训练日志已保存到: {log_path}")

if __name__ == "__main__":
    # 创建严重程度分类器
    classifier = SeverityClassifier(
        data_dir='data',
        n_levels=6,  # 默认6级
        img_size=(64, 64)
    )
    
    # 使用对比学习的无监督方法进行疾病严重程度分级
    classifier.run_contrastive_severity_classification(epochs=100, learning_rate=0.001)
    
    # 如果需要使用原有的有监督方法，可以取消下面的注释
    # classifier.run_severity_classification(epochs=50, learning_rate=0.001)