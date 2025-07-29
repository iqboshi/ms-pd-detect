#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有diseased数据的严重程度分级

这个脚本专门用于对所有diseased文件夹中的图像进行严重程度分级测试。
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from severity_classifier import SeverityClassifier
import cv2
from datetime import datetime
import json

import warnings

# 全局抑制GDAL_NODATA相关警告
warnings.filterwarnings("ignore", message=".*GDAL_NODATA.*")
warnings.filterwarnings("ignore", message=".*is not castable to float32.*")
warnings.filterwarnings("ignore", message=".*parsing GDAL_NODATA tag raised ValueError.*")
warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")

try:
    import tifffile
except ImportError:
    tifffile = None

def load_and_preprocess_image(image_path, img_size=(64, 64)):
    """加载和预处理单张图像"""
    try:
        # 处理tif文件
        if image_path.lower().endswith(('.tif', '.tiff')):
            if tifffile is not None:
                # 使用tifffile读取多通道tif
                img_array = tifffile.imread(image_path)
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
                img_array = cv2.resize(img_array, img_size)
                # 转换为PIL图像
                image = Image.fromarray(img_array)
            else:
                # 回退到PIL
                image = Image.open(image_path).convert('RGB')
                image = image.resize(img_size)
        else:
            # 使用PIL读取其他格式
            image = Image.open(image_path).convert('RGB')
            image = image.resize(img_size)
            
        return image
        
    except Exception as e:
        print(f"加载图像 {image_path} 时出错: {e}")
        return None

def test_all_diseased_images():
    """测试所有diseased图像的严重程度分级"""
    print("=" * 60)
    print("测试所有diseased数据的严重程度分级")
    print("=" * 60)
    
    # 使用已知的训练模型
    model_dir = 'outputs/severity_run_20250729_164244'
    model_path = os.path.join(model_dir, 'models', 'best_severity_model.pth')
    
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'models', 'severity_classifier_model.pth')
        
    if not os.path.exists(model_path):
        print(f"在 {model_dir} 中没有找到模型文件")
        return
        
    print(f"使用模型: {model_path}")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        n_levels = checkpoint['n_levels']
        severity_levels = checkpoint['severity_levels']
        img_size = checkpoint.get('img_size', (64, 64))
        
        print(f"严重程度级别数: {n_levels}")
        print(f"级别名称: {severity_levels}")
        print(f"图像尺寸: {img_size}")
        
        # 创建模型（不创建完整的分类器实例）
        from severity_classifier import SeverityClassifier
        
        # 创建一个临时的分类器来获取模型结构
        class TempClassifier:
            def __init__(self):
                self.n_levels = n_levels
                self.img_size = img_size
                self.device = device
                self.severity_levels = severity_levels
                self.model = None
                
            def create_model(self):
                """创建CNN模型"""
                from severity_classifier import SeverityClassifier
                import torch.nn as nn
                
                class SeverityCNN(nn.Module):
                    def __init__(self, num_classes, img_size):
                        super(SeverityCNN, self).__init__()
                        
                        # 卷积层
                        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
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
        
        classifier = TempClassifier()
        classifier.create_model()
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        classifier.model.eval()
        
        print("模型加载完成")
        
        # 获取所有diseased图像
        diseased_dir = os.path.join('data', 'diseased')
        if not os.path.exists(diseased_dir):
            print(f"数据目录不存在: {diseased_dir}")
            return
            
        image_files = []
        for ext in ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([f for f in os.listdir(diseased_dir) if f.lower().endswith(ext)])
        
        if len(image_files) == 0:
            print(f"在 {diseased_dir} 中没有找到图像文件")
            return
            
        print(f"\n找到 {len(image_files)} 张diseased图像，开始批量预测...")
        
        # 数据转换
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        results = []
        failed_count = 0
        
        for i, filename in enumerate(image_files):
            image_path = os.path.join(diseased_dir, filename)
            print(f"处理 {i+1}/{len(image_files)}: {filename}", end=" ")
            
            # 加载和预处理图像
            image = load_and_preprocess_image(image_path, img_size)
            if image is None:
                print("- 加载失败")
                failed_count += 1
                continue
                
            # 转换为tensor
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # 预测
            with torch.no_grad():
                outputs = classifier.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
            predicted_level = severity_levels[predicted_class]
            print(f"- {predicted_level} (置信度: {confidence:.2%})")
            
            results.append({
                'filename': filename,
                'predicted_level': predicted_level,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': probabilities[0].cpu().numpy().tolist()
            })
        
        # 统计结果
        print("\n" + "=" * 60)
        print("批量预测结果统计:")
        print("=" * 60)
        
        if results:
            level_counts = {}
            for r in results:
                level = r['predicted_level']
                level_counts[level] = level_counts.get(level, 0) + 1
                
            print(f"成功处理: {len(results)} 张图像")
            print(f"失败: {failed_count} 张图像")
            print(f"总计: {len(image_files)} 张图像")
            print("\n各级别分布:")
            
            for level in severity_levels:
                count = level_counts.get(level, 0)
                percentage = count / len(results) * 100 if results else 0
                print(f"{level}: {count} 张图像 ({percentage:.1f}%)")
                
            avg_confidence = np.mean([r['confidence'] for r in results])
            print(f"\n平均置信度: {avg_confidence:.2%}")
            
            # 保存详细结果到输出目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"outputs/all_diseased_test_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            result_file = os.path.join(output_dir, "predictions_results.json")
            
            detailed_results = {
                'timestamp': datetime.now().isoformat(),
                'model_path': model_path,
                'total_images': len(image_files),
                'successful_predictions': len(results),
                'failed_predictions': failed_count,
                'severity_levels': severity_levels,
                'level_distribution': level_counts,
                'average_confidence': avg_confidence,
                'detailed_results': results
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
                
            print(f"\nDetailed results saved to: {result_file}")
            
            # 生成简单的分布图
            # 创建输出目录
            output_dir = f"outputs/all_diseased_test_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            plt.figure(figsize=(10, 6))
            levels = list(level_counts.keys())
            counts = list(level_counts.values())
            
            plt.bar(levels, counts, color='skyblue', edgecolor='navy', alpha=0.7)
            plt.title(f'Severity Distribution of All Diseased Images\n(Total: {len(results)} images)', fontsize=14)
            plt.xlabel('Severity Level', fontsize=12)
            plt.ylabel('Number of Images', fontsize=12)
            plt.xticks(rotation=45)
            
            # 添加数值标签
            for i, (level, count) in enumerate(zip(levels, counts)):
                plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
                
            plt.tight_layout()
            
            # 保存图表到输出目录
            chart_file = os.path.join(output_dir, "severity_distribution.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"Distribution chart saved to: {chart_file}")
            plt.show()
            
        else:
            print("没有成功的预测结果")
            
        print("=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_all_diseased_images()