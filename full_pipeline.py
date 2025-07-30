#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的图像处理流水线
功能：大图切片 → 健康/疾病分类 → 疾病严重程度分级 → 重组可视化
"""

import os
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import json
import warnings
from tqdm import tqdm
import shutil

# 抑制警告
warnings.filterwarnings("ignore")

try:
    import tifffile
except ImportError:
    tifffile = None
    print("警告: tifffile库未安装，将使用PIL处理TIFF文件")

from image_classifier import ImageClassifier
from severity_classifier import SeverityClassifier
from pipeline_config import PipelineConfig, ConfigTemplates, create_pipeline_config, validate_config

class FullPipeline:
    def __init__(self, config=None, **kwargs):
        """
        初始化完整处理流水线
        
        Args:
            config: 配置对象(PipelineConfig)或配置字典，如果为None则使用默认配置
            **kwargs: 额外的配置参数，会覆盖config中的对应参数
        """
        # 处理配置
        if config is None:
            # 使用默认配置
            config_obj = PipelineConfig()
        elif isinstance(config, PipelineConfig):
            # 直接使用配置对象
            config_obj = config
        elif isinstance(config, dict):
            # 从字典创建配置对象
            config_obj = PipelineConfig()
            config_obj.load_from_dict(config)
        else:
            raise ValueError("config必须是PipelineConfig对象、字典或None")
        
        # 应用额外的参数
        if kwargs:
            config_obj.load_from_dict(kwargs)
        
        # 验证配置
        config_dict = config_obj.to_dict()
        is_valid, errors = validate_config(config_dict)
        if not is_valid:
            raise ValueError(f"配置验证失败: {'; '.join(errors)}")
        
        # 设置实例属性
        self.tile_size = config_obj.tile_size
        self.overlap = config_obj.overlap
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = config_obj.get_output_dir()
        self.health_classifier_config = config_obj.health_classifier_config
        self.severity_classifier_config = config_obj.severity_classifier_config
        self.color_map = config_obj.color_map
        
        # 创建输出目录结构
        self.create_output_dirs()
        
        print(f"流水线初始化完成，输出目录: {self.output_dir}")
        print(f"切片大小: {self.tile_size}, 重叠: {self.overlap}")
        print(f"严重程度级别数: {self.severity_classifier_config['n_levels']}")
        print(f"配置模板: 外部配置文件")
    
    def create_output_dirs(self):
        """创建基础输出目录结构（按需创建，避免空目录）"""
        # 只创建主输出目录，其他目录在需要时创建
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"主输出目录已创建: {self.output_dir}")
        print("其他子目录将在需要时自动创建")
    
    def load_large_image(self, image_path):
        """
        加载大图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            tuple: (图像数组, nodata掩码)
        """
        print(f"正在加载大图像: {image_path}")
        
        try:
            nodata_mask = None
            
            if image_path.lower().endswith(('.tif', '.tiff')) and tifffile is not None:
                # 使用tifffile处理多通道TIFF
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    image = tifffile.imread(image_path)
                
                # 检测nodata值
                # 首先尝试从TIFF元数据获取nodata值
                try:
                    with tifffile.TiffFile(image_path) as tif:
                        page = tif.pages[0]
                        # 检查是否有nodata标签
                        nodata_value = None
                        if hasattr(page, 'tags') and 'GDAL_NODATA' in page.tags:
                            try:
                                nodata_value = float(page.tags['GDAL_NODATA'].value)
                            except:
                                pass
                        
                        if nodata_value is not None:
                            print(f"检测到GDAL nodata值: {nodata_value}")
                            if len(image.shape) == 3:
                                nodata_mask = np.all(image == nodata_value, axis=2)
                            else:
                                nodata_mask = (image == nodata_value)
                        else:
                            # 回退到检查0值
                            if len(image.shape) == 3:
                                nodata_mask = np.all(image == 0, axis=2)
                            else:
                                nodata_mask = (image == 0)
                except:
                    # 如果无法读取元数据，回退到检查0值
                    if len(image.shape) == 3:
                        nodata_mask = np.all(image == 0, axis=2)
                    else:
                        nodata_mask = (image == 0)
                
                # 处理多通道图像
                if len(image.shape) == 3 and image.shape[2] > 3:
                    # 取前3个通道
                    image = image[:, :, :3]
                elif len(image.shape) == 2:
                    # 灰度图转RGB
                    image = np.stack([image] * 3, axis=-1)
                
                # 数据类型转换
                if image.dtype != np.uint8:
                    # 保存原始的nodata区域
                    original_nodata = nodata_mask.copy() if nodata_mask is not None else None
                    
                    # 对非nodata区域进行归一化
                    if nodata_mask is not None:
                        valid_pixels = image[~nodata_mask]
                        if len(valid_pixels) > 0:
                            min_val, max_val = valid_pixels.min(), valid_pixels.max()
                            if max_val > min_val:
                                image = ((image - min_val) / (max_val - min_val + 1e-8) * 255).astype(np.uint8)
                            else:
                                image = image.astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)
                    else:
                        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
                    
                    # 恢复nodata区域为0
                    if original_nodata is not None:
                        if len(image.shape) == 3:
                            image[original_nodata] = [0, 0, 0]
                        else:
                            image[original_nodata] = 0
                        nodata_mask = original_nodata
            else:
                # 使用PIL处理其他格式
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
                # 对于非TIFF文件，检查黑色像素作为nodata
                nodata_mask = np.all(image == 0, axis=2)
            
            print(f"图像加载成功，尺寸: {image.shape}")
            if nodata_mask is not None:
                nodata_count = np.sum(nodata_mask)
                total_pixels = nodata_mask.size
                print(f"检测到 {nodata_count} 个nodata像素 ({nodata_count/total_pixels*100:.2f}%)")
            
            # 保存nodata掩码到实例变量
            self.nodata_mask = nodata_mask
            
            return image, nodata_mask
            
        except Exception as e:
            print(f"图像加载失败: {e}")
            return None, None
    
    def create_tiles(self, image, save_tiles=True):
        """
        将大图切分为小图块
        
        Args:
            image: 输入图像数组
            save_tiles: 是否保存切片
            
        Returns:
            list: 切片信息列表
        """
        print("开始切分图像...")
        
        height, width = image.shape[:2]
        tile_height, tile_width = self.tile_size
        
        tiles_info = []
        tile_count = 0
        
        # 计算切片数量
        rows = (height - self.overlap) // (tile_height - self.overlap)
        cols = (width - self.overlap) // (tile_width - self.overlap)
        
        print(f"将生成 {rows} x {cols} = {rows * cols} 个切片")
        
        for row in tqdm(range(rows), desc="切分行"):
            for col in range(cols):
                # 计算切片位置
                y_start = row * (tile_height - self.overlap)
                x_start = col * (tile_width - self.overlap)
                y_end = min(y_start + tile_height, height)
                x_end = min(x_start + tile_width, width)
                
                # 提取切片
                tile = image[y_start:y_end, x_start:x_end]
                
                # 调整切片大小到标准尺寸
                if tile.shape[:2] != self.tile_size:
                    tile = cv2.resize(tile, self.tile_size)
                
                # 创建切片信息（包括有效和无效区域）
                tile_filename = f"tile_{row:04d}_{col:04d}.tif"
                # 按需创建tiles目录
                tiles_dir = os.path.join(self.output_dir, 'tiles', 'all')
                tile_path = os.path.join(tiles_dir, tile_filename)
                
                # 提取对应的nodata掩码
                tile_nodata_mask = None
                if hasattr(self, 'nodata_mask') and self.nodata_mask is not None:
                    tile_nodata_mask = self.nodata_mask[y_start:y_end, x_start:x_end]
                    # 调整掩码大小到标准尺寸
                    if tile_nodata_mask.shape != self.tile_size:
                        tile_nodata_mask = cv2.resize(tile_nodata_mask.astype(np.uint8), self.tile_size, interpolation=cv2.INTER_NEAREST).astype(bool)
                
                # 检查是否为有效区域
                is_valid = self.is_valid_tile(tile, tile_nodata_mask)
                
                # 只保存有效切片，无效切片不保存也不参与后续预测
                if save_tiles and is_valid:
                    # 按需创建tiles目录
                    os.makedirs(tiles_dir, exist_ok=True)
                    # 保存为TIFF格式以保持多通道数据完整性
                    if tifffile is not None:
                        # 使用tifffile保存，支持多通道
                        tifffile.imwrite(tile_path, tile)
                    else:
                        # 回退到cv2，但只能保存3通道
                        if len(tile.shape) == 3 and tile.shape[2] > 3:
                            # 如果是多通道，只保存前3个通道
                            tile_rgb = tile[:, :, :3]
                            cv2.imwrite(tile_path, cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))
                        else:
                            cv2.imwrite(tile_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                elif not is_valid:
                    # 无效切片不保存文件，设置路径为None
                    tile_path = None
                
                tiles_info.append({
                    'filename': tile_filename,
                    'path': tile_path,
                    'row': row,
                    'col': col,
                    'x_start': x_start,
                    'y_start': y_start,
                    'x_end': x_end,
                    'y_end': y_end,
                    'tile_data': tile,
                    'is_valid': is_valid,
                    'nodata_mask': tile_nodata_mask
                })
                
                if is_valid:
                    tile_count += 1
        
        print(f"切片完成，生成 {tile_count} 个有效切片")
        return tiles_info
    
    def is_valid_tile(self, tile, tile_nodata_mask=None):
        """
        检查切片是否为有效区域（基于nodata掩码）
        
        Args:
            tile: 切片图像
            tile_nodata_mask: 切片对应的nodata掩码
            
        Returns:
            bool: 是否有效
        """
        if tile_nodata_mask is None:
            # 如果没有nodata掩码，检查是否为全黑区域
            return not np.all(tile == 0)
        
        # 如果nodata像素占切片的80%以上，认为是无效切片
        nodata_ratio = np.sum(tile_nodata_mask) / tile_nodata_mask.size
        return nodata_ratio < 0.8
    
    def classify_health_disease(self, tiles_info):
        """
        使用image_classifier进行健康/疾病分类
        
        Args:
            tiles_info: 切片信息列表
            
        Returns:
            dict: 分类结果
        """
        print("开始健康/疾病分类...")
        
        # 设置分类器输出目录（直接使用主输出目录，不创建health_classifier子目录）
        classifier_output_dir = self.health_classifier_config.get('output_dir')
        if classifier_output_dir is None:
            classifier_output_dir = self.output_dir  # 直接使用主输出目录
        
        # 初始化分类器（使用lazy_init避免创建不必要的目录）
        classifier = ImageClassifier(output_dir=classifier_output_dir, lazy_init=True)
        
        # 检查是否有预训练模型
        model_files = self.health_classifier_config.get('model_search_paths', [])
        
        model_loaded = False
        for model_pattern in model_files:
            if '*' in model_pattern:
                import glob
                model_paths = glob.glob(model_pattern)
                if model_paths:
                    model_path = sorted(model_paths)[-1]  # 使用最新的模型
                    try:
                        classifier.load_model(model_path)
                        print(f"加载预训练模型: {model_path}")
                        model_loaded = True
                        break
                    except:
                        continue
            else:
                if os.path.exists(model_pattern):
                    try:
                        classifier.load_model(model_pattern)
                        print(f"加载预训练模型: {model_pattern}")
                        model_loaded = True
                        break
                    except:
                        continue
        
        if not model_loaded:
            print("未找到预训练模型，需要先训练image_classifier")
            print("请先运行: python image_classifier.py")
            return None
        
        # 检查模型状态
        if classifier.model is None:
            print("模型加载失败，classifier.model为None")
            return None
        
        print(f"模型加载成功，设备: {classifier.device}")
        print(f"类别名称: {classifier.class_names}")
        
        # 分类结果
        classification_results = {
            'healthy': [],
            'diseased': [],
            'statistics': {'healthy_count': 0, 'diseased_count': 0}
        }
        
        # 对每个切片进行分类
        for tile_info in tqdm(tiles_info, desc="分类切片"):
            # 跳过无效切片（路径为None或is_valid为False）
            if not tile_info.get('is_valid', True) or tile_info.get('path') is None:
                tile_info['health_prediction'] = 'invalid'
                tile_info['health_confidence'] = 0.0
                continue
                
            try:
                # 预测
                prediction_result = classifier.predict_single_image(tile_info['path'])
                
                if prediction_result is None:
                    print(f"分类切片 {tile_info['filename']} 时出错: predict_single_image返回None")
                    tile_info['health_prediction'] = 'invalid'
                    tile_info['health_confidence'] = 0.0
                    continue
                
                result, confidence = prediction_result
                
                # 更新切片信息
                tile_info['health_prediction'] = result
                tile_info['health_confidence'] = confidence
                
                # 复制文件到对应文件夹
                if result == 'healthy':
                    classified_dir = os.path.join(self.output_dir, 'classified', 'healthy')
                    os.makedirs(classified_dir, exist_ok=True)  # 按需创建目录
                    dest_path = os.path.join(classified_dir, tile_info['filename'])
                    classification_results['healthy'].append(tile_info)
                    classification_results['statistics']['healthy_count'] += 1
                else:
                    classified_dir = os.path.join(self.output_dir, 'classified', 'diseased')
                    os.makedirs(classified_dir, exist_ok=True)  # 按需创建目录
                    dest_path = os.path.join(classified_dir, tile_info['filename'])
                    classification_results['diseased'].append(tile_info)
                    classification_results['statistics']['diseased_count'] += 1
                
                shutil.copy2(tile_info['path'], dest_path)
                
            except Exception as e:
                print(f"分类切片 {tile_info['filename']} 时出错: {e}")
                tile_info['health_prediction'] = 'invalid'
                tile_info['health_confidence'] = 0.0
                continue
        
        print(f"分类完成: 健康 {classification_results['statistics']['healthy_count']} 个, "
              f"疾病 {classification_results['statistics']['diseased_count']} 个")
        
        return classification_results
    
    def grade_severity(self, diseased_tiles):
        """
        使用基于对比学习的无监督方法对疾病切片进行严重程度分级
        
        Args:
            diseased_tiles: 疾病切片列表
            
        Returns:
            dict: 分级结果
        """
        if not diseased_tiles:
            print("没有疾病切片需要分级")
            return {'graded_tiles': [], 'statistics': {}}
        
        print(f"开始使用对比学习无监督方法对 {len(diseased_tiles)} 个疾病切片进行严重程度分级...")
        
        # 创建临时目录用于存储疾病切片
        temp_diseased_dir = os.path.join(self.output_dir, 'temp_diseased')
        os.makedirs(temp_diseased_dir, exist_ok=True)
        
        # 复制疾病切片到临时目录
        diseased_paths = []
        for tile_info in diseased_tiles:
            if tile_info.get('is_valid', True) and tile_info.get('path') is not None:
                temp_path = os.path.join(temp_diseased_dir, tile_info['filename'])
                shutil.copy2(tile_info['path'], temp_path)
                diseased_paths.append(temp_path)
        
        if not diseased_paths:
            print("没有有效的疾病切片可以分级")
            return {'graded_tiles': [], 'statistics': {}}
        
        # 初始化严重程度分类器，使用对比学习方法（使用lazy_init避免创建不必要的目录）
        severity_classifier = SeverityClassifier(
            data_dir=self.output_dir,  # 使用当前输出目录
            n_levels=self.severity_classifier_config.get('n_levels', 6),
            img_size=self.severity_classifier_config.get('img_size', (64, 64)),
            output_dir=self.output_dir,  # 指定输出目录，避免创建新的severity_run_目录
            lazy_init=True  # 延迟初始化，避免创建不必要的目录
        )
        
        print("============================================================")
        print("使用基于对比学习的无监督严重程度分级")
        print(f"严重程度级别数: {self.severity_classifier_config.get('n_levels', 6)}")
        print(f"图像尺寸: {self.severity_classifier_config.get('img_size', (64, 64))}")
        print("============================================================")
        
        # 检查是否有预训练的对比学习模型
        contrastive_model_files = self.severity_classifier_config.get('model_search_paths', [])
        
        model_loaded = False
        for model_pattern in contrastive_model_files:
            if '*' in model_pattern:
                import glob
                model_paths = glob.glob(model_pattern)
                if model_paths:
                    model_path = sorted(model_paths)[-1]  # 使用最新的模型
                    try:
                        # 创建模型
                        severity_classifier.create_model()
                        # 加载对比学习模型
                        checkpoint = torch.load(model_path, map_location=severity_classifier.device)
                        
                        # 检查checkpoint格式并正确加载
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            # 完整的checkpoint格式
                            severity_classifier.model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            # 直接的state_dict格式
                            severity_classifier.model.load_state_dict(checkpoint)
                        
                        print(f"加载预训练对比学习模型: {model_path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"加载对比学习模型 {model_path} 失败: {e}")
                        continue
            else:
                if os.path.exists(model_pattern):
                    try:
                        # 创建模型
                        severity_classifier.create_model()
                        # 加载对比学习模型
                        checkpoint = torch.load(model_pattern, map_location=severity_classifier.device)
                        
                        # 检查checkpoint格式并正确加载
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            # 完整的checkpoint格式
                            severity_classifier.model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            # 直接的state_dict格式
                            severity_classifier.model.load_state_dict(checkpoint)
                        
                        print(f"加载预训练对比学习模型: {model_pattern}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"加载对比学习模型 {model_pattern} 失败: {e}")
                        continue
        
        if not model_loaded:
            print("未找到预训练对比学习模型，将使用实时训练...")
            print("注意：这将需要较长时间进行对比学习训练")
            
            # 运行完整的对比学习训练和聚类流程
            try:
                # 使用配置的训练参数
                training_config = self.severity_classifier_config.get('training_config', {})
                epochs = training_config.get('epochs', 50)
                learning_rate = training_config.get('learning_rate', 0.001)
                
                print(f"开始对比学习训练 - Epochs: {epochs}, Learning Rate: {learning_rate}")
                severity_classifier.run_contrastive_severity_classification(
                    epochs=epochs,
                    learning_rate=learning_rate
                )
                model_loaded = True
            except Exception as e:
                print(f"对比学习训练失败: {e}")
                return None
        
        # 初始化分级结果（在try块外部，确保在所有分支中都可访问）
        grading_results = {
            'graded_tiles': [],
            'statistics': {f'Level_{i}': 0 for i in range(1, 7)}
        }
        
        if model_loaded:
            print("使用预训练模型进行特征提取和聚类分析...")
            
            try:
                # 提取特征
                features = severity_classifier.extract_features(diseased_paths)
                
                # 进行聚类分级
                severity_labels, level_counts = severity_classifier.cluster_severity_levels(features, diseased_paths)
                
                # 解析聚类结果
                level_assignments = {}
                for i, img_path in enumerate(diseased_paths):
                    filename = os.path.basename(img_path)
                    # 聚类标签从0开始，转换为1-6的等级
                    level_assignments[filename] = severity_labels[i] + 1
        
                # 对每个疾病切片进行分级
                for tile_info in tqdm(diseased_tiles, desc="分级切片"):
                    # 跳过无效切片（路径为None或is_valid为False）
                    if not tile_info.get('is_valid', True) or tile_info.get('path') is None:
                        # 为无效切片设置默认分级信息
                        tile_info['severity_level'] = 1  # 默认为最低级
                        tile_info['severity_confidence'] = 0.0
                        tile_info['final_level'] = 1
                        continue
                        
                    try:
                        # 从聚类结果中获取分级
                        filename = tile_info['filename']
                        if filename in level_assignments:
                            severity_level = level_assignments[filename]
                        else:
                            severity_level = 1  # 默认级别
                        
                        # 更新切片信息
                        tile_info['severity_level'] = severity_level
                        tile_info['severity_confidence'] = 0.8  # 聚类方法的默认置信度
                        tile_info['final_level'] = severity_level  # 最终等级（1-6）
                        
                        grading_results['graded_tiles'].append(tile_info)
                        grading_results['statistics'][f'Level_{severity_level}'] += 1
                        
                        # 复制文件到分级文件夹
                        level_dir = os.path.join(self.output_dir, 'severity_graded', f'Level_{severity_level}')
                        os.makedirs(level_dir, exist_ok=True)
                        dest_path = os.path.join(level_dir, tile_info['filename'])
                        shutil.copy2(tile_info['path'], dest_path)
                        
                    except Exception as e:
                        print(f"分级切片 {tile_info['filename']} 时出错: {e}")
                        continue
                        
            except Exception as e:
                print(f"特征提取或聚类过程中出错: {e}")
                # 回退到默认分级
                for tile_info in diseased_tiles:
                    if tile_info.get('is_valid', True) and tile_info.get('path') is not None:
                        tile_info['severity_level'] = 1
                        tile_info['severity_confidence'] = 0.5
                        tile_info['final_level'] = 1
                        grading_results['graded_tiles'].append(tile_info)
                        grading_results['statistics']['Level_1'] += 1
        
        # 清理临时目录
        try:
            shutil.rmtree(temp_diseased_dir)
        except:
            pass
        
        print("基于对比学习的严重程度分级完成:")
        for level, count in grading_results['statistics'].items():
            print(f"  {level}: {count} 个切片")
        
        return grading_results
    
    def create_result_visualization(self, original_image, all_tiles_info):
        """
        创建结果可视化图像
        
        Args:
            original_image: 原始大图
            all_tiles_info: 所有切片信息
        """
        print("创建结果可视化...")
        
        height, width = original_image.shape[:2]
        
        # 创建结果图像（RGB），初始化为白色（无效区域）
        result_image = np.full((height, width, 3), 255, dtype=np.uint8)
        
        # 创建等级统计
        level_stats = {i: 0 for i in range(7)}  # 0-6级
        
        # 填充每个切片的颜色
        for tile_info in all_tiles_info:
            # 确定最终等级
            if (tile_info.get('health_prediction') == 'invalid' or 
                not tile_info.get('is_valid', True) or 
                tile_info.get('path') is None):
                # 无效区域保持白色，不填充颜色
                tile_info['final_level'] = -1  # 标记为无效
                continue
            elif tile_info.get('health_prediction') == 'healthy':
                final_level = 0  # 健康
            else:
                final_level = tile_info.get('severity_level', 1)  # 疾病等级1-6
            
            tile_info['final_level'] = final_level
            level_stats[final_level] += 1
            
            # 获取对应颜色
            color = self.color_map[final_level]
            
            # 填充区域
            y_start, y_end = tile_info['y_start'], tile_info['y_end']
            x_start, x_end = tile_info['x_start'], tile_info['x_end']
            
            result_image[y_start:y_end, x_start:x_end] = color
        
        # 保存结果图像
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)  # 按需创建目录
        result_path = os.path.join(results_dir, 'classification_result.png')
        cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        # 创建带图例的可视化
        self.create_visualization_with_legend(result_image, level_stats)
        
        print(f"结果可视化已保存: {result_path}")
        
        return result_image, level_stats
    
    def create_visualization_with_legend(self, result_image, level_stats):
        """
        创建带图例的可视化图像
        
        Args:
            result_image: 结果图像
            level_stats: 等级统计
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 显示结果图像
        ax1.imshow(result_image)
        ax1.set_title('Disease Severity Classification Results', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 创建图例
        legend_labels = {
            0: 'Healthy',
            1: 'Disease Level 1',
            2: 'Disease Level 2',
            3: 'Disease Level 3',
            4: 'Disease Level 4',
            5: 'Disease Level 5',
            6: 'Disease Level 6'
        }
        
        # 在图像上添加图例
        legend_elements = []
        for level, count in level_stats.items():
            if count > 0:
                color = np.array(self.color_map[level]) / 255.0
                label = f"{legend_labels[level]}: {count} tiles"
                legend_elements.append(patches.Patch(color=color, label=label))
        
        ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # 创建统计柱状图
        levels = list(level_stats.keys())
        counts = list(level_stats.values())
        colors = [np.array(self.color_map[level]) / 255.0 for level in levels]
        
        bars = ax2.bar(levels, counts, color=colors)
        ax2.set_xlabel('Disease Level', fontsize=12)
        ax2.set_ylabel('Number of Tiles', fontsize=12)
        ax2.set_title('Tile Count Statistics by Level', fontsize=14, fontweight='bold')
        ax2.set_xticks(levels)
        ax2.set_xticklabels([legend_labels[level] for level in levels], rotation=45, ha='right')
        
        # 在柱状图上添加数值标签
        for bar, count in zip(bars, counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存可视化图像
        viz_dir = os.path.join(self.output_dir, 'visualization')
        os.makedirs(viz_dir, exist_ok=True)  # 按需创建目录
        viz_path = os.path.join(viz_dir, 'result_with_legend.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"带图例的可视化已保存: {viz_path}")
    
    def save_results_summary(self, all_tiles_info, level_stats, processing_time):
        """
        保存处理结果摘要
        
        Args:
            all_tiles_info: 所有切片信息
            level_stats: 等级统计
            processing_time: 处理时间
        """
        summary = {
            'timestamp': self.timestamp,
            'processing_time_seconds': processing_time,
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'total_tiles': len(all_tiles_info),
            'level_statistics': level_stats,
            'level_percentages': {},
            'color_mapping': self.color_map,
            'output_directory': self.output_dir
        }
        
        # 计算百分比
        total_tiles = len(all_tiles_info)
        for level, count in level_stats.items():
            percentage = (count / total_tiles * 100) if total_tiles > 0 else 0
            summary['level_percentages'][level] = round(percentage, 2)
        
        # 保存摘要
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)  # 按需创建目录
        summary_path = os.path.join(results_dir, 'processing_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"处理摘要已保存: {summary_path}")
        
        # 打印摘要
        print("\n=== 处理结果摘要 ===")
        print(f"总切片数: {total_tiles}")
        print(f"处理时间: {processing_time:.2f} 秒")
        print("\n各等级分布:")
        for level in range(7):
            count = level_stats[level]
            percentage = summary['level_percentages'][level]
            level_name = "健康" if level == 0 else f"疾病等级{level}"
            print(f"  {level_name}: {count} 个 ({percentage}%)")
    
    def run_full_pipeline(self, image_path):
        """
        运行完整的处理流水线
        
        Args:
            image_path: 输入大图路径
        """
        start_time = datetime.now()
        print(f"开始运行完整处理流水线: {start_time}")
        print(f"输入图像: {image_path}")
        
        try:
            # 1. 加载大图
            result = self.load_large_image(image_path)
            if result is None or result[0] is None:
                print("图像加载失败，流水线终止")
                return
            
            original_image, nodata_mask = result
            self.nodata_mask = nodata_mask  # 保存nodata掩码到实例属性
            print(f"图像加载完成，nodata掩码: {'已检测' if nodata_mask is not None else '未检测'}")
            
            # 2. 切分图像
            tiles_info = self.create_tiles(original_image)
            if not tiles_info:
                print("图像切分失败，流水线终止")
                return
            
            # 统计有效和无效切片数量
            valid_count = sum(1 for tile in tiles_info if tile.get('is_valid', True))
            invalid_count = len(tiles_info) - valid_count
            print(f"切片统计: 总计 {len(tiles_info)} 个，有效 {valid_count} 个，无效 {invalid_count} 个")
            
            # 3. 健康/疾病分类
            classification_results = self.classify_health_disease(tiles_info)
            if classification_results is None:
                print("健康/疾病分类失败，流水线终止")
                return
            
            # 4. 疾病严重程度分级
            grading_results = self.grade_severity(classification_results['diseased'])
            if grading_results is None:
                print("严重程度分级失败，将使用默认分级继续流水线")
                # 为所有疾病切片分配默认严重程度级别1
                grading_results = {
                    'graded_tiles': [],
                    'statistics': {f'Level_{i}': 0 for i in range(1, 7)}
                }
                for tile_info in classification_results['diseased']:
                    tile_info['severity_level'] = 1  # 默认严重程度级别
                    tile_info['severity_confidence'] = 0.5  # 默认置信度
                    tile_info['final_level'] = 1
                    grading_results['graded_tiles'].append(tile_info)
                    grading_results['statistics']['Level_1'] += 1
            
            # 5. 更新所有切片信息（包括无效切片）
            all_tiles_info = tiles_info  # 使用原始的所有切片信息
            
            # 6. 创建结果可视化
            result_image, level_stats = self.create_result_visualization(original_image, all_tiles_info)
            
            # 7. 保存结果摘要
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            self.save_results_summary(all_tiles_info, level_stats, processing_time)
            
            print(f"\n流水线处理完成! 总耗时: {processing_time:.2f} 秒")
            print(f"结果保存在: {self.output_dir}")
            
            return {
                'success': True,
                'output_dir': self.output_dir,
                'tiles_info': all_tiles_info,
                'level_stats': level_stats,
                'processing_time': processing_time
            }
            
        except Exception as e:
            print(f"流水线处理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

def main(image_path=None, config_template='default', config_file=None, **config_kwargs):
    """
    主函数 - 示例用法
    
    Args:
        image_path: 输入图像路径，如果为None则使用默认路径
        config_template: 配置模板名称 ('default', 'high_precision', 'fast_processing', 'medium', 'custom_color')
        config_file: 外部配置文件路径（暂未实现）
        **config_kwargs: 额外的配置参数，会覆盖模板中的对应参数
    """
    print("=== 完整图像处理流水线 ===")
    print("功能: 大图切片 → 健康/疾病分类 → 疾病严重程度分级 → 重组可视化")
    print()
    
    # 处理输入图像路径
    if image_path is None:
        # 使用配置中的默认路径
        default_config = PipelineConfig()
        for path in default_config.default_image_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if image_path is None:
            print("未找到默认图像文件，请提供图像路径:")
            print("可用的默认路径:", default_config.default_image_paths)
            return
    
    print(f"使用输入图像: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return
    
    # 创建配置
    try:
        if config_file:
            # TODO: 实现从文件加载配置
            print(f"从配置文件加载: {config_file}")
            config_dict = create_pipeline_config(config_template, **config_kwargs)
        else:
            # 使用模板配置
            config_dict = create_pipeline_config(config_template, **config_kwargs)
        
        print(f"使用配置模板: {config_template}")
        print(f"切片大小: {config_dict['tile_size']}")
        print(f"严重程度级别数: {config_dict['severity_classifier_config']['n_levels']}")
        
    except Exception as e:
        print(f"配置创建失败: {e}")
        return
    
    # 创建流水线实例
    try:
        pipeline = FullPipeline(config=config_dict)
    except Exception as e:
        print(f"流水线初始化失败: {e}")
        return
    
    # 运行完整流水线
    result = pipeline.run_full_pipeline(image_path)
    
    if result and result.get('success', False):
        print("\n流水线执行成功!")
        print(f"查看结果: {result['output_dir']}")
    else:
        print("\n流水线执行失败!")
        if result:
            print(f"错误信息: {result.get('error', '未知错误')}")
        else:
            print("流水线返回None")
    
    return result

if __name__ == "__main__":
    # 示例1: 使用默认配置
    print("\n=== 示例1: 使用默认配置 ===")
    main()
    
    # 示例2: 使用高精度配置模板
    print("\n=== 示例2: 使用高精度配置模板 ===")
    # main(config_template='high_precision')
    
    # 示例3: 使用快速处理配置模板
    print("\n=== 示例3: 使用快速处理配置模板 ===")
    # main(config_template='fast_processing')
    
    # 示例4: 使用模板+自定义参数
    print("\n=== 示例4: 使用模板+自定义参数 ===")
    # main(config_template='medium', tile_size=(256, 256), overlap=32)
    
    # 示例5: 直接使用配置对象
    print("\n=== 示例5: 直接使用配置对象 ===")
    # custom_config = ConfigTemplates.get_high_precision_config()
    # pipeline = FullPipeline(config=custom_config)
    # result = pipeline.run_full_pipeline("your_image.tif")
    
    print("\n=== 外部配置文件流水线使用说明 ===")
    print("1. 默认配置: main() - 使用默认配置模板")
    print("2. 指定模板: main(config_template='high_precision') - 使用预定义模板")
    print("3. 模板+参数: main(config_template='default', tile_size=(128,128)) - 模板基础上自定义")
    print("4. 配置对象: pipeline = FullPipeline(config=ConfigTemplates.get_xxx_config())")
    print("\n可用配置模板:")
    print("- 'default': 默认配置 (64x64切片, 6级别)")
    print("- 'high_precision': 高精度配置 (512x512切片, 12级别)")
    print("- 'fast_processing': 快速处理配置 (32x32切片, 3级别)")
    print("- 'medium': 中等配置 (128x128切片, 8级别)")
    print("- 'custom_color': 自定义颜色配置 (5级别)")
    print("\n配置文件位置: pipeline_config.py")
    print("可以直接修改 pipeline_config.py 来调整默认参数和添加新模板")