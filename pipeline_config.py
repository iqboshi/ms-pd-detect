#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流水线配置文件
包含所有可配置的参数和默认值
"""

import os
from datetime import datetime

class PipelineConfig:
    """
    流水线配置类
    """
    
    def __init__(self):
        """初始化默认配置"""
        # 基本参数
        self.tile_size = (64, 64)
        self.overlap = 0
        self.output_dir = None  # None表示自动生成
        
        # 健康/疾病分类器配置
        self.health_classifier_config = {
            'output_dir': None,  # 将在运行时动态设置
            'model_search_paths': [
                'outputs/image_run_*/models/best_model.pth',
                'outputs/image_run_*/models/disease_classifier_model.pth',
                'disease_classifier_model.pth',
                'models/disease_classifier_model.pth'
            ]
        }
        
        # 严重程度分类器配置
        self.severity_classifier_config = {
            'n_levels': 6,
            'img_size': None,  # 将使用tile_size
            'model_search_paths': [
                'outputs/severity_run_*/models/final_contrastive_model.pth',
                'outputs/severity_run_*/models/contrastive_model_epoch_*.pth',
                'models/final_contrastive_model.pth'
            ],
            'training_config': {
                'epochs': 50,
                'learning_rate': 0.001
            }
        }
        
        # 颜色映射（7级：0=健康，1-6=疾病严重程度）
        self.color_map = {
            0: [0, 255, 0],      # 绿色 - 健康
            1: [255, 255, 0],    # 黄色 - Level_1
            2: [255, 200, 0],    # 橙黄色 - Level_2
            3: [255, 150, 0],    # 橙色 - Level_3
            4: [255, 100, 0],    # 深橙色 - Level_4
            5: [255, 50, 0],     # 红橙色 - Level_5
            6: [255, 0, 0]       # 红色 - Level_6
        }
        
        # 默认图像路径
        self.default_image_paths = [
            "水稻所地块裁剪15m.tif",
            "test_image.tif",
            "input.tif"
        ]
    
    def update_img_size(self):
        """更新img_size为tile_size"""
        if self.severity_classifier_config['img_size'] is None:
            self.severity_classifier_config['img_size'] = self.tile_size
    
    def get_output_dir(self):
        """获取输出目录，如果为None则生成时间戳目录"""
        if self.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"outputs/full_pipeline_{timestamp}"
        return self.output_dir
    
    def to_dict(self):
        """转换为字典格式"""
        self.update_img_size()
        return {
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'output_dir': self.get_output_dir(),
            'health_classifier_config': self.health_classifier_config,
            'severity_classifier_config': self.severity_classifier_config,
            'color_map': self.color_map
        }
    
    def load_from_dict(self, config_dict):
        """从字典加载配置"""
        self.tile_size = config_dict.get('tile_size', self.tile_size)
        self.overlap = config_dict.get('overlap', self.overlap)
        self.output_dir = config_dict.get('output_dir', self.output_dir)
        
        if 'health_classifier_config' in config_dict:
            self.health_classifier_config.update(config_dict['health_classifier_config'])
        
        if 'severity_classifier_config' in config_dict:
            self.severity_classifier_config.update(config_dict['severity_classifier_config'])
        
        if 'color_map' in config_dict:
            self.color_map.update(config_dict['color_map'])
        
        self.update_img_size()

# 预定义配置模板
class ConfigTemplates:
    """预定义的配置模板"""
    
    @staticmethod
    def get_default_config():
        """获取默认配置"""
        return PipelineConfig()
    
    @staticmethod
    def get_high_precision_config():
        """高精度配置 - 大切片，更多级别，更多训练轮数"""
        config = PipelineConfig()
        config.tile_size = (512, 512)
        config.overlap = 64
        config.severity_classifier_config['n_levels'] = 12
        config.severity_classifier_config['training_config']['epochs'] = 300
        config.severity_classifier_config['training_config']['learning_rate'] = 0.0001
        return config
    
    @staticmethod
    def get_fast_processing_config():
        """快速处理配置 - 小切片，少级别，少训练轮数"""
        config = PipelineConfig()
        config.tile_size = (32, 32)
        config.overlap = 0
        config.severity_classifier_config['n_levels'] = 3
        config.severity_classifier_config['training_config']['epochs'] = 20
        config.severity_classifier_config['training_config']['learning_rate'] = 0.01
        return config
    
    @staticmethod
    def get_medium_config():
        """中等配置 - 平衡精度和速度"""
        config = PipelineConfig()
        config.tile_size = (128, 128)
        config.overlap = 16
        config.severity_classifier_config['n_levels'] = 8
        config.severity_classifier_config['training_config']['epochs'] = 100
        config.severity_classifier_config['training_config']['learning_rate'] = 0.0005
        return config
    
    @staticmethod
    def get_custom_color_config():
        """自定义颜色配置"""
        config = PipelineConfig()
        config.color_map = {
            0: [0, 255, 0],       # 健康 - 绿色
            1: [255, 255, 0],     # 轻微 - 黄色
            2: [255, 128, 0],     # 中等 - 橙色
            3: [255, 0, 0],       # 严重 - 红色
            4: [128, 0, 128],     # 极严重 - 紫色
            5: [64, 0, 64],       # 最严重 - 深紫色
        }
        config.severity_classifier_config['n_levels'] = 5
        return config

def create_pipeline_config(template='default', **kwargs):
    """
    创建流水线配置的便捷函数
    
    Args:
        template: 配置模板名称 ('default', 'high_precision', 'fast_processing', 'medium', 'custom_color')
        **kwargs: 额外的配置参数，会覆盖模板中的对应参数
    
    Returns:
        dict: 配置字典
    """
    # 获取模板配置
    if template == 'default':
        config = ConfigTemplates.get_default_config()
    elif template == 'high_precision':
        config = ConfigTemplates.get_high_precision_config()
    elif template == 'fast_processing':
        config = ConfigTemplates.get_fast_processing_config()
    elif template == 'medium':
        config = ConfigTemplates.get_medium_config()
    elif template == 'custom_color':
        config = ConfigTemplates.get_custom_color_config()
    else:
        raise ValueError(f"未知的配置模板: {template}")
    
    # 应用自定义参数
    if kwargs:
        config.load_from_dict(kwargs)
    
    return config.to_dict()

# 配置验证函数
def validate_config(config_dict):
    """
    验证配置参数的合理性
    
    Args:
        config_dict: 配置字典
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # 验证tile_size
    tile_size = config_dict.get('tile_size', (64, 64))
    if not isinstance(tile_size, (tuple, list)) or len(tile_size) != 2:
        errors.append("tile_size必须是包含两个元素的元组或列表")
    elif tile_size[0] <= 0 or tile_size[1] <= 0:
        errors.append("tile_size的值必须大于0")
    elif tile_size[0] > 1024 or tile_size[1] > 1024:
        errors.append("tile_size过大，建议不超过1024")
    
    # 验证overlap
    overlap = config_dict.get('overlap', 0)
    if not isinstance(overlap, int) or overlap < 0:
        errors.append("overlap必须是非负整数")
    elif overlap >= min(tile_size):
        errors.append("overlap不能大于等于tile_size的最小值")
    
    # 验证严重程度级别数
    severity_config = config_dict.get('severity_classifier_config', {})
    n_levels = severity_config.get('n_levels', 6)
    if not isinstance(n_levels, int) or n_levels < 2:
        errors.append("n_levels必须是大于等于2的整数")
    elif n_levels > 20:
        errors.append("n_levels过大，建议不超过20")
    
    # 验证训练参数
    training_config = severity_config.get('training_config', {})
    epochs = training_config.get('epochs', 50)
    if not isinstance(epochs, int) or epochs <= 0:
        errors.append("epochs必须是正整数")
    elif epochs > 1000:
        errors.append("epochs过大，建议不超过1000")
    
    learning_rate = training_config.get('learning_rate', 0.001)
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        errors.append("learning_rate必须是正数")
    elif learning_rate > 1.0:
        errors.append("learning_rate过大，建议不超过1.0")
    
    # 验证颜色映射
    color_map = config_dict.get('color_map', {})
    if not isinstance(color_map, dict):
        errors.append("color_map必须是字典")
    else:
        for level, color in color_map.items():
            if not isinstance(level, int) or level < 0:
                errors.append(f"颜色映射的级别{level}必须是非负整数")
            if not isinstance(color, (list, tuple)) or len(color) != 3:
                errors.append(f"颜色映射的颜色值{color}必须是包含3个元素的列表或元组")
            elif any(not isinstance(c, int) or c < 0 or c > 255 for c in color):
                errors.append(f"颜色映射的颜色值{color}必须是0-255之间的整数")
    
    return len(errors) == 0, errors

if __name__ == "__main__":
    # 测试配置功能
    print("=== 测试配置功能 ===")
    
    # 测试默认配置
    print("\n1. 测试默认配置")
    default_config = create_pipeline_config()
    print(f"默认切片大小: {default_config['tile_size']}")
    print(f"默认严重程度级别数: {default_config['severity_classifier_config']['n_levels']}")
    
    # 测试高精度配置
    print("\n2. 测试高精度配置")
    high_precision_config = create_pipeline_config('high_precision')
    print(f"高精度切片大小: {high_precision_config['tile_size']}")
    print(f"高精度严重程度级别数: {high_precision_config['severity_classifier_config']['n_levels']}")
    
    # 测试自定义配置
    print("\n3. 测试自定义配置")
    custom_config = create_pipeline_config(
        'default',
        tile_size=(256, 256),
        overlap=32,
        severity_classifier_config={
            'n_levels': 10,
            'training_config': {
                'epochs': 200,
                'learning_rate': 0.0001
            }
        }
    )
    print(f"自定义切片大小: {custom_config['tile_size']}")
    print(f"自定义严重程度级别数: {custom_config['severity_classifier_config']['n_levels']}")
    
    # 测试配置验证
    print("\n4. 测试配置验证")
    is_valid, errors = validate_config(custom_config)
    print(f"配置有效性: {is_valid}")
    if errors:
        print(f"错误信息: {errors}")
    
    print("\n配置功能测试完成!")