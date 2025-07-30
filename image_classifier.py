import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from tqdm import tqdm
import random
import warnings
import logging
from datetime import datetime


def set_seed(seed=42):
    """设置所有随机种子以确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置环境变量以确保完全确定性
    os.environ['PYTHONHASHSEED'] = str(seed)



# 简化的数据集类
class SpectralImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 确保图像是HWC格式（Height, Width, Channels）
        if len(image.shape) == 3 and image.shape[0] == 3:
            # 如果是CHW格式，转换为HWC格式
            image = np.transpose(image, (1, 2, 0))
        
        # 确保图像值在0-1范围内
        if image.max() > 1.0:
            image = image / 255.0
        
        # 转换为uint8格式，因为PIL需要这种格式
        image = (image * 255).astype(np.uint8)
        
        # 应用数据变换
        if self.transform:
            image = self.transform(image)
        else:
            # 如果没有变换，直接转换为tensor
            image = torch.FloatTensor(image).permute(2, 0, 1)  # HWC -> CHW
        
        return image, torch.LongTensor([label])

# CNN模型定义
class CNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()
        
        # 第一个卷积块 - 增加更多特征提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        
        # 第四个卷积块
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        
        # 第五个卷积块 - 增加深度
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4)
        )
        
        # 全局平均池化层 - 减少参数数量
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

class ImageClassifier:
    def __init__(self, data_dir='data', img_size=(64, 64), output_dir=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['diseased', 'healthy']
        print(f"Using device: {self.device}")
        
        # 创建输出目录管理
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"outputs/image_run_{timestamp}"
        else:
            self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        
    def load_and_preprocess_data(self):
        """Load and preprocess remote sensing spectral data"""
        print("Loading data...")
        
        images = []
        labels = []
        
        # Load diseased spectral data
        diseased_dir = os.path.join(self.data_dir, 'diseased')
        for filename in os.listdir(diseased_dir):
            if filename.endswith('.tif'):
                img_path = os.path.join(diseased_dir, filename)
                img = self._load_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(0)  # diseased = 0
        
        # Load healthy spectral data
        healthy_dir = os.path.join(self.data_dir, 'healthy')
        for filename in os.listdir(healthy_dir):
            if filename.endswith('.tif'):
                img_path = os.path.join(healthy_dir, filename)
                img = self._load_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(1)  # healthy = 1
        
        # Check if any images were successfully loaded
        if len(images) == 0:
            print("Error: No images were successfully loaded!")
            print("Please check data directory and image file formats.")
            return None, None
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Loading completed: {len(X)} spectral images")
        print(f"Diseased: {np.sum(y == 0)} images")
        print(f"Healthy: {np.sum(y == 1)} images")
        
        return X, y
    
    def _load_image(self, img_path):
        """Load and preprocess single spectral image"""
        try:
            # Check file extension to determine loading method
            file_ext = os.path.splitext(img_path)[1].lower()
            
            if file_ext in ['.tif', '.tiff']:
                # Try using tifffile library to load multi-channel TIFF images
                try:
                    import tifffile
                    import warnings
                    import logging
                    
                    # Suppress all warnings and logging during TIFF reading
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Temporarily disable logging
                        logging.disable(logging.CRITICAL)
                        try:
                            img_array = tifffile.imread(img_path)
                        finally:
                            # Re-enable logging
                            logging.disable(logging.NOTSET)
                    
                    # Handle multi-channel spectral data - use more channels for better classification
                    if len(img_array.shape) == 3 and img_array.shape[2] > 3:
                        # For spectral data, use more meaningful channel combinations
                        # Take channels that might represent different spectral bands
                        if img_array.shape[2] >= 6:
                            # Use channels with better spacing for spectral analysis
                            selected_channels = [0, img_array.shape[2]//3, img_array.shape[2]*2//3]
                            img_array = img_array[:, :, selected_channels]
                        else:
                            # If less than 6 channels, take first 3
                            img_array = img_array[:, :, :3]
                    elif len(img_array.shape) == 2:
                        # If grayscale, convert to RGB
                        img_array = np.stack([img_array] * 3, axis=2)
                    elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                        # If single channel with third dimension, convert to RGB
                        img_array = np.repeat(img_array, 3, axis=2)
                    
                    # Improved normalization for spectral data
                    if img_array.dtype != np.uint8:
                        # Use percentile-based normalization to handle outliers
                        p2, p98 = np.percentile(img_array, (2, 98))
                        img_array = np.clip(img_array, p2, p98)
                        img_array = ((img_array - p2) / (p98 - p2) * 255).astype(np.uint8)
                    
                    # Convert to PIL image for resizing
                    img = Image.fromarray(img_array)
                    img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array and normalize to 0-1
                    img_array = np.array(img) / 255.0
                    
                    return img_array
                    
                except ImportError:
                    # If tifffile library not available, fall through to PIL
                    pass
            
            # Use PIL to load all other image formats (PNG, JPG, etc.) and TIFF as fallback
            with Image.open(img_path) as img:
                # If multi-page TIFF, take only first page
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    img.seek(0)
                
                # Convert to RGB (if grayscale or other modes)
                if img.mode not in ['RGB', 'L']:
                    # For special modes, first convert to L (grayscale), then to RGB
                    img = img.convert('L')
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array and normalize
                img_array = np.array(img) / 255.0
                
                return img_array
                
        except Exception as e:
            # If all methods fail, print error message (but only first few)
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            if self._error_count < 5:
                print(f"Failed to load image {img_path}: {e}")
                self._error_count += 1
            elif self._error_count == 5:
                print("...more image loading failures (omitted)")
                self._error_count += 1
            
            return None
    
    def create_model(self):
        """Create CNN model for spectral data classification"""
        model = CNNModel(num_classes=2)
        model = model.to(self.device)
        
        self.model = model
        return model
    
    def _save_training_log(self, history, best_val_acc, epochs, learning_rate, batch_size):
        """保存训练日志到文件"""
        import json
        
        log_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'training_parameters': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'image_size': self.img_size,
                'device': str(self.device)
            },
            'best_validation_accuracy': best_val_acc,
            'final_training_accuracy': history['train_acc'][-1] if history['train_acc'] else 0,
            'final_validation_accuracy': history['val_acc'][-1] if history['val_acc'] else 0,
            'training_history': history
        }
        
        log_file = os.path.join(self.output_dir, 'logs', 'training_log.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"Training log saved to: {log_file}")
    

    
    def train_model(self, X, y, test_size=0.2, epochs=30, batch_size=32, learning_rate=0.001):
        """Train the spectral classification model"""
        # 设置随机种子确保训练可重复
        set_seed(42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} spectral images")
        print(f"Test set: {len(X_test)} spectral images")
        
        # 减少数据增强的随机性以提高训练稳定性
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),  # 减少旋转角度
            transforms.RandomHorizontalFlip(p=0.3),  # 降低翻转概率
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 减少颜色抖动
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets and data loaders with deterministic settings
        train_dataset = SpectralImageDataset(X_train, y_train, transform=train_transform)
        test_dataset = SpectralImageDataset(X_test, y_test, transform=test_transform)
        
        # 使用固定的generator确保DataLoader的确定性
        generator = torch.Generator()
        generator.manual_seed(42)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=0, generator=generator, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=0, drop_last=False)
        
        # Use simple but effective class weighting
        class_counts = np.bincount(y_train)
        # Calculate balanced class weights
        class_weights = torch.FloatTensor([len(y_train)/(2*class_counts[i]) for i in range(len(class_counts))])
        
        print(f"Class weights: diseased={class_weights[0]:.3f}, healthy={class_weights[1]:.3f}")
        # Use standard CrossEntropyLoss with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        # 使用更稳定的优化器配置
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, 
                              weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        
        # 使用更稳定的学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                       factor=0.5, patience=5, 
                                                       verbose=True, min_lr=1e-7)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        print("Starting training...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.squeeze().to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for data, target in val_pbar:
                    data, target = data.to(self.device), target.squeeze().to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            # Calculate average loss and accuracy
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # Record history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling - ReduceLROnPlateau based on validation accuracy
            scheduler.step(val_acc)
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model to output directory
                best_model_path = os.path.join(self.output_dir, 'models', 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model
        best_model_path = os.path.join(self.output_dir, 'models', 'best_model.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        
        print(f"Training completed, best validation accuracy: {best_val_acc:.2f}%")
        
        # 保存训练日志
        self._save_training_log(history, best_val_acc, epochs, learning_rate, batch_size)
        
        return history, X_test, y_test
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance on spectral data"""
        self.model.eval()
        
        # Data transforms
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create test dataset
        test_dataset = SpectralImageDataset(X_test, y_test, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        y_pred = []
        y_pred_prob = []
        y_true = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating model"):
                data, target = data.to(self.device), target.squeeze().to(self.device)
                output = self.model(data)
                
                # Get probabilities and predictions
                prob = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                y_pred.extend(predicted.cpu().numpy())
                y_pred_prob.extend(prob.cpu().numpy())
                y_true.extend(target.cpu().numpy())
        
        y_pred = np.array(y_pred)
        y_pred_prob = np.array(y_pred_prob)
        y_true = np.array(y_true)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_true)
        
        print(f"\nTest accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # 保存到输出目录
        confusion_matrix_path = os.path.join(self.output_dir, 'plots', 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {confusion_matrix_path}")
        plt.show()
        
        return accuracy, y_pred, y_pred_prob
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history['train_acc'], label='Training Accuracy')
        ax1.plot(history['val_acc'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history['train_loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存到输出目录
        training_history_path = os.path.join(self.output_dir, 'plots', 'training_history.png')
        plt.savefig(training_history_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {training_history_path}")
        plt.show()
    
    def save_model(self, filename='disease_classifier_model.pth'):
        """Save the trained model"""
        if self.model:
            # 保存到models子目录
            filepath = os.path.join(self.output_dir, 'models', filename)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'img_size': self.img_size,
                'class_names': self.class_names
            }, filepath)
            print(f"Model saved to: {filepath}")
    
    def load_model(self, filename='disease_classifier_model.pth'):
        """Load a trained model with optimal threshold"""
        # 优先从输出目录加载
        filepath = os.path.join(self.output_dir, 'models', filename)
        
        # 如果输出目录中没有，则尝试从根目录加载（向后兼容）
        if not os.path.exists(filepath):
            filepath = filename
            
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Create model
        self.model = CNNModel(num_classes=2)
        
        # Check checkpoint format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Complete checkpoint format
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.img_size = checkpoint.get('img_size', (224, 224))
            self.class_names = checkpoint.get('class_names', ['diseased', 'healthy'])
            self.optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
            print(f"Model loaded from {filepath} with optimal threshold: {self.optimal_threshold:.3f}")
        else:
            # State dict only format
            self.model.load_state_dict(checkpoint)
            self.img_size = (224, 224)
            self.class_names = ['diseased', 'healthy']
            self.optimal_threshold = 0.5
            print(f"Model loaded from {filepath} (using default threshold: 0.5)")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict_single_image(self, img_path):
        """Predict single spectral image"""
        if self.model is None:
            print("Please train or load a model first")
            return None
        
        img = self._load_image(img_path)
        if img is None:
            return None
        
        # Ensure image values are in correct range
        if img.max() > 1.0:
            img = img / 255.0
        
        # Convert to uint8 format
        img = (img * 255).astype(np.uint8)
        
        # Data transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms and add batch dimension
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            prob = torch.softmax(output, dim=1)
            
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()
            confidence = prob[0][predicted_class].item()
            
            result = self.class_names[predicted_class]
        
        return result, confidence

def main():
    """Main function for remote sensing spectral data classification"""
    # 设置随机种子确保结果可重复
    set_seed(42)
    
    # Create classifier instance
    classifier = ImageClassifier()
    
    # Load data
    X, y = classifier.load_and_preprocess_data()
    
    # Check if data was successfully loaded
    if X is None or len(X) == 0:
        print("\nError: No spectral images were successfully loaded!")
        print("Please check:")
        print("1. data folder exists")
        print("2. diseased and healthy subfolders exist")
        print("3. Image files are valid TIFF format")
        return
    
    # Create model
    model = classifier.create_model()
    print("\nModel Structure:")
    print(model)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model with stable parameters for consistent results
    print("\nStarting model training...")
    history, X_test, y_test = classifier.train_model(X, y, epochs=40, learning_rate=0.001, batch_size=16)
    
    # Evaluate model on test set
    print("\nEvaluating model performance on test set...")
    test_accuracy, _, _ = classifier.evaluate_model(X_test, y_test)
    
    # Evaluate model on full dataset for comprehensive analysis
    print("\nEvaluating model performance on full dataset...")
    full_accuracy, y_pred_full, y_pred_prob_full = classifier.evaluate_model(X, y)
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # Save model
    classifier.save_model()
    
    print("\nTraining completed!")
    print(f"Test set accuracy: {test_accuracy:.4f}")
    print(f"Full dataset accuracy: {full_accuracy:.4f}")

if __name__ == "__main__":
    main()