import yaml
from pathlib import Path
from ultralytics import YOLO
from app import app

class ModelTrainer:
    def __init__(self, config_path='config/training_config.yaml'):
        self.config_path = config_path
        self.load_config()
        self.model = None
        
    def load_config(self):
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
            
    def prepare_dataset(self, data_path):
        """准备训练数据集"""
        # 实现数据集准备逻辑
        dataset_yaml = {
            'path': data_path,
            'train': 'train/images',
            'val': 'valid/images',
            'names': self.config['classes']
        }
        
        yaml_path = Path(data_path) / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f)
            
        return str(yaml_path)
        
    def train(self, dataset_path, config):
        with app.app_context():
            try:
                # 初始化YOLO模型
                self.model = YOLO('yolov8n.pt')
                
                # 开始训练
                results = self.model.train(
                    data=dataset_path,
                    epochs=config.get('epochs', 100),
                    batch=config.get('batchSize', 16),
                    imgsz=640,
                    save=True,
                    project='models',
                    name=config.get('name', 'custom')
                )
                
                return {
                    'success': True,
                    'model_path': str(results.save_dir)
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                } 