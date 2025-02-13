from abc import abstractmethod
from app import db
from app.models.algorithm import Algorithm

class BaseAlgorithm(Algorithm):
    """算法基类"""
    __abstract__ = True  # SQLAlchemy 不会为这个类创建表
    
    @abstractmethod
    def process(self, camera, parameters):
        """处理单帧图像"""
        pass


    @classmethod
    def get_subclasses(cls):
        """获取所有子类"""
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass

    @classmethod
    def register(cls):
        """注册算法到数据库"""
        pass

    @classmethod
    def register_algorithms(cls):
        """注册所有算法到数据库"""
        print(__file__)
        for algorithm_class in cls.get_subclasses():
            algorithm_class.register()
        db.session.commit() 