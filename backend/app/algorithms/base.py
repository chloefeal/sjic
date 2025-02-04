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

    @abstractmethod
    def get_parameters_schema(self):
        """获取算法参数schema"""
        pass

    @abstractmethod
    def validate_parameters(self, parameters):
        """验证参数是否合法"""
        pass

    @classmethod
    def get_subclasses(cls):
        """获取所有已注册的算法类"""
        def get_all_subclasses(c):
            subclasses = c.__subclasses__()
            for d in list(subclasses):
                subclasses.extend(get_all_subclasses(d))
            return subclasses
        return get_all_subclasses(cls)

    @classmethod
    @abstractmethod
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