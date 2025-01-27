from abc import abstractmethod
from app import db
from app.models.algorithm import Algorithm

class BaseAlgorithm(Algorithm):
    """算法基类"""
    __abstract__ = True  # SQLAlchemy 不会为这个类创建表
    
    @abstractmethod
    def process(self, frame, parameters):
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
    def register_algorithms(cls):
        """注册所有算法到数据库"""
        for algorithm_class in cls.get_subclasses():
            if not algorithm_class.__abstract__:
                # 使用 __mapper_args__ 中定义的 polymorphic_identity
                type_name = algorithm_class.__mapper_args__['polymorphic_identity']
                algorithm = algorithm_class.query.filter_by(type=type_name).first()
                if not algorithm:
                    algorithm = algorithm_class(
                        name=algorithm_class.__doc__.strip() or algorithm_class.__name__,
                        type=type_name,  # 使用相同的 type_name
                        description=algorithm_class.__doc__ or '',
                        parameters=algorithm_class().get_parameters_schema()
                    )
                    print(algorithm.to_dict())
                    db.session.add(algorithm)
        db.session.commit() 