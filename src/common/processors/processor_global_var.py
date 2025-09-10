# 一个单例模式的配置管理器，用于在整个视频处理流程中共享关键参数。
from src.core.dicts import VideoInfoDict # 自定义类型注解
from src.utils import singleton # 单例装饰器（确保类只有一个实例）


@singleton
class ProcessorGlobalVar:
    _instance = None

    def __new__(cls, *args, **kwargs):
    # 效果：整个应用中只有一个ProcessorGlobalVar实例
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
    # 初始化为_data: 类型为VideoInfoDict的字典
        self._data: VideoInfoDict = {} 

    def get_data(self) -> VideoInfoDict:
    # 返回整个配置字典
        return self._data

    def get(self, key: str):
    # 返回字典键的对应值（不存在则返回None）
        if key not in VideoInfoDict.__annotations__:
            raise KeyError(f"{key} is not a valid key.")
        return self._data.get(key)

    def update(self, key: str, value):
    # 更新字典键的对应值
        if key not in VideoInfoDict.__annotations__:
            raise KeyError(f"{key} is not a valid key.")

        self._data[key] = value

    def clear(self):
    # 清空数据，在处理新视频前重置状态
        self._data.clear()

    def __repr__(self):
    # 调试时显示当前存储的所有值
        return f"ProcessorGlobalVar({self._data})"
