"""
基础处理器类
定义统一的处理器接口
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseProcessor(ABC):
    """基础处理器抽象类"""
    
    def __init__(self, generator):
        """
        初始化处理器
        
        Args:
            generator: 数据生成器实例
        """
        self.generator = generator
        self.world = generator.world
        self.role = generator.role
        self.config = generator.config
        self.path_manager = generator.path_manager
        
        # 初始化prompt管理器
        try:
            import sys
            import os
            # 添加项目根目录到sys.path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from prompts import PromptManager
            language = self.config.get('language', 'zh')
            self.prompt_manager = PromptManager(language)
        except ImportError as e:
            print(f"Warning: Failed to import PromptManager: {e}")
            self.prompt_manager = None
    
    @abstractmethod
    def process(self):
        """
        处理数据
        
        Returns:
            处理是否成功
        """
        pass
    
    def get_name(self) -> str:
        """获取处理器名称"""
        return self.__class__.__name__
    
    def log(self, message: str):
        """记录日志"""
        print(f"    [{self.get_name()}] {message}")
    
    def call_api(self, messages, model=None, temperature=0.8):
        """调用OpenAI API"""
        return self.generator.call_openai_api(messages, model, temperature)
    
    def get_prompt(self, prompt_type: str, **kwargs):
        """获取prompt模板"""
        if self.prompt_manager:
            return self.prompt_manager.get_prompt(prompt_type, **kwargs)
        else:
            # 如果没有prompt管理器，返回简单的提示
            return f"请为{self.role}生成{prompt_type}类型的数据"
    
    def limit_data_for_demo(self, data_list: list) -> list:
        """
        在示例模式下限制数据量
        
        Args:
            data_list: 原始数据列表
            
        Returns:
            限制后的数据列表
        """
        if not self.config.get('demo_mode.enabled', False):
            return data_list
        
        max_items = self.config.get('demo_mode.max_items_per_api_call', 2)
        limited_data = data_list[:max_items]
        
        if len(data_list) > max_items:
            self.log(f"示例模式：限制数据量从 {len(data_list)} 条到 {len(limited_data)} 条")
        
        return limited_data 