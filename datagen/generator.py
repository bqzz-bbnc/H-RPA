"""
主数据生成器类
集成所有数据处理流程，统一接口和风格
"""

import os
import time
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI

from utils import Config, PathManager, load_json, save_json, load_jsonl, save_jsonl, shuffle_data, split_train_test, format_filename, get_file_count
from processors.wiki2statement import Wiki2StatementProcessor
from processors.statement2qa import Statement2QAProcessor
from processors.conv2summary import Conv2SummaryProcessor
from processors.summary2qa import Summary2QAProcessor
from processors.chat2qa import Chat2QAProcessor
from processors.wiki2anti import Wiki2AntiProcessor
from processors.anti2qa import Anti2QAProcessor
from processors.conv2qa import Conv2QAProcessor
from processors.conv2style import Conv2StyleProcessor
from processors.qa2all import QA2AllProcessor


class DataGenerator:
    """主数据生成器"""
    
    def __init__(self, world: str, role: str, config_path: str = "config.yaml"):
        """
        初始化数据生成器
        
        Args:
            world: 世界名称
            role: 角色名称
            config_path: 配置文件路径
        """
        self.world = world
        self.role = role
        
        # 初始化配置和路径管理
        self.config = Config(config_path)
        self.path_manager = PathManager(self.config)
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.config.get('openai.api_key'),
            base_url=self.config.get('openai.base_url')
        )
        
        # 初始化处理器
        self._init_processors()
        
        # 确保输出目录存在
        self._ensure_directories()
    
    def _init_processors(self):
        """初始化所有处理器"""
        self.processors = {
            'wiki2statement': Wiki2StatementProcessor(self),
            'statement2qa': Statement2QAProcessor(self),
            'conv2summary': Conv2SummaryProcessor(self),
            'summary2qa': Summary2QAProcessor(self),
            'chat2qa': Chat2QAProcessor(self),
            'wiki2anti': Wiki2AntiProcessor(self),
            'anti2qa': Anti2QAProcessor(self),
            'conv2qa': Conv2QAProcessor(self),
            'conv2style': Conv2StyleProcessor(self),
            'qa2all': QA2AllProcessor(self)
        }
    
    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        output_base = self.config.get('paths.output_base')
        if output_base:
            dirs = [
                self.config.get('paths.process_dir'),
                self.config.get('paths.qa_dir'),
                self.config.get('paths.all_dir'),
                self.config.get('paths.train_dir'),
                self.config.get('paths.test_dir'),
                os.path.join(output_base, "style")  # 添加style目录
            ]
            for dir_path in dirs:
                if dir_path and isinstance(dir_path, str):  # 确保路径不为空且为字符串
                    self.path_manager.ensure_dir(dir_path)
    
    def call_openai_api(self, messages: List[Dict], model: str = None, temperature: float = 0.8) -> str:
        """
        调用OpenAI API
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
        
        Returns:
            API响应内容
        """
        if model is None:
            model = self.config.get('models.base_model')
        
        # 简化API调用，移除复杂的重试逻辑
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
        except Exception as e:
            # 直接抛出异常，不包装
            print(f"API调用失败:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            raise
    
    def generate_wiki2statement(self):
        """生成Wiki到陈述的数据"""
        print(f"  - 生成 {self.role} 的Wiki陈述...")
        self.processors['wiki2statement'].process()
    
    def generate_statement2qa(self):
        """生成陈述到问答的数据"""
        print(f"  - 生成 {self.role} 的陈述问答...")
        self.processors['statement2qa'].process()
    
    def generate_conv2summary(self):
        """生成对话到摘要的数据"""
        print(f"  - 生成 {self.role} 的对话摘要...")
        self.processors['conv2summary'].process()
    
    def generate_summary2qa(self):
        """生成摘要到问答的数据"""
        print(f"  - 生成 {self.role} 的摘要问答...")
        self.processors['summary2qa'].process()
    
    def generate_chat2qa(self):
        """生成聊天问答数据"""
        print(f"  - 生成 {self.role} 的聊天问答...")
        self.processors['chat2qa'].process()
    
    def generate_wiki2anti(self):
        """生成Wiki到反例的数据"""
        print(f"  - 生成 {self.role} 的Wiki反例...")
        self.processors['wiki2anti'].process()
    
    def generate_anti2qa(self):
        """生成反例问答数据"""
        print(f"  - 生成 {self.role} 的反例问答...")
        self.processors['anti2qa'].process()
    
    def generate_conv2qa(self):
        """生成对话问答数据"""
        print(f"  - 生成 {self.role} 的对话问答...")
        self.processors['conv2qa'].process()
    
    def generate_conv2style(self):
        """生成对话风格数据"""
        print(f"  - 生成 {self.role} 的风格数据...")
        self.processors['conv2style'].process()
    
    def generate_qa2all(self):
        """合并所有问答数据"""
        print(f"  - 合并 {self.role} 的所有问答数据...")
        self.processors['qa2all'].process()
    
    def split_train_test(self):
        """切分训练集和测试集"""
        print(f"  - 切分 {self.role} 的训练测试集...")
        
        # 查找合并后的QA文件
        qa_dir = self.config.get('paths.all_dir')
        pattern = f"{self.world}_{self.role}_qa_*.json"
        
        import glob
        qa_files = glob.glob(os.path.join(qa_dir, pattern))
        
        if not qa_files:
            print(f"    警告: 未找到 {self.role} 的QA文件")
            return
        
        qa_file = qa_files[0]  # 使用第一个匹配的文件
        data = load_json(qa_file)
        
        # 按source_type分组
        from collections import defaultdict
        grouped_data = defaultdict(list)
        for item in data:
            source_type = item.get("source_type", "unknown")
            grouped_data[source_type].append(item)
        
        # 分层采样
        train_data = []
        test_data = []
        split_ratio = self.config.get('generation.train_test_split', 0.8)
        seed = self.config.get('generation.random_seed', 42)
        
        for source_type, items in grouped_data.items():
            train_items, test_items = split_train_test(items, split_ratio, seed)
            train_data.extend(train_items)
            test_data.extend(test_items)
        
        # 保存训练集和测试集
        train_file = os.path.join(self.config.get('paths.train_dir'), f"{self.world}_{self.role}_train.json")
        test_file = os.path.join(self.config.get('paths.test_dir'), f"{self.world}_{self.role}_test.json")
        
        save_json(train_data, train_file)
        save_json(test_data, test_file)
        
        print(f"    训练集: {len(train_data)} 条")
        print(f"    测试集: {len(test_data)} 条")
    
    def run(self):
        """运行完整的数据生成流程"""
        print(f"\n开始为角色 {self.role} 生成数据...")
        
        try:
            # 执行所有处理步骤
            self.generate_wiki2statement()
            self.generate_statement2qa()
            self.generate_conv2summary()
            self.generate_summary2qa()
            self.generate_chat2qa()
            self.generate_wiki2anti()
            self.generate_anti2qa()
            self.generate_conv2qa()
            self.generate_conv2style()  # style相关，在qa2all之前
            self.generate_qa2all()
            self.split_train_test()
            
            print(f"角色 {self.role} 数据生成完成！")
            
        except Exception as e:
            print(f"角色 {self.role} 数据生成失败: {e}")
            raise 