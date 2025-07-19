"""
Wiki转反例处理器
从Wiki数据生成反例问题
"""

import os
import json
import random
import time
from tqdm import tqdm
import re

from .base_processor import BaseProcessor
from utils import load_json, save_json, format_filename, get_file_count



class Wiki2AntiProcessor(BaseProcessor):
    """Wiki转反例处理器"""
    

    
    def process(self):
        """从Wiki数据生成反例问题"""
        self.log("开始从Wiki数据生成反例问题...")
        
        # 输入输出路径
        wiki_path = self.path_manager.get_local_input_path("wiki", f"wiki_{self.role}.txt")
        output_path = self.path_manager.get_output_path("process", "anti", f"{self.role}_anti.json")
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            self.log(f"输出文件已存在，跳过处理: {output_path}")
            return True
        
        # 检查输入文件是否存在
        if not os.path.exists(wiki_path):
            self.log(f"Wiki文件不存在，跳过处理: {wiki_path}")
            return True
        
        # 读取Wiki数据
        self.log(f"读取Wiki数据: {wiki_path}")
        wiki_data = self._load_wiki_data(wiki_path)
        
        if not wiki_data:
            self.log("Wiki数据为空，跳过处理")
            return True
        
        # 读取通用背景信息
        general_path = self.path_manager.get_local_input_path("general", f"general_{self.role}.txt")
        if not os.path.exists(general_path):
            self.log(f"通用背景信息文件不存在: {general_path}")
            return False
        
        with open(general_path, 'r', encoding='utf-8') as f:
            general_info = f.read().strip()
        
        # 用于存储生成的反例
        all_anti_data = []
        
        # 从Wiki数据生成反例问题
        self.log(f"开始生成反例问题，共 {len(wiki_data)} 个段落...")
        
        # 在示例模式下限制段落数量
        wiki_passages = self.limit_data_for_demo(wiki_data)
        
        for passage_index, passage in enumerate(tqdm(wiki_passages, desc=f"处理 {self.role} 的Wiki段落")):
            if not passage.strip():
                continue
            
            # 使用完整的prompt模板
            prompt = self.get_prompt("wiki2anti", character=self.role, passage=passage)
            
            # 调用API生成反例问题
            try:
                # 将prompt转换为messages格式
                messages = [{"role": "user", "content": prompt}]
                response = self.call_api(messages, temperature=0.8)
                anti_items = self._parse_anti_response(response)
                
                # 处理每个反例类型
                for anti_item in anti_items:
                    if "type" in anti_item and "description" in anti_item and "example_keywords" in anti_item:
                        anti_data = {
                            "type": anti_item["type"],
                            "description": anti_item["description"],
                            "example_keywords": anti_item["example_keywords"],
                            "source": passage
                        }
                        all_anti_data.append(anti_data)
                
            except Exception as e:
                self.log(f"生成反例时出错: {e}")
                continue
        
        # 保存反例数据
        if all_anti_data:
            self.log(f"保存反例数据: {output_path}")
            save_json(all_anti_data, output_path)
            self.log(f"反例生成完成，共生成 {len(all_anti_data)} 个反例")
        else:
            self.log("未生成任何反例")
        
        return True
    
    def _load_wiki_data(self, file_path: str) -> list:
        """加载Wiki数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 按段落分割
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                return paragraphs
        except Exception as e:
            self.log(f"加载Wiki数据失败: {e}")
            return []
    
    def _parse_anti_response(self, response: str) -> list:
        """解析API响应中的反例列表"""
        try:
            # 直接输出API响应用于调试
            self.log(f"API响应内容: {response}")
            
            # 尝试提取JSON部分
            json_content = response
            if not json_content.startswith('['):
                json_match = re.search(r'\[\s*{.*}\s*\]', json_content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0)
            
            anti_items = json.loads(json_content)
            if isinstance(anti_items, list):
                return anti_items
            else:
                return []
        except Exception as e:
            self.log(f"解析反例响应失败: {e}")
            self.log(f"原始响应: {response}")
            return [] 