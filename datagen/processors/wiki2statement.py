"""
Wiki转陈述处理器
从Wiki段落生成角色陈述
"""

import os
import json
import random
import time
from tqdm import tqdm
import re

from .base_processor import BaseProcessor
from utils import load_json, save_json, format_filename, get_file_count



class Wiki2StatementProcessor(BaseProcessor):
    """Wiki转陈述处理器"""
    

    
    def process(self):
        """从Wiki段落生成角色陈述"""
        self.log("开始从Wiki段落生成角色陈述...")
        
        # 输入输出路径
        wiki_path = self.path_manager.get_local_input_path("wiki", f"wiki_{self.role}.txt")
        output_path = self.path_manager.get_output_path("process", "statement", f"{self.role}_statement.json")
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            self.log(f"输出文件已存在，跳过处理: {output_path}")
            return True
        
        # 检查输入文件是否存在
        if not os.path.exists(wiki_path):
            self.log(f"Wiki文件不存在，跳过处理: {wiki_path}")
            return True
        
        # 读取Wiki数据
        wiki_path = self.path_manager.get_local_input_path("wiki", f"wiki_{self.role}.txt")
        if not os.path.exists(wiki_path):
            self.log(f"Wiki文件不存在，跳过处理: {wiki_path}")
            return True
        
        with open(wiki_path, 'r', encoding='utf-8') as f:
            wiki_content = f.read().strip()
        
        # 按段落分割Wiki内容
        wiki_passages = [p.strip() for p in wiki_content.split('\n\n') if p.strip()]
        
        # 在示例模式下限制段落数量
        wiki_passages = self.limit_data_for_demo(wiki_passages)
        
        # 读取通用背景信息
        general_path = self.path_manager.get_local_input_path("general", f"general_{self.role}.txt")
        if not os.path.exists(general_path):
            self.log(f"通用背景信息文件不存在: {general_path}")
            return False
        
        with open(general_path, 'r', encoding='utf-8') as f:
            general_info = f.read().strip()
        
        # 用于存储生成的陈述
        all_statements = []
        
        # 从Wiki段落生成陈述
        self.log(f"开始生成陈述，共 {len(wiki_passages)} 个段落...")
        
        for passage_index, passage in enumerate(tqdm(wiki_passages, desc=f"处理 {self.role} 的Wiki段落")):
            if not passage.strip():
                continue
            
            # 使用完整的prompt模板
            prompt = self.get_prompt("wiki2statement", character=self.role, passage=passage, general=general_info)
            
            # 调用API生成陈述
            try:
                # 将prompt转换为messages格式
                messages = [{"role": "user", "content": prompt}]
                response = self.call_api(messages, temperature=0.8)
                statements = self._parse_statements_response(response)
                
                if statements:
                    statement_item = {
                        "passage": passage,
                        "statements": statements
                    }
                    all_statements.append(statement_item)
                
            except Exception as e:
                self.log(f"生成陈述时出错: {str(e)}")
                if hasattr(e, '__class__'):
                    self.log(f"错误类型: {e.__class__.__name__}")
                continue
        
        # 保存陈述数据
        if all_statements:
            self.log(f"保存陈述数据: {output_path}")
            save_json(all_statements, output_path)
            self.log(f"陈述生成完成，共生成 {len(all_statements)} 个段落的数据")
        else:
            self.log("未生成任何陈述")
        
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
    
    def _parse_statements_response(self, response: str) -> list:
        """解析API响应中的陈述列表"""
        try:
            # 提取以"- "开头的陈述
            statements = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('- ') and self.role in line:
                    statement = line[2:]  # 去掉"- "前缀
                    statements.append(statement)
            
            return statements
        except Exception as e:
            self.log(f"解析陈述响应失败: {e}")
            return [] 