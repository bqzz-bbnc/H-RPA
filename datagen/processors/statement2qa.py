"""
从角色陈述生成问答对处理器
"""

import os
import json
import random
import time
from tqdm import tqdm
import re

from .base_processor import BaseProcessor
from utils import load_json, save_json, format_filename, get_file_count



class Statement2QAProcessor(BaseProcessor):
    """从角色陈述生成问答对处理器"""
    

    
    def process(self):
        """从角色陈述生成问答对"""
        self.log("开始从角色陈述生成问答对...")
        
        # 输入输出路径
        statement_path = self.path_manager.get_output_path("process", "statement", f"{self.role}_statement.json")
        output_path = self.path_manager.get_output_path("qa", "qa_statement", f"{self.world}_{self.role}_qa_statement.json")
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            self.log(f"输出文件已存在，跳过处理: {output_path}")
            return True
        
        # 检查输入文件是否存在
        if not os.path.exists(statement_path):
            self.log(f"输入文件不存在，跳过处理: {statement_path}")
            return True
        
        # 读取角色陈述数据
        self.log(f"读取角色陈述数据: {statement_path}")
        statements_data = load_json(statement_path)
        
        if not statements_data:
            self.log("角色陈述数据为空，跳过处理")
            return True
        
        # 读取通用背景信息
        general_path = self.path_manager.get_local_input_path("general", f"general_{self.role}.txt")
        if not os.path.exists(general_path):
            self.log(f"通用背景信息文件不存在: {general_path}")
            return False
        
        with open(general_path, 'r', encoding='utf-8') as f:
            general_info = f.read().strip()
        
        # 用于存储生成的问答对
        all_qa_pairs = []
        
        # 随机打乱陈述顺序以获得多样性
        all_statements = []
        for item in statements_data:
            for statement in item["statements"]:
                all_statements.append(statement)
        
        random.shuffle(all_statements)
        
        # 从陈述生成问答对
        self.log(f"开始生成问答对，共 {len(all_statements)} 个陈述...")
        
        # 在示例模式下限制陈述数量
        all_statements = self.limit_data_for_demo(all_statements)
        
        for statement_index, statement in enumerate(tqdm(all_statements, desc=f"处理 {self.role} 的陈述")):
            # 使用完整的prompt模板
            prompt = self.get_prompt("statement2qa", character=self.role, statement=statement, general=general_info)
            
            # 调用API生成问答对
            try:
                # 将prompt转换为messages格式
                messages = [{"role": "user", "content": prompt}]
                response = self.call_api(messages, temperature=0.8)
                qa_pairs = self._parse_qa_response(response)
                
                # 处理每个问答对
                for qa_pair in qa_pairs:
                    if "question" in qa_pair and "answer" in qa_pair:
                        qa_item = {
                            "question": qa_pair["question"],
                            "answer": qa_pair["answer"],
                            "retrieve": statement
                        }
                        all_qa_pairs.append(qa_item)
                
            except Exception as e:
                self.log(f"生成问答对时出错: {e}")
                continue
        
        # 保存问答对数据
        if all_qa_pairs:
            self.log(f"保存问答对数据: {output_path}")
            save_json(all_qa_pairs, output_path)
            self.log(f"问答对生成完成，共生成 {len(all_qa_pairs)} 条数据")
        else:
            self.log("未生成任何问答对")
        
        return True
    
    def _parse_qa_response(self, response: str) -> list:
        """解析API响应中的问答对"""
        try:
            # 尝试提取JSON部分
            json_content = response
            if not json_content.startswith('['):
                json_match = re.search(r'\[\s*{.*}\s*\]', json_content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0)
            
            qa_pairs = json.loads(json_content)
            if isinstance(qa_pairs, list):
                return qa_pairs
            else:
                return []
        except Exception as e:
            self.log(f"解析问答对响应失败: {e}")
            return []
    
 