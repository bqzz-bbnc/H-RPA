"""
反例问答处理器
从反例数据生成问答对
"""

import os
import json
import random
import time
from tqdm import tqdm
import re

from .base_processor import BaseProcessor
from utils import load_json, save_json, format_filename, get_file_count



class Anti2QAProcessor(BaseProcessor):
    """反例问答处理器"""
    

    
    def process(self):
        """从反例数据生成问答对"""
        self.log("开始从反例数据生成问答对...")
        
        # 输入输出路径
        anti_path = self.path_manager.get_output_path("process", "anti", f"{self.role}_anti.json")
        output_path = self.path_manager.get_output_path("qa", "qa_anti", f"{self.world}_{self.role}_qa_anti.json")
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            self.log(f"输出文件已存在，跳过处理: {output_path}")
            return True
        
        # 检查输入文件是否存在
        if not os.path.exists(anti_path):
            self.log(f"输入文件不存在，跳过处理: {anti_path}")
            return True
        
        # 读取反例数据
        self.log(f"读取反例数据: {anti_path}")
        anti_data = load_json(anti_path)
        
        if not anti_data:
            self.log("反例数据为空，跳过处理")
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
        
        # 从反例生成问答对
        self.log(f"开始生成问答对，共 {len(anti_data)} 个反例...")
        
        # 在示例模式下限制反例数量
        anti_data = self.limit_data_for_demo(anti_data)
        
        for anti_index, anti_item in enumerate(tqdm(anti_data, desc=f"处理 {self.role} 的反例")):
            question_type = anti_item.get("type", "")
            description = anti_item.get("description", "")
            example_keywords = anti_item.get("example_keywords", [])
            
            for keyword in example_keywords:
                # 使用完整的prompt模板
                prompt = self.get_prompt("anti2qa", world=self.world, role=self.role, question_type=question_type, description=description, keyword=keyword, general=general_info)
                
                # 调用API生成问答对
                try:
                    # 将prompt转换为messages格式
                    messages = [{"role": "user", "content": prompt}]
                    response = self.call_api(messages, temperature=0.8)
                    qa_pairs = self._parse_anti_qa_response(response)
                    
                    # 处理每个问答对
                    for qa_pair in qa_pairs:
                        if "query" in qa_pair and "answer" in qa_pair:
                            qa_item = {
                                "question": qa_pair["query"],
                                "answer": qa_pair["answer"],
                                "retrieve": "",
                                "hallucination": question_type
                            }
                            all_qa_pairs.append(qa_item)
                    
                except Exception as e:
                    self.log(f"生成反例问答对时出错: {e}")
                    continue
        
        # 保存问答对数据
        if all_qa_pairs:
            self.log(f"保存反例问答对数据: {output_path}")
            save_json(all_qa_pairs, output_path)
            self.log(f"反例问答对生成完成，共生成 {len(all_qa_pairs)} 条数据")
        else:
            self.log("未生成任何反例问答对")
        
        return True
    
    def _parse_anti_qa_response(self, response: str) -> list:
        """解析API响应中的反例问答对"""
        try:
            # 直接输出API响应用于调试
            self.log(f"API响应内容: {response}")
            
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
            self.log(f"解析反例问答对响应失败: {e}")
            self.log(f"原始响应: {response}")
            return []
    
 