"""
聊天问答处理器
从角色背景信息生成聊天问答对
"""

import os
import json
import random
import time
from tqdm import tqdm
import re

from .base_processor import BaseProcessor
from utils import load_json, save_json, format_filename, get_file_count



class Chat2QAProcessor(BaseProcessor):
    """聊天问答处理器"""
    

    
    def process(self):
        """生成聊天问答对"""
        self.log("开始生成聊天问答对...")
        
        # 输出文件路径
        output_path = self.path_manager.get_output_path("qa", "qa_chat", f"{self.world}_{self.role}_qa_chat.json")
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            self.log(f"输出文件已存在，跳过处理: {output_path}")
            return True
        
        # 读取通用背景信息
        general_path = self.path_manager.get_local_input_path("general", f"general_{self.role}.txt")
        if not os.path.exists(general_path):
            self.log(f"通用背景信息文件不存在: {general_path}")
            return False
        
        with open(general_path, 'r', encoding='utf-8') as f:
            general = f.read().strip()
        
        # 用于存储生成的问答对
        all_qa_pairs = []
        
        # Step 1: 生成聊天主题
        self.log("生成聊天主题...")
        topics_prompt = self.get_prompt("chat2qa_topics", character=self.role, general=general)
        
        try:
            # 将prompt转换为messages格式
            topics_messages = [{"role": "user", "content": topics_prompt}]
            topics_response = self.call_api(topics_messages, temperature=0.8)
            topics = self._parse_topics_response(topics_response)
            
            if not topics:
                self.log("未能生成聊天主题，跳过处理")
                return False
            
            # 在示例模式下限制主题数量
            topics = self.limit_data_for_demo(topics)
            
            self.log(f"生成了 {len(topics)} 个聊天主题")
            
        except Exception as e:
            self.log(f"生成聊天主题时出错: {e}")
            return False
        
        # Step 2: 基于每个主题生成问答对
        self.log("基于主题生成问答对...")
        
        for topic_index, topic in enumerate(tqdm(topics, desc=f"处理 {self.role} 的聊天主题")):
            # 使用完整的prompt模板
            qa_prompt = self.get_prompt("chat2qa", character=self.role, general=general, topic=topic)
            
            try:
                # 将prompt转换为messages格式
                qa_messages = [{"role": "user", "content": qa_prompt}]
                qa_response = self.call_api(qa_messages, temperature=0.8)
                qa_pairs = self._parse_qa_response(qa_response)
                
                # 处理每个问答对
                for qa_pair in qa_pairs:
                    if "question" in qa_pair and "answer" in qa_pair:
                        qa_item = {
                            "question": qa_pair["question"],
                            "answer": qa_pair["answer"],
                            "retrieve": ""
                        }
                        all_qa_pairs.append(qa_item)
                
            except Exception as e:
                self.log(f"生成问答对时出错: {e}")
                continue
        
        # 保存问答对数据
        if all_qa_pairs:
            self.log(f"保存聊天问答对数据: {output_path}")
            save_json(all_qa_pairs, output_path)
            self.log(f"聊天问答对生成完成，共生成 {len(all_qa_pairs)} 条数据")
        else:
            self.log("未生成任何聊天问答对")
        
        return True
    
    def _parse_topics_response(self, response: str) -> list:
        """解析API响应中的主题列表"""
        try:
            # 尝试提取JSON部分
            json_content = response
            if not json_content.startswith('['):
                json_match = re.search(r'\[\s*".*"\s*\]', json_content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0)
            
            topics = json.loads(json_content)
            if isinstance(topics, list):
                return topics
            else:
                return []
        except Exception as e:
            self.log(f"解析主题响应失败: {e}")
            return []
    
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
    
 