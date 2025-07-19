"""
摘要转问答处理器
从对话摘要生成问答对
"""

import os
import json
import random
import time
from tqdm import tqdm
import re

from .base_processor import BaseProcessor
from utils import load_json, save_json, format_filename, get_file_count



class Summary2QAProcessor(BaseProcessor):
    """摘要转问答处理器"""
    

    
    def process(self):
        """从对话摘要生成问答对"""
        self.log("开始从对话摘要生成问答对...")
        
        # 输入输出路径
        summary_path = self.path_manager.get_output_path("process", "summary", f"{self.world}_{self.role}_summary.json")
        output_path = self.path_manager.get_output_path("qa", "qa_summary", f"{self.world}_{self.role}_qa_summary.json")
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            self.log(f"输出文件已存在，跳过处理: {output_path}")
            return True
        
        # 检查输入文件是否存在
        if not os.path.exists(summary_path):
            self.log(f"输入文件不存在，跳过处理: {summary_path}")
            return True
        
        # 读取摘要数据
        self.log(f"读取摘要数据: {summary_path}")
        summary_data = load_json(summary_path)
        
        if not summary_data:
            self.log("摘要数据为空，跳过处理")
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
        
        # 从摘要生成问答对
        self.log(f"开始生成问答对，共 {len(summary_data)} 个摘要...")
        
        # 在示例模式下限制摘要数量
        summary_data = self.limit_data_for_demo(summary_data)
        
        for summary_index, summary_item in enumerate(tqdm(summary_data, desc=f"处理 {self.role} 的摘要")):
            summary = summary_item.get("summary", "")
            if not summary:
                continue
            
            # 使用完整的prompt模板
            prompt = self.get_prompt("summary2qa", world=self.world, role=self.role, summary=summary, role_highlight=f"{self.role}在场景中的表现")
            
            # 调用API生成问答对
            try:
                # 将prompt转换为messages格式
                messages = [{"role": "user", "content": prompt}]
                response = self.call_api(messages, temperature=0.8)
                qa_pair = self._parse_qa_response(response)
                
                # 处理问答对
                if qa_pair and "question" in qa_pair and "answer" in qa_pair:
                    qa_item = {
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"]
                    }
                    all_qa_pairs.append(qa_item)
                
            except Exception as e:
                self.log(f"生成问答对时出错: {e}")
                continue
        
        # 保存问答对数据
        if all_qa_pairs:
            self.log(f"保存摘要问答对数据: {output_path}")
            save_json(all_qa_pairs, output_path)
            self.log(f"摘要问答对生成完成，共生成 {len(all_qa_pairs)} 条数据")
        else:
            self.log("未生成任何摘要问答对")
        
        return True
    
    def _parse_qa_response(self, response: str) -> dict:
        """解析API响应中的问答对"""
        try:
            # 直接输出API响应用于调试
            self.log(f"API响应内容: {response}")
            
            # 尝试多种解析方式
            # 首先尝试JSON格式（优先级最高）
            try:
                qa_data = json.loads(response)
                if isinstance(qa_data, dict) and 'question' in qa_data and 'answer' in qa_data:
                    return qa_data
                elif isinstance(qa_data, list) and len(qa_data) > 0:
                    return qa_data[0]
            except:
                pass
            
            # 然后尝试文本格式
            if '问题：' in response and '回答：' in response:
                # 方式1: 问题：[内容] 回答：[内容]
                question_part = response.split('问题：')[1].split('回答：')[0].strip()
                answer_part = response.split('回答：')[1].strip()
            elif '问题:' in response and '回答:' in response:
                # 方式2: 问题:[内容] 回答:[内容]
                question_part = response.split('问题:')[1].split('回答:')[0].strip()
                answer_part = response.split('回答:')[1].strip()
            elif 'question' in response.lower() and 'answer' in response.lower():
                # 方式3: 英文格式
                response_lower = response.lower()
                if 'question:' in response_lower and 'answer:' in response_lower:
                    question_part = response.split('question:')[1].split('answer:')[0].strip()
                    answer_part = response.split('answer:')[1].strip()
                else:
                    raise Exception("无法识别的问答格式")
            else:
                raise Exception("无法识别的响应格式")
            
            # 清理引号
            if question_part.startswith('"') and question_part.endswith('"'):
                question_part = question_part[1:-1]
            if answer_part.startswith('"') and answer_part.endswith('"'):
                answer_part = answer_part[1:-1]
            
            return {
                "question": question_part,
                "answer": answer_part
            }
        except Exception as e:
            self.log(f"解析问答对响应失败: {e}")
            self.log(f"原始响应: {response}")
            return {}
    
 