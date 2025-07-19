"""
对话转问答处理器
从对话数据直接生成问答对
"""

import os
import json
import random
import time
from tqdm import tqdm
import re
from collections import defaultdict

from .base_processor import BaseProcessor
from utils import load_json, save_json, format_filename, get_file_count



class Conv2QAProcessor(BaseProcessor):
    """对话转问答处理器"""
    

    
    def process(self):
        """从对话数据直接生成问答对"""
        self.log("开始从对话数据生成问答对...")
        
        # 输入输出路径
        conversation_path = self.path_manager.get_profile_path(self.world, self.role)
        output_path = self.path_manager.get_output_path("qa", "qa_conv", f"{self.world}_{self.role}_qa_conv.json")
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            self.log(f"输出文件已存在，跳过处理: {output_path}")
            return True
        
        # 检查输入文件是否存在
        if not os.path.exists(conversation_path):
            self.log(f"输入文件不存在，跳过处理: {conversation_path}")
            return True
        
        # 读取对话数据
        self.log(f"读取对话数据: {conversation_path}")
        conversation_data = self._load_conversation_data(conversation_path)
        
        if not conversation_data:
            self.log("对话数据为空，跳过处理")
            return True
        
        # 用于存储生成的问答对
        all_qa_pairs = []
        
        # 从对话中生成问答对
        self.log(f"开始生成问答对，共 {len(conversation_data)} 个对话场景...")
        
        # 在示例模式下限制对话场景数量
        conversation_data = self.limit_data_for_demo(conversation_data)
        
        for scene_id, conversation in enumerate(tqdm(conversation_data, desc=f"处理 {self.role} 的对话")):
            if not conversation:
                continue
            
            # 使用完整的prompt模板
            prompt = self.get_prompt("conv2qa", role=self.role, scene_id=scene_id, roles=", ".join([self.role]), content=conversation)
            
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
            self.log(f"保存对话问答对数据: {output_path}")
            save_json(all_qa_pairs, output_path)
            self.log(f"对话问答对生成完成，共生成 {len(all_qa_pairs)} 条数据")
        else:
            self.log("未生成任何对话问答对")
        
        return True
    
    def _load_conversation_data(self, file_path: str) -> list:
        """加载对话数据并按场景重组"""
        # 按scene_id分组对话
        scene_conversations = defaultdict(list)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        scene_id = data.get('scene_id', 0)
                        role = data.get('role', '')
                        content = data.get('content', '')
                        
                        if content:
                            scene_conversations[scene_id].append({
                                'role': role,
                                'content': content
                            })
        except Exception as e:
            self.log(f"加载对话数据失败: {e}")
            return []
        
        # 将每个场景的对话组合成完整对话文本
        conversations = []
        for scene_id in sorted(scene_conversations.keys()):
            scene_data = scene_conversations[scene_id]
            
            # 组合对话内容
            conversation_text = ""
            for item in scene_data:
                role = item['role']
                content = item['content']
                conversation_text += f"{role}: {content}\n"
            
            if conversation_text.strip():
                conversations.append(conversation_text.strip())
        
        return conversations
    
    def _parse_qa_response(self, response: str) -> dict:
        """解析API响应中的问答对"""
        try:
            # 直接输出API响应用于调试
            self.log(f"API响应内容: {response}")
            
            # 尝试提取JSON部分
            json_content = response
            if not json_content.startswith('{'):
                json_match = re.search(r'\{.*\}', json_content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0)
            
            qa_pair = json.loads(json_content)
            if isinstance(qa_pair, dict):
                return qa_pair
            else:
                return {}
        except Exception as e:
            self.log(f"解析问答对响应失败: {e}")
            self.log(f"原始响应: {response}")
            return {} 