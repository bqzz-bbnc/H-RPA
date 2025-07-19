"""
对话风格迁移处理器
从对话数据生成风格迁移测试数据
"""

import os
import json
import random
from tqdm import tqdm
from collections import defaultdict

from .base_processor import BaseProcessor
from utils import load_json, save_json, format_filename, get_file_count



class Conv2StyleProcessor(BaseProcessor):
    """对话风格迁移处理器"""
    

    
    def process(self):
        """从对话数据生成风格迁移测试数据"""
        self.log("开始处理角色风格训练数据...")
        
        # 输入输出路径
        conversation_path = self.path_manager.get_profile_path(self.world, self.role)
        output_path = self.path_manager.get_style_path(self.world, self.role)
        
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
        
        # 用于存储生成的风格迁移数据
        style_transfer_data = []
        
        # 从对话中生成风格迁移数据
        self.log(f"开始生成风格迁移数据，共 {len(conversation_data)} 个对话场景...")
        
        # 在示例模式下限制对话场景数量
        conversation_data = self.limit_data_for_demo(conversation_data)
        
        for scene_id, conversation in enumerate(tqdm(conversation_data, desc=f"处理 {self.role} 的对话")):
            if not conversation:
                continue
            
            # 提取该角色在对话中的回答
            role_responses = self._extract_role_responses(conversation)
            
            for response in role_responses:
                # 生成错误的风格回答
                try:
                    # 使用完整的prompt模板
                    broken_styles = ["书面语", "翻译腔", "去情绪化"]
                    broken_style = random.choice(broken_styles)
                    prompt = self.get_prompt("conv2style", role=self.role, input_data="", chosen=response, broken_style=broken_style)
                    
                    # 将prompt转换为messages格式
                    messages = [{"role": "user", "content": prompt}]
                    rejected_response = self.call_api(messages, temperature=0.8)
                    
                    # 清理响应，移除"- rejected:"前缀
                    rejected_response = rejected_response.strip()
                    if rejected_response.startswith('- rejected:'):
                        rejected_response = rejected_response[len('- rejected:'):].strip()
                    if rejected_response.startswith('"'):
                        rejected_response = rejected_response[1:]
                    if rejected_response.endswith('"'):
                        rejected_response = rejected_response[:-1]
                    
                    # 构造风格迁移数据
                    style_item = {
                        "system": "你是一个语言改写助手，将这段语句转换为扮演人物的说话语气",
                        "instruction": f"你正在扮演{self.role}，你需要将下面的句子转写成{self.role}的口吻",
                        "input": rejected_response,
                        "output": response
                    }
                    style_transfer_data.append(style_item)
                    
                except Exception as e:
                    self.log(f"生成风格迁移数据时出错: {e}")
                    continue
        
        # 保存风格迁移数据
        if style_transfer_data:
            self.log(f"保存风格迁移数据: {output_path}")
            save_json(style_transfer_data, output_path)
            self.log(f"风格迁移数据生成完成，共生成 {len(style_transfer_data)} 条数据")
        else:
            self.log("未生成任何风格迁移数据")
        
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
    
    def _extract_role_responses(self, conversation: str) -> list:
        """从对话中提取指定角色的回答"""
        responses = []
        lines = conversation.split('\n')
        
        for line in lines:
            if line.strip() and ':' in line:
                role_part, content_part = line.split(':', 1)
                role = role_part.strip()
                content = content_part.strip()
                
                # 检查是否是目标角色（支持角色名称包含目标角色名的情况）
                if self.role in role or role in self.role:
                    responses.append(content)
        
        return responses 