"""
对话转摘要处理器
从对话数据生成摘要
"""

import os
import json
import random
import time
from tqdm import tqdm
import re

from .base_processor import BaseProcessor
from utils import load_json, save_json, format_filename, get_file_count



class Conv2SummaryProcessor(BaseProcessor):
    """对话转摘要处理器"""
    

    
    def process(self):
        """从对话数据生成摘要"""
        self.log("开始从对话数据生成摘要...")
        
        # 输入输出路径
        conversation_path = self.path_manager.get_profile_path(self.world, self.role)
        output_path = self.path_manager.get_output_path("process", "summary", f"{self.world}_{self.role}_summary.json")
        
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
        
        # 用于存储生成的摘要
        all_summaries = []
        
        # 从对话中生成摘要
        self.log(f"开始生成摘要，共 {len(conversation_data)} 个对话场景...")
        
        # 在示例模式下限制对话场景数量
        conversation_data = self.limit_data_for_demo(conversation_data)
        
        for scene_id, conversation in enumerate(tqdm(conversation_data, desc=f"处理 {self.role} 的对话")):
            if not conversation:
                continue
            
            # 使用完整的prompt模板
            prompt = self.get_prompt("conv2summary", role=self.role, scene_id=scene_id, roles=", ".join([self.role]), content=conversation)
            
            # 调用API生成摘要
            try:
                # 将prompt转换为messages格式
                messages = [{"role": "user", "content": prompt}]
                response = self.call_api(messages, temperature=0.8)
                summary = response.strip()
                
                if summary:
                    summary_item = {
                        "conversation": conversation,
                        "summary": summary
                    }
                    all_summaries.append(summary_item)
                
            except Exception as e:
                self.log(f"生成摘要时出错: {e}")
                continue
        
        # 保存摘要数据
        if all_summaries:
            self.log(f"保存摘要数据: {output_path}")
            save_json(all_summaries, output_path)
            self.log(f"摘要生成完成，共生成 {len(all_summaries)} 个摘要")
        else:
            self.log("未生成任何摘要")
        
        return True
    
    def _load_conversation_data(self, file_path: str) -> list:
        """加载对话数据"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except Exception as e:
            self.log(f"加载对话数据失败: {e}")
        return data 