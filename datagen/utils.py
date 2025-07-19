"""
工具函数模块
包含配置加载、路径处理、文件操作等通用功能
"""

import os
import json
import yaml
import random
from typing import Dict, List, Any, Optional
from pathlib import Path


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化配置"""
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_path(self, *path_keys: str) -> str:
        """获取完整路径"""
        path_parts = []
        for path_key in path_keys:
            value = self.get(path_key)
            if value is None:
                raise ValueError(f"配置键不存在: {path_key}")
            path_parts.append(value)
        
        return os.path.join(*path_parts)


class PathManager:
    """路径管理类"""
    
    def __init__(self, config: Config):
        """初始化路径管理器"""
        self.config = config
        self.s1e1_worlds = set(config.get('s1e1_worlds', []))
    
    def get_roleagentbench_path(self, world: str, filename: str) -> str:
        """
        获取RoleAgentBench数据集中的文件路径
        
        Args:
            world: 世界名称
            filename: 文件名（如 scene_summary.json）
        
        Returns:
            完整文件路径
        """
        root = self.config.get('paths.roleagentbench_root')
        
        # 检查是否需要添加S1E1后缀
        if world in self.s1e1_worlds:
            world_path = f"{world} S1E1"
        else:
            world_path = world
        
        return os.path.join(root, world_path, "raw", filename)
    
    def get_input_path(self, world: str, filename: str) -> str:
        """
        获取输入文件路径
        
        Args:
            world: 世界名称
            filename: 文件名（如 scene_summary.json, character_profile.json）
        
        Returns:
            完整输入文件路径
        """
        return self.get_roleagentbench_path(world, filename)
    
    def get_local_input_path(self, *path_parts: str) -> str:
        """
        获取本地输入文件路径
        
        Args:
            *path_parts: 路径部分
        
        Returns:
            完整本地输入文件路径
        """
        input_base = self.config.get('paths.input_base', 'input')
        return os.path.join(input_base, *path_parts)
    
    def get_profile_path(self, world: str, role: str) -> str:
        """
        获取角色配置文件路径
        
        Args:
            world: 世界名称
            role: 角色名称
        
        Returns:
            角色配置文件路径
        """
        root = self.config.get('paths.roleagentbench_root')
        
        # 检查是否需要添加S1E1后缀
        if world in self.s1e1_worlds:
            world_path = f"{world} S1E1"
        else:
            world_path = world
        
        return os.path.join(root, world_path, "profiles", f"{role}.jsonl")
    
    def get_scene_summary_path(self, world: str) -> str:
        """
        获取场景摘要文件路径
        
        Args:
            world: 世界名称
        
        Returns:
            场景摘要文件路径
        """
        return self.get_roleagentbench_path(world, "scene_summary.json")
    
    def get_output_path(self, *path_parts: str) -> str:
        """获取输出路径"""
        output_base = self.config.get('paths.output_base')
        return os.path.join(output_base, *path_parts)
    
    def get_style_path(self, world: str, role: str) -> str:
        """
        获取风格迁移数据输出路径
        
        Args:
            world: 世界名称
            role: 角色名称
        
        Returns:
            风格迁移数据输出路径
        """
        output_base = self.config.get('paths.output_base')
        return os.path.join(output_base, "style", f"{world}_{role}_style.json")
    
    def ensure_dir(self, path: str):
        """确保目录存在"""
        os.makedirs(path, exist_ok=True)


def load_json(file_path: str) -> List[Dict]:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict], file_path: str):
    """保存JSON文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """保存JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def shuffle_data(data: List[Dict], seed: int = 42) -> List[Dict]:
    """打乱数据"""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    return shuffled


def split_train_test(data: List[Dict], split_ratio: float = 0.8, seed: int = 42) -> tuple:
    """切分训练集和测试集"""
    random.seed(seed)
    shuffled = shuffle_data(data, seed)
    split_idx = int(len(shuffled) * split_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def get_file_count(data: List[Dict]) -> int:
    """获取数据条数"""
    return len(data)


def format_filename(world: str, role: str, data_type: str, count: int) -> str:
    """格式化文件名"""
    return f"{world}_{role}_{data_type}_{count}.json" 