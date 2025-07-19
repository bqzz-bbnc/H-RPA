#!/usr/bin/env python3
"""
中文版数据生成器主入口
用于生成中文角色扮演数据集
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from generator import DataGenerator


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="中文版角色扮演数据生成器")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    parser.add_argument("--world", "-w", required=True, help="世界名称")
    parser.add_argument("--role", "-r", required=True, help="角色名称")
    parser.add_argument("--api-key", "-k", help="OpenAI API密钥")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 创建数据生成器
    generator = DataGenerator(
        world=args.world,
        role=args.role,
        config_path=args.config
    )
    
    # 如果命令行提供了API密钥，覆盖配置文件中的设置
    if args.api_key:
        # 直接修改配置字典
        keys = 'openai.api_key'.split('.')
        config_dict = generator.config._config
        for key in keys[:-1]:
            config_dict = config_dict[key]
        config_dict[keys[-1]] = args.api_key
    
    # 设置语言为中文
    generator.config._config['language'] = 'zh'
    
    # 执行数据生成流程
    try:
        print(f"开始为 {args.world} 的 {args.role} 生成中文数据集")
        
        # 执行完整的生成流程
        generator.run()
        
        print(f"数据生成完成！")
        print(f"输出目录: {generator.config.get('paths.all_dir')}")
            
    except KeyboardInterrupt:
        print("用户中断了数据生成过程")
        sys.exit(1)
    except Exception as e:
        print(f"数据生成过程中发生错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        # 打印完整的错误堆栈
        import traceback
        print("完整错误堆栈:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 