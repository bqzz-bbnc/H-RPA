# 角色扮演数据生成器

H-RPA中用于生成高质量角色扮演训练数据的工具，支持中英文双版本

## 数据流程

### 主要数据流
1. **Wiki数据** → 角色陈述 → 问答对 (`qa_statement`)
2. **对话数据** → 摘要 → 问答对 (`qa_summary`)
3. **背景信息** → 聊天问答对 (`qa_chat`)
4. **Wiki数据** → 反例问题 → 反例问答对 (`qa_anti`)
5. **对话数据** → 直接问答对 (`qa_conv`)
6. **所有QA数据** → 合并 (`qa_all`) → 训练/测试集切分

### 风格迁移数据流
- **对话数据** → 风格迁移训练数据 (`style`) - 独立存放，不参与QA合并

## 使用方法

### 1. 环境准备
```bash
pip install openai tqdm pyyaml
```

### 2. 配置设置
编辑 `config.yaml` 文件：
```yaml
openai:
  api_key: "your-api-key"
  base_url: "https://api.openai.com/v1"

demo_mode:
  enabled: true
  max_items_per_api_call: 2

paths:
  roleagentbench_root: "/path/to/RoleAgentBench"
  input_base: "/path/to/input"
  output_base: "/path/to/output"
```

### 3. 准备输入文件
在 `input/` 目录下创建：
- `wiki/wiki_{角色名}.txt` - Wiki段落文件
- `general/general_{角色名}.txt` - 通用背景信息

### 4. 运行数据生成

#### 中文版
```bash
python main_zh.py --world "家有儿女" --role "刘星"
```

#### 英文版
```bash
python main_en.py --world "Harry_Potter" --role "Harry"
```

## Prompt模板说明

所有prompt模板都在 `prompts.py` 中定义，支持中英文版本：

- **Wiki2Statement**: 从Wiki段落生成角色陈述
- **Statement2QA**: 从陈述生成问答对
- **Conv2Summary**: 从对话生成摘要
- **Summary2QA**: 从摘要生成问答对
- **Chat2QA**: 生成聊天问答对
- **Wiki2Anti**: 生成反例问题
- **Anti2QA**: 从反例生成问答对
- **Conv2QA**: 从对话生成问答对
- **Conv2Style**: 生成风格迁移数据

## 配置说明

### 示例模式配置
- `demo_mode.enabled`: 是否启用示例模式（默认true）
- `demo_mode.max_items_per_api_call`: 每次API调用最多生成的数据条数（默认2）

### 路径配置
- `roleagentbench_root`: 数据集根目录
- `input_base`: 本地输入文件目录
- `output_base`: 输出文件根目录
- `process_dir`: 中间处理文件目录
- `qa_dir`: 问答数据目录
- `all_dir`: 合并数据目录
- `train_dir`: 训练集目录
- `test_dir`: 测试集目录

### 模型配置
- `base_model`: 基础模型（用于简单任务）
- `adv_model`: 高级模型（用于复杂任务）

### 生成配置
- `train_test_split`: 训练测试集切分比例
- `random_seed`: 随机种子
- `sleep_interval`: API调用间隔



### 快速测试
启用示例模式可以快速测试代码功能：
```yaml
demo_mode:
  enabled: true
  max_items_per_api_call: 2
```

这样每次API调用只会生成2条数据。

## 完整示例：运行刘星角色数据生成

### 1. 准备配置文件
创建 `config_example.yaml` 文件：

```yaml
# 数据生成器配置示例

# OpenAI API 配置
openai:
  api_key: "your-api-key-here"
  base_url: "https://api.openai.com/v1"
  timeout: 60
  max_retries: 3

# 模型配置
models:
  base_model: "gpt-4o-mini"
  adv_model: "gpt-4o"

# 示例模式配置（快速测试用）
demo_mode:
  enabled: true
  max_items_per_api_call: 2

# 基础路径配置
paths:
  # 数据集根目录
  roleagentbench_root: "./datasets"
  
  # 本地输入目录
  input_base: "./input"
  
  # 输出目录
  output_base: "./output"
  process_dir: "./output/process"
  qa_dir: "./output/qa"
  all_dir: "./output/all"
  train_dir: "./output/train"
  test_dir: "./output/test"

# 需要添加S1E1后缀的worlds
s1e1_worlds:
  - "家有儿女"
  - "狂飙"
  - "Friends"
  - "The Big Bang Theory"

# 世界和角色配置
worlds:
  "家有儿女":
    - "刘星"
    - "夏东海"
    - "小雨"
    - "小雪"
  "Harry_Potter":
    - "Harry"
    - "Hermione"
    - "Ron"
    - "Dumbledore"
    - "Voldemort"

# 数据生成配置
generation:
  train_test_split: 0.8
  random_seed: 42
  sleep_interval: 1
```

### 2. 运行命令

#### 使用配置文件运行
```bash
python main_zh.py --config config_example.yaml --world "家有儿女" --role "刘星"
```

#### 使用命令行参数指定API密钥
```bash
python main_zh.py --config config_example.yaml --world "家有儿女" --role "刘星" --api-key "your-actual-api-key"
```

#### 启用详细输出模式
```bash
python main_zh.py --config config_example.yaml --world "家有儿女" --role "刘星" --verbose
```

### 3. 预期输出
运行成功后，会在 `./output/` 目录下生成以下文件结构：

```
output/
├── process/          # 中间处理文件
│   ├── statement/    # 角色陈述
│   ├── summary/      # 对话摘要
│   └── anti/         # 反例问题
├── qa/              # 各类问答数据
│   ├── qa_statement/ # 陈述问答对
│   ├── qa_summary/   # 摘要问答对
│   ├── qa_chat/      # 聊天问答对
│   ├── qa_anti/      # 反例问答对
│   └── qa_conv/      # 对话问答对
├── all/             # 合并后的所有问答数据
├── train/           # 训练集
└── test/            # 测试集
```

### 4. 示例模式说明
在示例模式下：
- 每次API调用最多生成2条数据
- 适合快速测试和验证代码功能
- 生成的数据量较少，但格式完整
- 可以快速验证所有数据流程是否正常工作

要生成完整数据集，请将 `demo_mode.enabled` 设置为 `false`。 