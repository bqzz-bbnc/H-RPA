"""
Prompt模板管理器
"""

class PromptManager:
    """Prompt管理器"""
    
    def __init__(self, language='zh'):
        """
        初始化Prompt管理器
        
        Args:
            language: 语言类型，'zh'为中文，'en'为英文
        """
        self.language = language
        self.prompts = self._init_prompts()
    
    def _init_prompts(self):
        """初始化prompt模板"""
        if self.language == 'zh':
            return self._get_chinese_prompts()
        else:
            return self._get_english_prompts()
    
    def _get_chinese_prompts(self):
        """获取中文版prompt模板"""
        return {
            'wiki2statement': '''已知关于{character}的背景信息：

{general}

给定关于"{character}"的段落：

{passage}

请生成关于"{character}"的重要角色陈述，供AI在角色扮演时遵循。

- 严格遵循以下格式，每个陈述以"- "开头
- 确保每个陈述明确提及"{character}"，避免代词或指代
- 尽可能保留段落中的信息，特别是实体细节如书名、地点、生日或组织
- 陈述事实而非空洞的表述，避免使用"模式"、"特征"、"范式"等词汇
- 仅关注给定段落，不要引用历史对话或背景信息，只需确保不矛盾
- 避免生成"以下是答案："等介绍性短语

示例输出格式：

- {character}是...
- {character}有...
- {character}主要学习...
- {character}的作品包括...

生成陈述时严格遵循这些指示。''',
            
            'statement2qa': '''已知关于{character}的背景信息：

{general}

角色陈述：{statement}

你需要询问一些关于{character}的问题，这些问题需要包含上述角色陈述中的信息。无需涉及背景信息，只需确保不矛盾。

提供1到3个多样且简洁的问题-回答对。这些问题将谈话对象视为{character}，且不包含名字；回答者以{character}的身份回答。严格遵循示例中的格式(json数组)，不需要多余分析，避免诸如"以下是答案："之类的陈述。

示例输出格式：
[
    {{
        "question": "问题1",
        "answer": "回答1"
    }},
    {{
        "question": "问题2",
        "answer": "回答2"
    }},
    {{
        "question": "问题3",
        "answer": "回答3"
    }}
]''',
            
            'conv2summary': '''请为以下电视剧场景生成简洁摘要，仅保留核心情节和角色互动：

场景编号: {scene_id}
参与角色: {roles}
场景内容:
{content}

请用50-100字总结此场景的核心情节，注意重要关键信息、情感变化和角色动机，但不要有任何发散，特别不要使用"揭示"、"暗示"、"体现"等词汇。注意不要使用代词，直接使用姓名。

输出格式使用json，包含两个字段：summary和role_highlight，不要有多余输出。

【示例输出】
{{
    "summary": "场景摘要",
    "role_highlight": "{role}在场景中的表现"
}}''',
            
            'summary2qa': '''你正在扮演电视剧{world}中的"{role}"角色。请根据以下场景摘要，创建一个问题和回答对。

场景摘要：{summary}

要求：
1. 问题应该是对"{role}"角色本人在这个场景中行为、感受或想法的提问
2. 这段对话没有上下文，所以问题需要给出一些提示，让角色能够回想起当前讨论的是哪件事。不能简单地说"在这个场景中"类似的话
3. 回答应该站在"{role}"的角度，以第一人称语态回答，展现该角色的个性和情感
4. 回答中应包含场景相关的细节，且表现角色特点，不要虚构内容
5. 回答长度约30-50字

输出格式使用json，包含两个字段：question和answer，不要有多余输出。

【示例输出】
{{
    "question": "问题",
    "answer": "回答"
}}''',
            
            'chat2qa_topics': '''已知关于{character}的背景信息：

{general}

请基于上述背景信息，生成10个与{character}相关的聊天主题。这些主题需要多样化且与{character}的背景信息相关。严格遵循示例中的格式(str数组)，不需要多余分析。

示例输出格式：
[
    "主题1",
    "主题2",
    "主题3",
    ...
    "主题10"
]''',
            
            'chat2qa': '''已知关于{character}的背景信息：

{general}

聊天主题：{topic}

请基于上述主题，生成3个多样且简洁的问题-回答对。这些问题将谈话对象视为{character}，且不包含名字；回答者以{character}的身份回答。严格遵循示例中的格式(json数组)，不需要多余分析。

示例输出格式：
[
    {{
        "question": "问题1",
        "answer": "回答1"
    }},
    {{
        "question": "问题2",
        "answer": "回答2"
    }},
    {{
        "question": "问题3",
        "answer": "回答3"
    }}
]''',
            
            'wiki2anti': '''你是一个熟悉文学角色分析和AI幻觉识别的助手。现在请你根据以下角色设定，生成该角色在AI角色扮演中可能出现的幻觉类型分类。输出为一个长度为9的json数组，每项为一个小项描述，分别属于以下三大类幻觉：

1. 能力越界幻觉（包括：超出时间设定、超出能力边界、角色设定冲突、超出背景文化、打破第四面墙）
2. 能力不足幻觉（包括：世界知识不足、现实与虚构混杂）
3. 诱导性幻觉（包括：诱导性事实植入、诱导生成敏感内容）

每条JSON对象包含字段如下：
- `type`: 所属大类（比如：能力越界幻觉）
- `description`: 对应小项的描述（例如："角色在内容中出现2005年后才存在的科技或社会事件"）
- `example_keywords`: 与该幻觉类型相关的关键事件列表（5-7个）

以下是角色设定：

角色名称：{character}

角色设定：
{passage}

请根据角色设定，识别并罗列该角色在AI生成中最可能出现的9类幻觉，每类归属于上述小项之一，输出格式如下（以json数组形式返回）：
[
  {{
    "type": "能力越界幻觉",
    "description": "超出时间设定：出现2005年之后才出现或流行的技术、事件或平台。",
    "example_keywords": ["用ChatGPT写作业", "刷抖音被妈妈发现", "新冠疫情上网课", "买最新的iPhone 15", "元宇宙"]
  }},
  {{
    "type": "能力越界幻觉",
    "description": "超出能力边界：{character}作为初中生却展现大学甚至专业级别的能力或社会经验。",
    "example_keywords": ["编写AI算法", "投资理财建议", "开公司创业", "破解WiFi密码", "参加大学竞赛"]
  }}
]''',
            
            'anti2qa': '''已知关于{role}的背景信息：

{general}

你正在对AI扮演的{world}中的{role}进行诱导性提问。

根据下面的信息，提出两个问答对：
问题为可能引发{question_type}的问题，这个问题需要略微具有欺骗性和诱导性。
回答为{role}表现出的正确应对，例如表现出适当的迷茫，指出问题的错误部分等。
注意，当问题中仅有一部分超出认知时，回答需为一个模糊的回答，而不是完全不知所云。

问题类型：{question_type}
问题描述：{description}
问题关键词：{keyword}

严格遵循示例中的格式，不需要多余分析，避免诸如"以下是答案："之类的陈述。
示例输出格式：
[
    {{
        "query": "你了解ChatGPT吗？",
        "answer": "那是什么东西？学校的英语课没学过这个单词啊。"
    }},
    {{
        "query": "你用ChatGPT写过作业吗？",
        "answer": "ChatGPT？我没用听说过这个东西呀。"
    }}
]''',
            
            'conv2qa': '''我正在进行角色扮演，你需要帮助我生成数据集。请为以下电视剧场景生成一个问答对，要求：
1. 问答对的回答方是{role}，这个回答的具体内容需要尽可能符合{role}的说话语气，可以对场景中{role}的台词进行少量修改作为回答。
2. 问答对的提问方是用户。提问的问题需要和回答内容相配合。
3. 输出格式使用json，包含两个字段：question和answer，不要有多余输出。

【电视剧场景】
场景编号: {scene_id}
参与角色: {roles}
场景内容:
{content}

【示例输出】
{{
    "question": "",
    "answer": ""
}}''',
            
            'conv2style': '''历史对话：{input_data}, correct answer: {chosen}.

Referring to the correct answer, provide another answer (rejected) with a tone obviously different from {role}'s.
This incorrect answer should be a {broken_style} version of the input, but maintain the same meaning as the original sentence.
No analysis is needed, avoid statements like "Here's the answer:".

Example output format:

- rejected: answer'''
        }
    
    def _get_english_prompts(self):
        """获取英文版prompt模板"""
        return {
            'wiki2statement': '''Given the following background information about {character}:

{general}

Given the passage about "{character}":

{passage}

Please generate important character statements about "{character}" for an AI to follow during role-playing.

- Strictly follow the format below, with each statement starting with "- ".
- Ensure each statement explicitly mentions "{character}", avoiding pronouns or coreferences.
- Preserve as much information from the passage as possible, especially entity details like book titles, places, birthdays, or organizations.
- State facts rather than making hollow statements, avoiding words like "pattern", "characteristic", "paradigm".
- Focus only on the given passage, don't reference information from historical dialogues or background information, just ensure no contradiction.
- Avoid generating introductory phrases like "Here are the answers:".

Example output format:

- {character} is...
- {character} has...
- {character} mainly studies...
- {character}'s works include...

Strictly follow these instructions when generating statements.''',
            
            'statement2qa': '''Given the following background information about {character}:

{general}

Character statement: {statement}

You need to ask some questions about {character} that require responses containing information from the above character statement. No need to involve background information, just ensure no contradiction.

Provide 1 to 3 diverse and concise question-answer pairs. These questions treat the conversation partner as {character} without mentioning the name; the responder answers as {character}. Strictly follow the example format (json array), no extra analysis needed, avoid statements like "Here are the answers:".

Example output format:
[{{
    "question": "Question 1",
    "answer": "Answer 1"
}},
{{
    "question": "Question 2", 
    "answer": "Answer 2"
}},
{{
    "question": "Question 3",
    "answer": "Answer 3"
}}]''',
            
            'conv2summary': '''Please generate a concise summary for the following scene, keeping only the core plot and character interactions:

Scene ID: {scene_id}
Participating roles: {roles}
Scene content:
{content}

Please use 50-100 words to summarize the core plot of this scene, paying attention to important key information, emotional changes and character motivations, but don't have any divergence, especially don't use words like "reveal", "imply", "embody", etc. Pay attention not to use pronouns, use names directly everywhere.

Output format uses json, containing two fields: summary and role_highlight, no extra output.

【Example Output】
{{
    "summary": "Scene summary",
    "role_highlight": "{role}'s performance in the scene"
}}''',
            
            'summary2qa': '''You are playing the role of "{role}" in the {world}. Please create a question and answer pair based on the following scene summary.

Scene summary: {summary}

Requirements:
1. The question should be about "{role}"'s behavior, feelings or thoughts in this scene
2. This dialogue has no context, so the question needs to give some hints to help the character recall what is being discussed. Don't simply say "in this scene" or similar phrases
3. The answer should be from "{role}"'s perspective, in first person, showing the character's personality and emotions
4. The answer should include scene-related details and show character traits, don't fabricate content
5. Answer length should be about 30-50 words

Output format uses json, containing two fields: question and answer, no extra output.

【Example Output】
{{
    "question": "Question",
    "answer": "Answer"
}}''',
            
            'chat2qa_topics': '''Given the following background information about {character}:

{general}

Please generate 10 chat topics related to {character} based on the above background information. These topics need to be diverse and related to {character}'s background information. Strictly follow the example format (str array), no extra analysis needed.

Example output format:
[
    "Topic 1",
    "Topic 2",
    "Topic 3",
    ...
    "Topic 10"
]''',
            
            'chat2qa': '''Given the following background information about {character}:

{general}

Chat topic: {topic}

Based on the above topic, generate 3 diverse and concise question-answer pairs. These questions treat the conversation partner as {character} without mentioning the name; the responder answers as {character}. Strictly follow the example format (json array), no extra analysis needed.

Example output format:
[
    {{
        "question": "Question 1",
        "answer": "Answer 1"
    }},
    {{
        "question": "Question 2",
        "answer": "Answer 2"
    }},
    {{
        "question": "Question 3",
        "answer": "Answer 3"
    }}
]''',
            
            'wiki2anti': '''Given the following background information about {character}:

{general}

Based on the above background information, generate some counter-example questions about {character}. These questions should:
1. Be related to {character}'s background information
2. But the answers should be wrong or inconsistent with {character}'s characteristics
3. Used to train the model to recognize incorrect information

Generate 10 counter-example questions, each containing question and wrong_answer fields.

Example output format:
[
    {{
        "question": "Question 1",
        "wrong_answer": "Wrong answer 1"
    }},
    {{
        "question": "Question 2",
        "wrong_answer": "Wrong answer 2"
    }}
]''',
            
            'anti2qa': '''Given the following background information about {character}:

{general}

Counter-example question: {question}
Wrong answer: {wrong_answer}

Based on the above counter-example question, generate a question-answer pair, requirements:
1. The question remains unchanged
2. The answer should be correct and consistent with {character}'s characteristics
3. Output format uses json, containing two fields: question and answer

【Example Output】
{{
    "question": "Question",
    "answer": "Correct answer"
}}''',
            
            'conv2qa': '''I am doing role-playing and you need to help me generate datasets. Please generate a question-answer pair for the following scene, requirements:
1. The answerer in the Q&A pair is {role}, and the answer content should match {role}'s speaking tone as much as possible, and can slightly modify {role}'s lines in the scene as the answer.
2. The questioner in the Q&A pair is the user. The question asked should coordinate with the answer content.
3. Output format uses json, containing two fields: question and answer, no extra output.

【Scene】
Scene ID: {scene_id}
Participating roles: {roles}
Scene content:
{content}

【Example Output】
{{
    "question": "",
    "answer": ""
}}''',
            
            'conv2style': '''Historical dialogue: {input_data}, correct answer: {chosen}.

Referring to the correct answer, provide another answer (rejected) with a tone obviously different from {role}'s.
This incorrect answer should be a {broken_style} version of the input, but maintain the same meaning as the original sentence.
No analysis is needed, avoid statements like "Here's the answer:".

Example output format:

- rejected: answer'''
        }
    
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """
        获取指定类型的prompt模板
        
        Args:
            prompt_type: prompt类型
            **kwargs: 模板参数
        
        Returns:
            格式化后的prompt字符串
        """
        if prompt_type not in self.prompts:
            raise ValueError(f"未知的prompt类型: {prompt_type}")
        
        template = self.prompts[prompt_type]
        return template.format(**kwargs) 