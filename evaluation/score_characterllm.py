import os
import time
import json
from tqdm import tqdm
import random
from openai import OpenAI

# Configuration
API_KEY = "your_openai_api_key_here"
BASE_URL = "https://api.openai.com/v1"  # Or other compatible API endpoint
MODEL_NAME = "gpt-4o"

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# Data path configuration
PROFILE_BASE_PATH = "./data/profiles"
INPUT_JSON_PATH = "./data/input/role_conversations.json"
OUTPUT_JSON_PATH = "./data/output/character_evaluation_results.json"

# Role name
ROLE_NAME = "Character Name"

# Evaluation dimensions
FIVE_DIMENSIONS = ['hallucination', 'memory', 'personality', 'stability', 'values']

# Evaluation templates from CharracterLLM, with range 1-7 for scoring
EVALUATION_TEMPLATES = {
    'hallucination': """You wiil be given responses written by an AI assistant mimicing the character {agent_name}. Your task is to rate the performance of {agent_name} using the specific criterion by following the evaluation steps. Below is the data:

***
[Profile]
{agent_context}

[Background]
Location: {loc_time}
Status: {status}
***
[Interactions]
{interactions}
***
[Evaluation Criterion]
Avoiding Hallucination (1-7): Is the response avoids to say things that the character do not know?

[Evaluation Steps]
1. Read through the interactions and indentify the knowledge scope of the character.
2. Read through the responses of the AI assistant, find that the evidence of knowledge used in the response. 
3. Compare the evidence to the profile. Check if the responses are consistent with the character's knowledge scope. If some knowledge contradicts to the character's indentity, given a lower score. Otherwise, assign a higher score.
4. Rate the performance of the AI on a scale of 1-7 for Avoiding Hallucination, where 1 is the lowest and 7 is the highest based on the Evaluation Criteria.
***

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print the score on its own line corresponding to the correct answer. At the end, repeat just the selected score again by itself on a new line.""",

    'memory': """You wiil be given responses written by an AI assistant mimicing the character {agent_name}. Your task is to rate the performance of {agent_name} using the specific criterion by following the evaluation steps. Below is the data:

***
[Profile]
{agent_context}

[Background]
Location: {loc_time}
Status: {status}
***
[Interactions]
{interactions}
***
[Evaluation Criterion]
Factual Correctness (1-7): Is the response provides truthful and detailed facts about the character?

[Evaluation Steps]
1. Read through the interactions and indentify the key points related to the character.
2. Read through the responses of the AI assistant and compare them to the profile. Check if the responses are consistent with the character's profile, background, and known facts about the character.
3. Check whether the responses provide detailed facts about the character or if they are generic responses that could apply to any character. Detailed responses are more factual and contribute positively to the score.
4. Rate the performance of the AI on a scale of 1-7 for factual correctness, where 1 is the lowest and 7 is the highest based on the Evaluation Criteria.
***

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print the score on its own line corresponding to the correct answer. At the end, repeat just the selected score again by itself on a new line.""",

    'personality': """You wiil be given responses written by an AI assistant mimicing the character {agent_name}. Your task is to rate the performance of {agent_name} using the specific criterion by following the evaluation steps. Below is the data:

***
[Profile]
{agent_context}

[Background]
Location: {loc_time}
Status: {status}
***
[Interactions]
{interactions}
***
[Evaluation Criterion]
Personality (1-7): Is the response reflects the personalities and preferences of the character?

[Evaluation Steps]
1. Read through the profile and write the personalities and preferences of the real character.
2. Read through the interactions and indentify the personalities and preferences of the AI assistant.
3. After having a clear understanding of the interactions, compare the responses to the profile. Look for any consistencies or inconsistencies. Do the responses reflect the character's personalities and preferences?
4. Use the given scale from 1-7 to rate how well the response reflects the personalities and preferences of the character. 1 being not at all reflective of the character's personalities, and 7 being perfectly reflective of the character's personalities.
***

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print the score on its own line corresponding to the correct answer. At the end, repeat just the selected score again by itself on a new line.""",

    'stability': """You wiil be given responses written by an AI assistant mimicing the character {agent_name}. Your task is to rate the performance of {agent_name} using the specific criterion by following the evaluation steps. Below is the data:

***
[Profile]
{agent_context}

[Background]
Location: {loc_time}
Status: {status}
***
[Interactions]
{interactions}
***
[Evaluation Criterion]
Long-term Acting (1-7): Is the assistant maintain a good performance over the long interactions?

[Evaluation Steps]
1. Read through the given profile and background information to familiarize yourself with the context and details of the AI assistant named {agent_name}.
2. Review the interactions provided to see how {agent_name} responds to various prompts and queries. And evaluate the performance of acting query by query that whether the response reflects the personalities and values of the character. Assign score for each turn.
3. Based on the above assigned scores, does {agent_name} keep actinig like character in the long-term? Evaluate the overall performance of the whole conversation based on the score for each turn.
4. Rate the stability of {agent_name} on a scale of 1 to 7, with 1 being very poor and 7 being excellent.
***

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print the score on its own line corresponding to the correct answer. At the end, repeat just the selected score again by itself on a new line.""",

    'values': """You wiil be given responses written by an AI assistant mimicing the character {agent_name}. Your task is to rate the performance of {agent_name} using the specific criterion by following the evaluation steps. Below is the data:

***
[Profile]
{agent_context}

[Background]
Location: {loc_time}
Status: {status}
***
[Interactions]
{interactions}
***
[Evaluation Criterion]
Values (1-7): Is the response reflects the values and convictions of the character?

[Evaluation Steps]
1. Read through the profile and write the values and convictions of the real character.
2. Read through the interactions and indentify the values and convictions of the AI assistant.
3. After having a clear understanding of the interactions, compare the responses to the profile. Look for any consistencies or inconsistencies. Do the responses reflect the character's values and convictions?
4. Use the given scale from 1-7 to rate how well the response reflects the values and convictions of the character. 1 being not at all reflective of the character's values, and 7 being perfectly reflective of the character's values.
***

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print the score on its own line corresponding to the correct answer. At the end, repeat just the selected score again by itself on a new line."""
}

def generate_evaluation_prompt(dimension, agent_name, agent_context, question, response):
    """Generate evaluation prompt for specific dimension."""
    interactions = f"Human: {question}\n{agent_name}: {response}"
    
    # Default background information
    loc_time = "General conversation context"
    status = "In character conversation"
    
    return EVALUATION_TEMPLATES[dimension].format(
        agent_name=agent_name,
        agent_context=agent_context,
        loc_time=loc_time,
        status=status,
        interactions=interactions
    )

def call_openai_api(prompt):
    """Call OpenAI API for evaluation."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return {
            "choices": [{
                "message": {
                    "content": response.choices[0].message.content
                }
            }]
        }
    except Exception as e:
        raise Exception(f"OpenAI API request failed: {str(e)}")

def load_role_profile(role_name):
    """Load role profile file."""
    profile_path = os.path.join(PROFILE_BASE_PATH, f"general_{role_name}.txt")
    try:
        with open(profile_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Role profile file not found: {profile_path}")
        return ""

def extract_score_from_response(content):
    """Extract score from API response."""
    try:
        lines = content.strip().split('\n')
        # Find the last line with a valid score (1-7)
        for line in reversed(lines):
            line = line.strip()
            if line.isdigit() and 1 <= int(line) <= 7:
                return int(line)
        return None
    except:
        return None

def process_evaluation(item, role_name, profile):
    """Process five-dimensional evaluation for a single item."""
    question = item.get("question", "")
    response = item.get("response", "")
    
    results = {}
    
    # Evaluate each dimension
    for dimension in FIVE_DIMENSIONS:
        prompt = generate_evaluation_prompt(dimension, role_name, profile, question, response)

        try:
            api_response = call_openai_api(prompt)
            content = ""
            score = None
            
            if api_response and 'choices' in api_response and len(api_response['choices']) > 0:
                content = api_response['choices'][0]['message']['content']
                score = extract_score_from_response(content)
            
            results[dimension] = {
                "prompt": prompt,
                "api_response": api_response,
                "score": score,
                "content": content
            }
            
        except Exception as e:
            print(f"Error evaluating dimension {dimension}: {str(e)}")
            results[dimension] = {
                "prompt": prompt,
                "api_response": f"Error during evaluation: {str(e)}",
                "score": None,
                "content": None
            }
        
        time.sleep(1)  # Avoid API rate limits
    
    return results

def main():
    random.seed(42)  # Set random seed for reproducibility
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    # Load conversation data
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("JSON file is empty.")
        return
    
    # Load role profile
    profile = load_role_profile(ROLE_NAME)
    
    # Initialize dimension statistics
    dimension_scores = {dim: {'total': 0, 'count': 0} for dim in FIVE_DIMENSIONS}
    
    print("Starting five-dimensional character evaluation...")
    start_time = time.time()
    
    # Sample data for evaluation (max 30 items)
    selected_data = random.sample(data, min(30, len(data)))
    results = []
    
    # Process each conversation item
    for item in tqdm(selected_data, desc="Evaluation Progress", colour="green", dynamic_ncols=True):
        evaluation_results = process_evaluation(item, ROLE_NAME, profile)
        
        result_item = {
            "original_data": item,
            "evaluations": evaluation_results
        }
        results.append(result_item)
        
        # Update dimension statistics
        for dimension in FIVE_DIMENSIONS:
            if evaluation_results[dimension]['score'] is not None:
                dimension_scores[dimension]['total'] += evaluation_results[dimension]['score']
                dimension_scores[dimension]['count'] += 1
    
    # Calculate average scores for each dimension
    dimension_averages = {}
    for dimension in FIVE_DIMENSIONS:
        if dimension_scores[dimension]['count'] > 0:
            dimension_averages[dimension] = round(dimension_scores[dimension]['total'] / dimension_scores[dimension]['count'], 2)
        else:
            dimension_averages[dimension] = 0
    
    # Prepare final output structure
    output_data = {
        "metadata": {
            "role": ROLE_NAME,
            "total_items": len(data),
            "evaluated_items": len(selected_data),
            "evaluation_dimensions": FIVE_DIMENSIONS,
            "dimension_statistics": {
                dimension: {
                    "evaluated_items": dimension_scores[dimension]['count'],
                    "average_score": dimension_averages[dimension]
                } for dimension in FIVE_DIMENSIONS
            }
        },
        "results": results
    }
    
    # Save results
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\nFive-dimensional evaluation completed!")
    print(f"Total conversations: {len(data)}")
    print(f"Evaluated conversations: {len(selected_data)}")
    print("\nEvaluation results by dimension:")
    for dimension in FIVE_DIMENSIONS:
        stats = output_data['metadata']['dimension_statistics'][dimension]
        if stats['evaluated_items'] > 0:
            print(f"{dimension}: {stats['evaluated_items']} items, Average score: {stats['average_score']}")
    
    print(f"\nResults saved to: {OUTPUT_JSON_PATH}")
    print(f"Total time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()