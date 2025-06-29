# H-RPA: A Unified Framework for Hallucination-Reasoning Role-Playing Agent

This repository contains the official author's implementation and datasets associated with the paper **"H-RPA: A Unified Framework for Hallucination-Reasoning Role-Playing Agent"**.

<p align="center">
    <a href="https://bqzz-bbnc.github.io/H-RPA/"><strong>Homepage</strong></a> |
    <a href="#"><strong>Paper (Coming Soon)</strong></a>
</p>

-----

## 📝 Overview

Role-playing with large language models (LLMs) have gained significant attention due to their potential in simulation, and dialogue systems. However, existing methods primarily rely on single-stage fine-tuning or retrieval-augmented generation, which over-depends on the model’s one-stage output capability and dataset quality, lacking systematic reasoning to address hallucination issues caused by complex character scenarios (e.g., temporal inconsistencies or factual errors). To address these issues, we propose an agent framework, **H-RPA (Hallucination-Reasoning Role-Playing Agent)**, which integrates Chain-of-Thought (CoT) reasoning to mitigate hallucinations and structures the role-playing process into a coherent three-stage model architecture: perception, planning, and action. Specifically, in the perception stage, the agent retrieves character-related knowledge via a RAG module to form a reasoning foundation. In the reasoning stage, Role-Aware CoT process analyzes entities, events, and their spatiotemporal relationships, ensuring both optimal knowledge use and logical consistency. Finally, in the action stage, the agent generates responses that reflect the character's tone and style. Experimental results show that our framework outperforms existing methods on the role-playing benchmark dataset, offering a unified solution for controlled character generation.

## 🚀 Open Source Plan

Our release is planned in three stages to progressively share our work with the community.

  - [x] **Stage 1: Example Datasets & Evaluation Code (Available Now)**

      - We have released two sample datasets to demonstrate the data format and content, along with our evaluation code (`score_hrpa`). See the Datasets section for more details.

  - [ ] **Stage 2: Full Code Release (Coming Soon)**

      - The complete training, inference, and **dataset construction code** for the H-RPA framework will be open-sourced soon.

  - [ ] **Stage 3: Full Release (Upon Paper Acceptance)**

      - Once our paper is officially accepted, we will release the complete dataset used in our experiments and the final model weights.

## 📚 Datasets

Our datasets are constructed based on the [RoleAgentBench](https://huggingface.co/datasets/RoleAgent/RoleAgentBench) and are located in the `/datasets` directory. The datasets include two main components:

  * **RAB-QA**: Question-Answer pairs designed for role-playing dialogue.
  * **RAB-CoT**: Detailed Chain-of-Thought data that outlines the reasoning process of the agent.

The full collection will eventually contain five Chinese and five English datasets. As part of our initial release, we have open-sourced the following two examples: 家有儿女 (Home with Kids) and Harry Potter.

The complete set of datasets will be made available in subsequent releases. Stay tuned\!

## 🛠️ Requirements & Installation

Our implementation is built on top of the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

For installation and environment setup, please refer to the official [LLaMA-Factory installation guide](https://github.com/hiyouga/LLaMA-Factory#installation).

## ⚙️ Usage

### Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 USE_RAY=1 llamafactory-cli train configs/train_lora/glm4_lora_sft_ray_cot.yaml
```

### Inference

```bash
llamafactory-cli cot 
```

## 📜 License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for more details.
