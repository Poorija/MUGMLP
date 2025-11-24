# Deep Learning & Advanced Architecture Roadmap

This document outlines the technical strategy for implementing advanced Deep Learning concepts into the platform, covering Architectures, Alignment, Efficiency, RAG, and Evaluation.

## 1. Hardware Awareness & Optimization (Implemented)
**Goal**: Automatically detect system capabilities (CPU vs. GPU) and recommend/restrict models accordingly.

*   **Implementation**:
    *   **Scanner**: `backend/hardware_scanner.py` utilizes `torch.cuda` and `psutil` to inspect:
        *   **GPU**: VRAM size, Compute Capability, Driver Version.
        *   **CPU**: Core count, Available RAM.
    *   **Logic**: The `check_feasibility` function validates requests against available hardware (e.g., blocks FlashAttention on CPU).

## 2. Advanced Architectures

### A. Vision Transformers (ViT) (Implemented)
**Dependencies**: `transformers`, `torchvision`, `pillow`
*   **Concept**: Treating images as sequences of patches (tokens).
*   **Integration**:
    *   **Input**: `.zip` files containing labeled image folders.
    *   **Model**: `ViTForImageClassification` from Hugging Face.
    *   **Training**: Fine-tuning pre-trained ImageNet checkpoints via `backend/training_vision.py`.

### B. Mamba / State Space Models (SSM) (Implemented)
**Dependencies**: `transformers` (provides both GPU/CPU paths)
*   **Concept**: Linear-time sequence modeling (O(N)) vs. Quadratic Attention (O(N^2)).
*   **Strategy**:
    *   **Implementation**: `backend/training_mamba.py` uses `MambaForCausalLM` from Hugging Face.
    *   **Hardware Fallback**: Automatically uses CUDA kernels if available, otherwise falls back to PyTorch implementation (CPU compatible but slower).

### C. Mixture of Experts (MoE)
**Dependencies**: `transformers` (Mixtral/Switch), `accelerate`
*   **Concept**: Sparse activation of parameters (only a subset of FFN layers runs per token).
*   **Integration**:
    *   Load models like `Mixtral-8x7B` via `bitsandbytes` (4-bit loading) to fit in consumer memory.
    *   **Offloading**: Use `accelerate` to offload experts to system RAM if VRAM is full.

## 3. Efficiency & PEFT (Parameter-Efficient Fine-Tuning)

### QLoRA & DoRA
**Dependencies**: `peft`, `bitsandbytes`
*   **QLoRA**: Backpropagate through a frozen, 4-bit quantized backbone into Low-Rank Adapters (LoRA).
*   **DoRA**: Decomposed LoRA for better stability.
*   **Strategy**: Wrap base models with `peft.LoraConfig` before training.

### FlashAttention-2
**Dependencies**: `flash-attn`
*   **Requirement**: Nvidia Ampere (A100, RTX 3090) or newer.
*   **Strategy**: Automatically enable via `torch_dtype=torch.float16, attn_implementation="flash_attention_2"` if hardware permits.

## 4. Alignment & Training

### DPO (Direct Preference Optimization)
**Dependencies**: `trl` (Transformer Reinforcement Learning)
*   **Concept**: Optimizing LLMs for human preference without a separate Reward Model.
*   **Data**: Requires `(prompt, chosen, rejected)` datasets.
*   **Implementation**: Use `DPOTrainer` from `trl` library.

## 5. RAG Systems (Retrieval Augmented Generation)

### Hybrid Search
*   **Components**:
    *   **Dense**: Vector Database (FAISS or ChromaDB) for semantic similarity.
    *   **Sparse**: BM25 algorithm for keyword matching (using `rank_bm25`).
    *   **Fusion**: Reciprocal Rank Fusion (RRF) to combine results.

### GraphRAG
*   **Concept**: Using Knowledge Graphs to capture structural relationships between entities.
*   **Implementation**:
    *   **Store**: Use `Neo4j` or a lightweight `NetworkX` graph for relationship mapping.
    *   **Retrieval**: Traversal of the graph (2-hop neighbors) + Vector Search.

## 6. Data & Evaluation
*   **Synthetic Data**: Use a strong teacher model (e.g., hosted GPT-4 or local Llama-3-70B) to generate instruction-response pairs.
*   **LLM-as-a-Judge**: A pipeline where a "Judge" LLM scores the output of the "Student" LLM based on a rubric.
