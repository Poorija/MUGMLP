import json
import asyncio
import torch
from sqlalchemy.orm import Session
from database import SessionLocal
import crud
from websocket_manager import manager
from hardware_scanner import scanner

# Transformers & TRL
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import ORPOTrainer, ORPOConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

def train_orpo_task(model_id: int, dataset_id: int, model_info: dict):
    """
    Trains a model using Odds Ratio Preference Optimization (ORPO).
    Does not require a reference model, making it memory efficient.
    """
    db: Session = SessionLocal()
    try:
        # 1. Hardware Check
        hw_info = scanner.get_hardware_info()
        feasibility = scanner.check_feasibility({
            "min_vram_gb": 10.0,
            "requires_gpu": True
        })

        if not feasibility["feasible"]:
             error_msg = f"Hardware Insufficient for ORPO: {'; '.join(feasibility['errors'])}"
             crud.update_model_status(db, model_id, "failed")
             asyncio.run(manager.broadcast(json.dumps({
                "model_id": model_id, "status": "failed", "error": error_msg
             })))
             return

        crud.update_model_status(db, model_id, "running")
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "started"})))

        # 2. Load Data
        dataset_record = crud.get_dataset(db, dataset_id)
        file_location = f"uploads/{dataset_record.filename}"

        # Load JSON/JSONL dataset with keys: prompt, chosen, rejected
        dataset = load_dataset("json", data_files=file_location, split="train")

        # 3. Model
        model_name = "gpt2" # Placeholder for demo

        # Check Flash Attention
        attn_implementation = "eager"
        if hw_info.get("supports_flash_attn"):
            attn_implementation = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation=attn_implementation
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Apply LoRA
        hyperparams = model_info.get('hyperparameters', {})
        peft_config = None
        if hyperparams.get("use_lora", True):
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )

        # 4. Training
        orpo_config = ORPOConfig(
            output_dir=f"ml_models/orpo_{model_id}",
            per_device_train_batch_size=2,
            max_steps=50, # Demo steps
            learning_rate=1e-5,
            remove_unused_columns=False,
            beta=0.1, # ORPO beta
            no_cuda=not torch.cuda.is_available()
        )

        trainer = ORPOTrainer(
            model=model,
            args=orpo_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

        trainer.train()

        save_path = f"ml_models/model_{model_id}"
        trainer.save_model(save_path)

        crud.update_model_status(db, model_id, "completed", model_path=save_path)
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "completed"})))

    except Exception as e:
        import traceback
        traceback.print_exc()
        crud.update_model_status(db, model_id, "failed")
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "failed", "error": str(e)})))
    finally:
        db.close()
