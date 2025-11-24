import os
import json
import asyncio
import torch
from sqlalchemy.orm import Session
from .database import SessionLocal
from . import crud
from .websocket_manager import manager
from .hardware_scanner import scanner

# Transformers & TRL
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

def train_dpo_task(model_id: int, dataset_id: int, model_info: dict):
    db: Session = SessionLocal()
    try:
        # 1. Hardware Check
        hw_info = scanner.get_hardware_info()
        feasibility = scanner.check_feasibility({
            "min_vram_gb": 12.0, # DPO is heavy
            "requires_gpu": True # DPO usually needs GPU
        })

        # Strict check for DPO
        if not feasibility["feasible"]:
             error_msg = f"Hardware Insufficient for DPO: {'; '.join(feasibility['errors'])}"
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
        # Using a small base model for demo, e.g., GPT-2 or TinyLlama
        model_name = "gpt2" # Placeholder. Real DPO needs SFT model.

        model = AutoModelForCausalLM.from_pretrained(model_name)
        # In DPO, we usually don't need to load the reference model manually if we use PEFT,
        # DPOTrainer can handle the adapter on top of the base model vs the base model itself.
        # But for clarity we keep the standard flow. To save memory, we can pass None for model_ref
        # if we use PEFT, as trl will treat the initial PEFT model state as reference (implicit).

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Apply LoRA
        hyperparams = model_info.get('hyperparameters', {})
        peft_config = None
        if hyperparams.get("use_lora", True): # Default to True for DPO to be feasible
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            # We don't wrap model here manually with get_peft_model because DPOTrainer accepts peft_config
            # and handles the wrapping + reference model logic optimized.

        # 4. Training
        training_args = TrainingArguments(
            output_dir=f"ml_models/dpo_{model_id}",
            per_device_train_batch_size=2,
            max_steps=50, # Demo steps
            learning_rate=1e-5,
            remove_unused_columns=False,
            no_cuda=not torch.cuda.is_available()
        )

        dpo_trainer = DPOTrainer(
            model,
            ref_model=None, # TRL handles this when peft_config is present
            peft_config=peft_config,
            args=training_args,
            beta=0.1,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_length=512,
            max_prompt_length=256,
        )

        dpo_trainer.train()

        save_path = f"ml_models/model_{model_id}"
        dpo_trainer.save_model(save_path)

        crud.update_model_status(db, model_id, "completed", model_path=save_path)
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "completed"})))

    except Exception as e:
        crud.update_model_status(db, model_id, "failed")
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "failed", "error": str(e)})))
    finally:
        db.close()
