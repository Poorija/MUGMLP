import json
import torch
import asyncio
from sqlalchemy.orm import Session
from database import SessionLocal
import crud
from websocket_manager import manager
from hardware_scanner import scanner

# Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

def train_moe_model_task(model_id: int, dataset_id: int, model_info: dict):
    db: Session = SessionLocal()
    try:
        # 1. Hardware Check
        hw_info = scanner.get_hardware_info()
        # MoE models (like Mixtral) are huge. We set a very high VRAM requirement.
        # or use a smaller MoE like Qwen-MoE or Switch-Base if available.
        feasibility = scanner.check_feasibility({
            "min_vram_gb": 24.0,
            "requires_gpu": True
        })

        if not feasibility["feasible"]:
             error_msg = f"Hardware Insufficient for MoE: {'; '.join(feasibility['errors'])}"
             crud.update_model_status(db, model_id, "failed")
             asyncio.run(manager.broadcast(json.dumps({
                "model_id": model_id,
                "status": "failed",
                "error": error_msg
             })))
             return

        crud.update_model_status(db, model_id, "running")
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "started"})))

        # 2. Load Data
        dataset_record = crud.get_dataset(db, dataset_id)
        file_location = f"uploads/{dataset_record.filename}"

        # Assume text dataset for Causal LM
        dataset = load_dataset("text", data_files={"train": file_location})
        train_ds = dataset["train"].train_test_split(test_size=0.1)["train"]

        # 3. Model
        # Using a placeholder or smaller MoE for compatibility if possible
        # Realistically, "mistralai/Mixtral-8x7B-v0.1"
        model_name = "openlm-research/open_llama_3b_v2" # Placeholder - MoE checkpoints are huge.
        # Note: To truly use MoE, one would load "mistralai/Mixtral-8x7B-v0.1"
        # handled transparently by AutoModelForCausalLM.

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )

        # Apply QLoRA / LoRA
        hyperparams = model_info.get('hyperparameters', {})
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"] # Standard attention targets
        )
        model = get_peft_model(model, peft_config)

        # 4. Training
        training_args = TrainingArguments(
            output_dir=f"ml_models/moe_{model_id}",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=20, # Demo
            learning_rate=2e-4,
            fp16=True,
            logging_steps=5,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            tokenizer=tokenizer,
        )

        trainer.train()

        save_path = f"ml_models/model_{model_id}"
        trainer.save_model(save_path)

        crud.update_model_status(db, model_id, "completed", model_path=save_path)
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "completed"})))

    except Exception as e:
        crud.update_model_status(db, model_id, "failed")
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "failed", "error": str(e)})))
    finally:
        db.close()
