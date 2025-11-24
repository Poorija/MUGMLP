import os
import json
import torch
import asyncio
from sqlalchemy.orm import Session
from .database import SessionLocal
from . import crud
from .websocket_manager import manager
from .hardware_scanner import scanner

# Transformers
from transformers import AutoTokenizer, MambaForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

def train_mamba_model_task(model_id: int, dataset_id: int, model_info: dict):
    db: Session = SessionLocal()
    try:
        # 1. Hardware Check
        hw_info = scanner.get_hardware_info()
        feasibility = scanner.check_feasibility({
            "min_vram_gb": 4.0,
            "requires_gpu": False # Mamba via Transformers has CPU fallback (slow)
        })

        if not feasibility["feasible"]:
             error_msg = f"Hardware Insufficient: {'; '.join(feasibility['errors'])}"
             crud.update_model_status(db, model_id, "failed")
             asyncio.run(manager.broadcast(json.dumps({
                "model_id": model_id,
                "status": "failed",
                "error": error_msg
             })))
             return

        crud.update_model_status(db, model_id, "running")
        start_msg = {
            "model_id": model_id,
            "status": "started",
            "hardware_info": hw_info,
            "feasibility_warnings": feasibility["warnings"]
        }
        asyncio.run(manager.broadcast(json.dumps(start_msg)))

        # 2. Load Data (Text)
        dataset_record = crud.get_dataset(db, dataset_id)
        file_location = f"uploads/{dataset_record.filename}"

        # Lazy load text dataset using Hugging Face datasets
        # "text" builder reads line by line
        hf_dataset = load_dataset("text", data_files={"train": file_location})

        # Split
        hf_dataset = hf_dataset["train"].train_test_split(test_size=0.1)
        train_ds = hf_dataset["train"]
        test_ds = hf_dataset["test"]

        # 3. Model & Tokenizer
        # Using state-spaces/mamba-130m or similar via Hugging Face
        model_name = "state-spaces/mamba-130m-hf"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token # Mamba doesn't use padding usually but for batched training we need it

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        tokenized_train = train_ds.map(tokenize_function, batched=True)
        tokenized_test = test_ds.map(tokenize_function, batched=True)

        # Check for GPU
        use_cuda = torch.cuda.is_available()
        # If no GPU, Mamba via HF uses a slow naive implementation.

        model = MambaForCausalLM.from_pretrained(model_name)

        # Training Args
        hyperparams = model_info.get('hyperparameters', {})
        epochs = int(hyperparams.get('epochs', 3))
        batch_size = int(hyperparams.get('batch_size', 4))
        lr = float(hyperparams.get('learning_rate', 2e-4))

        training_args = TrainingArguments(
            output_dir=f"ml_models/mamba_{model_id}",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            use_cpu=not use_cuda
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
        )

        # Train
        train_results = trainer.train()

        # Evaluate
        metrics = trainer.evaluate()

        # Save
        save_path = f"ml_models/model_{model_id}"
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)

        final_metrics = {
            "loss": metrics.get("eval_loss"),
            "train_runtime": train_results.metrics.get("train_runtime")
        }

        crud.update_model_status(db, model_id, "completed", metrics=final_metrics, model_path=save_path)

        asyncio.run(manager.broadcast(json.dumps({
            "model_id": model_id,
            "status": "completed",
            "metrics": final_metrics
        })))

    except Exception as e:
        import traceback
        traceback.print_exc()
        crud.update_model_status(db, model_id, "failed")
        asyncio.run(manager.broadcast(json.dumps({
            "model_id": model_id,
            "status": "failed",
            "error": str(e)
        })))

    finally:
        db.close()
