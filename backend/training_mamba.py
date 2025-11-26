import os
import json
import torch
import asyncio
from sqlalchemy.orm import Session
from database import SessionLocal
import crud
from websocket_manager import manager
from hardware_scanner import scanner

# Transformers
from transformers import AutoTokenizer, MambaForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Check for bitsandbytes
try:
    import bitsandbytes
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

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

        # Load args
        hyperparams = model_info.get('hyperparameters', {})
        use_qlora = hyperparams.get("use_qlora", False) and HAS_BNB

        # Load Model
        model_kwargs = {}
        if use_qlora:
            model_kwargs["load_in_4bit"] = True

        # Check Flash Attention
        if hw_info.get("supports_flash_attn"):
             # Mamba has its own efficient kernels, but for general transformers models we'd set:
             # model_kwargs["attn_implementation"] = "flash_attention_2"
             pass

        model = MambaForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Apply LoRA / DoRA if requested
        if hyperparams.get("use_lora"):

             # Mamba might not be natively supported by PEFT auto mapping yet in older versions,
             # but we can target modules manually.
             # Typical Mamba linear layers: "in_proj", "out_proj", "x_proj", "dt_proj"
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["in_proj", "out_proj"], # Common targets for SSM
                use_dora=hyperparams.get("use_dora", False) # Support DoRA
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
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
