import os
import json
import torch
import time
import zipfile
import asyncio
from sqlalchemy.orm import Session
from .database import SessionLocal
from . import crud
from .websocket_manager import manager
from .hardware_scanner import scanner

# Transformers / Torch
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from datasets import Dataset as HFDataset, Image
from torchvision.transforms import (
    Compose, Normalize, RandomResizedCrop, ColorJitter, ToTensor, Resize, CenterCrop
)
from PIL import Image as PILImage
from peft import get_peft_model, LoraConfig, TaskType

# --- Vision Training Task ---

def load_image_dataset(zip_path: str, extract_path: str):
    """
    Extracts a zip file of images and prepares a Hugging Face Dataset.
    Expected structure:
    root/
      class_a/
        img1.jpg
      class_b/
        img2.jpg
    """
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # Block Zip Slip
                member_path = os.path.abspath(os.path.join(extract_path, member))
                if not member_path.startswith(os.path.abspath(extract_path)):
                    continue # Skip malicious paths

                # Extract
                zip_ref.extract(member, extract_path)

    # Traverse directory to build dataset list
    image_paths = []
    labels = []
    label_names = sorted([d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))])
    label_map = {name: i for i, name in enumerate(label_names)}

    for label_name in label_names:
        class_dir = os.path.join(extract_path, label_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                full_path = os.path.join(class_dir, img_name)
                image_paths.append(full_path)
                labels.append(label_map[label_name])

    if not image_paths:
        raise ValueError("No images found in the zip file structure.")

    dataset = HFDataset.from_dict({"image": image_paths, "label": labels})
    dataset = dataset.cast_column("image", Image())

    return dataset, label_names, label_map

def train_vision_model_task(model_id: int, dataset_id: int, model_info: dict):
    db: Session = SessionLocal()
    try:
        # 1. Hardware Check
        hw_info = scanner.get_hardware_info()
        feasibility = scanner.check_feasibility({
            "min_vram_gb": 2.0, # ViT Base needs ~2GB for batch size 8
            "requires_gpu": False # We support CPU fallback (slow)
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

        # 2. Load Data
        dataset_record = crud.get_dataset(db, dataset_id)
        zip_path = f"uploads/{dataset_record.filename}"
        extract_path = f"uploads/extracted_{dataset_id}"

        hf_dataset, label_names, label_map = load_image_dataset(zip_path, extract_path)

        # Split
        hf_dataset = hf_dataset.train_test_split(test_size=0.2)
        train_ds = hf_dataset["train"]
        test_ds = hf_dataset["test"]

        # 3. Model & Processor
        model_name = "google/vit-base-patch16-224-in21k" # Good starting point for finetuning
        processor = ViTImageProcessor.from_pretrained(model_name)

        # Transforms
        normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
        size = (
            processor.size["shortest_edge"]
            if "shortest_edge" in processor.size
            else (processor.size["height"], processor.size["width"])
        )

        train_transforms = Compose([
            RandomResizedCrop(size),
            ColorJitter(brightness=0.1, hue=0.1),
            ToTensor(),
            normalize,
        ])

        val_transforms = Compose([
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ])

        def transform_train(examples):
            examples["pixel_values"] = [train_transforms(img.convert("RGB")) for img in examples["image"]]
            return examples

        def transform_val(examples):
            examples["pixel_values"] = [val_transforms(img.convert("RGB")) for img in examples["image"]]
            return examples

        train_ds.set_transform(transform_train)
        test_ds.set_transform(transform_val)

        # Load Model
        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {label: i for i, label in enumerate(label_names)}

        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id
        )

        # Apply LoRA / DoRA if requested
        hyperparams = model_info.get('hyperparameters', {})
        if hyperparams.get("use_lora"):
            peft_config = LoraConfig(
                task_type=TaskType.IMAGE_CLASSIF,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query", "value"], # Standard for ViT
                use_dora=hyperparams.get("use_dora", False) # Support DoRA
            )
            model = get_peft_model(model, peft_config)
            # Log trainable params
            model.print_trainable_parameters()
        epochs = int(hyperparams.get('epochs', 3))
        batch_size = int(hyperparams.get('batch_size', 8))
        lr = float(hyperparams.get('learning_rate', 2e-5))

        use_cuda = torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=f"ml_models/vit_{model_id}",
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir=f"ml_models/logs_{model_id}",
            use_cpu=not use_cuda
        )

        # Metric
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=1)
            return {"accuracy": (predictions == labels).mean()}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=processor,
            compute_metrics=compute_metrics,
        )

        # Train
        train_results = trainer.train()

        # Evaluate
        metrics = trainer.evaluate()

        # Save
        save_path = f"ml_models/model_{model_id}"
        trainer.save_model(save_path)
        processor.save_pretrained(save_path)

        # Final Metrics
        final_metrics = {
            "accuracy": metrics.get("eval_accuracy"),
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
