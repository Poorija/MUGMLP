import json
import torch
import asyncio
import os
from sqlalchemy.orm import Session
from database import SessionLocal
import crud
from websocket_manager import manager
from hardware_scanner import scanner
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset as HFDataset

# Use a default small model for generation/judging if none specified
DEFAULT_MODEL = "state-spaces/mamba-130m-hf"

def generate_synthetic_data_task(model_id: int, dataset_id: int, model_info: dict):
    """
    Generates synthetic data (Instruction/Response) using a local LLM.
    Note: dataset_id here is the TARGET dataset to create/update, or we create a new one.
    Actually, usually we create a new dataset. But the API structure expects a model_id.
    We'll treat this as a "Data Generation Job" tracked as a Model for simplicity in this prototype.
    """
    db: Session = SessionLocal()
    try:
        # Hardware Check
        hw_info = scanner.get_hardware_info()
        if not scanner.check_feasibility({"requires_gpu": False})["feasible"]:
             raise RuntimeError("Insufficient hardware for generation.")

        crud.update_model_status(db, model_id, "running")
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "started"})))

        hyperparams = model_info.get('hyperparameters', {})
        topic = hyperparams.get("topic", "General Assistant")
        count = int(hyperparams.get("count", 10))

        # Load Generator Model
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
        if torch.cuda.is_available():
            model = model.cuda()

        generated_data = []

        for i in range(count):
            # Simple few-shot prompt for synthesis
            prompt = f"""Generate a training example for a {topic} chatbot.
Example:
User: Hello
Assistant: Hi there! How can I help?

User: Define AI.
Assistant: AI stands for Artificial Intelligence.

User: """
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Very naive parsing
            generated_data.append({"id": i, "raw_text": decoded})

            # Progress update
            if i % 5 == 0:
                asyncio.run(manager.broadcast(json.dumps({
                    "model_id": model_id,
                    "status": "running",
                    "progress": f"{i}/{count}"
                })))

        # Save to file
        filename = f"synthetic_data_{model_id}.jsonl"
        filepath = f"uploads/{filename}"
        with open(filepath, "w") as f:
            for entry in generated_data:
                f.write(json.dumps(entry) + "\n")

        # Create a new dataset record for this generated data
        # (Or update the dummy one if passed)
        # For this flow, we'll just save the path in the 'model' record
        # as if the 'model' is the generator artifact.

        metrics = {"count": count, "output_file": filename}
        crud.update_model_status(db, model_id, "completed", metrics=metrics, model_path=filepath)
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "completed", "metrics": metrics})))

    except Exception as e:
        crud.update_model_status(db, model_id, "failed")
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "failed", "error": str(e)})))
    finally:
        db.close()

def generate_constitutional_data_task(model_id: int, dataset_id: int, model_info: dict):
    """
    Generates data using Constitutional AI principles (Critique -> Revise).
    """
    db: Session = SessionLocal()
    try:
        crud.update_model_status(db, model_id, "running")

        hyperparams = model_info.get('hyperparameters', {})
        constitution = hyperparams.get("constitution", "The response must be helpful, harmless, and honest.")
        count = int(hyperparams.get("count", 5))
        topic = hyperparams.get("topic", "AI Ethics")

        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
        if torch.cuda.is_available():
            model = model.cuda()

        generated_data = []

        for i in range(count):
            # 1. Generate Initial
            prompt = f"User: Write a response about {topic}. Assistant:"
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available(): inputs = inputs.to("cuda")
            out = model.generate(**inputs, max_new_tokens=50)
            initial_response = tokenizer.decode(out[0], skip_special_tokens=True).split("Assistant:")[-1].strip()

            # 2. Critique
            critique_prompt = f"Constitution: {constitution}\nResponse: {initial_response}\nCritique the response based on the constitution:"
            inputs = tokenizer(critique_prompt, return_tensors="pt")
            if torch.cuda.is_available(): inputs = inputs.to("cuda")
            out = model.generate(**inputs, max_new_tokens=50)
            critique = tokenizer.decode(out[0], skip_special_tokens=True).split("Critique:")[-1].strip()

            # 3. Revise
            revise_prompt = f"Original Response: {initial_response}\nCritique: {critique}\nRevised Response:"
            inputs = tokenizer(revise_prompt, return_tensors="pt")
            if torch.cuda.is_available(): inputs = inputs.to("cuda")
            out = model.generate(**inputs, max_new_tokens=50)
            revised_response = tokenizer.decode(out[0], skip_special_tokens=True).split("Revised Response:")[-1].strip()

            generated_data.append({
                "prompt": prompt,
                "initial": initial_response,
                "critique": critique,
                "revised": revised_response
            })

            if i % 2 == 0:
                asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "running", "progress": f"{i}/{count}"})))

        # Save
        filename = f"constitutional_data_{model_id}.jsonl"
        filepath = f"uploads/{filename}"
        with open(filepath, "w") as f:
            for entry in generated_data:
                f.write(json.dumps(entry) + "\n")

        metrics = {"count": count, "output_file": filename}
        crud.update_model_status(db, model_id, "completed", metrics=metrics, model_path=filepath)
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "completed", "metrics": metrics})))

    except Exception as e:
        crud.update_model_status(db, model_id, "failed")
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "failed", "error": str(e)})))
    finally:
        db.close()

def llm_judge_task(model_id: int, dataset_id: int, model_info: dict):
    """
    Evaluates a dataset using an LLM as a Judge.
    Expects dataset to have 'question' and 'answer' columns.
    """
    db: Session = SessionLocal()
    try:
        crud.update_model_status(db, model_id, "running")

        dataset_record = crud.get_dataset(db, dataset_id)
        file_location = f"uploads/{dataset_record.filename}"

        # Load dataset
        # Check type
        if dataset_record.filename.endswith('.jsonl'):
             with open(file_location) as f:
                 data = [json.loads(line) for line in f]
        else:
             # Fallback/Error
             raise ValueError("Judge requires JSONL dataset")

        # Load Judge Model
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
        if torch.cuda.is_available():
            model = model.cuda()

        scores = []
        for i, row in enumerate(data[:10]): # Limit to 10 for demo speed
            q = row.get("question", "") or row.get("raw_text", "")
            a = row.get("answer", "")

            rubric = f"""Act as a judge. Rate the following response 1 to 5.
Question: {q}
Response: {a}
Score (1-5):"""

            inputs = tokenizer(rubric, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            outputs = model.generate(**inputs, max_new_tokens=5)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract score (naive)
            try:
                score = int(output_text.split("Score (1-5):")[-1].strip()[0])
            except:
                score = 3 # Default fallback

            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0
        metrics = {"average_score": avg_score, "judged_count": len(scores)}

        crud.update_model_status(db, model_id, "completed", metrics=metrics)
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "completed", "metrics": metrics})))

    except Exception as e:
        crud.update_model_status(db, model_id, "failed")
        asyncio.run(manager.broadcast(json.dumps({"model_id": model_id, "status": "failed", "error": str(e)})))
    finally:
        db.close()
