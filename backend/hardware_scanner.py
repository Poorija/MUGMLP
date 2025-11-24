import torch
import psutil
import json
import logging
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardwareScanner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.system_info = self._scan_system()

    def _scan_system(self):
        """Scans the system for CPU and GPU resources."""
        info = {
            "system": platform.system(),
            "processor": platform.processor(),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "gpu_available": torch.cuda.is_available(),
            "gpus": []
        }

        if info["gpu_available"]:
            try:
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info = {
                        "index": i,
                        "name": props.name,
                        "vram_total_gb": round(props.total_memory / (1024**3), 2),
                        "compute_capability": f"{props.major}.{props.minor}",
                        # Basic heuristics for features
                        "supports_flash_attn": props.major >= 8, # Ampere or newer
                        "supports_bf16": props.major >= 8
                    }
                    info["gpus"].append(gpu_info)
            except Exception as e:
                logger.error(f"Error scanning GPU: {e}")
                info["gpu_scan_error"] = str(e)

        return info

    def get_hardware_info(self):
        return self.system_info

    def check_feasibility(self, model_requirements: dict):
        """
        Checks if the current hardware meets the model requirements.

        Args:
            model_requirements (dict): {
                "min_vram_gb": float,
                "min_ram_gb": float,
                "requires_gpu": bool,
                "requires_flash_attn": bool
            }

        Returns:
            dict: {"feasible": bool, "warnings": list, "errors": list}
        """
        requirements = {
            "min_vram_gb": 0,
            "min_ram_gb": 4, # Baseline for any OS
            "requires_gpu": False,
            "requires_flash_attn": False,
        }
        requirements.update(model_requirements)

        result = {
            "feasible": True,
            "warnings": [],
            "errors": []
        }

        # Check System RAM
        if self.system_info["ram_available_gb"] < requirements["min_ram_gb"]:
            msg = f"Low System RAM: {self.system_info['ram_available_gb']}GB available, {requirements['min_ram_gb']}GB recommended."
            result["warnings"].append(msg)
            # We don't block strictly on RAM as swap exists, but it's a strong warning

        # Check GPU
        if requirements["requires_gpu"] and not self.system_info["gpu_available"]:
            result["feasible"] = False
            result["errors"].append("This model requires a GPU, but none was detected.")
            return result

        if self.system_info["gpu_available"]:
            # Check VRAM (assume single GPU for now)
            max_vram = max([g["vram_total_gb"] for g in self.system_info["gpus"]]) if self.system_info["gpus"] else 0

            if max_vram < requirements["min_vram_gb"]:
                # If required VRAM is more than available, it might OOM
                msg = f"Insufficient VRAM: {max_vram}GB detected, {requirements['min_vram_gb']}GB required."
                if requirements.get("strict_vram"):
                    result["feasible"] = False
                    result["errors"].append(msg)
                else:
                    result["warnings"].append(msg + " Training might fall back to CPU offloading (slow).")

            # Check Flash Attention
            if requirements["requires_flash_attn"]:
                 supported = any([g["supports_flash_attn"] for g in self.system_info["gpus"]])
                 if not supported:
                     result["feasible"] = False
                     result["errors"].append("Flash Attention requires NVIDIA Ampere (RTX 30xx/A100) or newer.")

        elif requirements["min_vram_gb"] > 0:
             # GPU not available but VRAM requested
             result["warnings"].append("Running GPU-optimized model on CPU. This will be extremely slow.")

        return result

# Singleton instance
scanner = HardwareScanner()

if __name__ == "__main__":
    print(json.dumps(scanner.get_hardware_info(), indent=2))
