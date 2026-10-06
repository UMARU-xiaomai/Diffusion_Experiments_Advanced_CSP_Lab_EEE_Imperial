"""Compare the same checkpoint and inputs on CPU and the notebook's GPU."""
import json
import time
import torch
import check_denoising as diagnostic


def main():
    diagnostic.load_definitions()
    checkpoint = torch.load(diagnostic.ROOT / "pokemon_ddpm_student.pt", map_location="cpu", weights_only=True)
    cpu_model = diagnostic.TinyDenoiser().eval()
    cpu_model.load_state_dict(checkpoint["model_state_dict"])
    report = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
              "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        report["device"] = torch.cuda.get_device_name(0)
        report["capability"] = torch.cuda.get_device_capability(0)
        report["compiled_architectures"] = torch.cuda.get_arch_list()
        print(json.dumps(report), flush=True)
        try:
            gpu_model = diagnostic.TinyDenoiser().eval().cuda()
            gpu_model.load_state_dict(checkpoint["model_state_dict"])
            clean = diagnostic.load_clean_batch(4)
            schedule = diagnostic.make_linear_schedule(checkpoint["num_diffusion_steps"], device="cpu")
            torch.manual_seed(42)
            evaluation = []
            with torch.no_grad():
                for step in (999, 500, 100, 0):
                    t = torch.full((4,), step, dtype=torch.long)
                    x, noise = diagnostic.q_sample(clean, t, schedule)
                    cpu = cpu_model(x, t)
                    start = time.perf_counter()
                    gpu = gpu_model(x.cuda(), t.cuda()).cpu()
                    row = {"step": step, "max_abs_difference": (cpu-gpu).abs().max().item(),
                           "rms_difference": (cpu-gpu).square().mean().sqrt().item(),
                           "cpu_epsilon_mse": (cpu-noise).square().mean().item(),
                           "gpu_epsilon_mse": (gpu-noise).square().mean().item(),
                           "cpu_rgb_mean": cpu.mean(dim=(0, 2, 3)).tolist(),
                           "gpu_rgb_mean": gpu.mean(dim=(0, 2, 3)).tolist(),
                           "gpu_elapsed_s": time.perf_counter() - start}
                    print(json.dumps(row), flush=True)
                    evaluation.append(row)
            report["predictions"] = evaluation
        except RuntimeError as exc:
            report["runtime_error"] = str(exc)
            print("GPU runtime error:", str(exc), flush=True)
    (diagnostic.OUT / "device_consistency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
