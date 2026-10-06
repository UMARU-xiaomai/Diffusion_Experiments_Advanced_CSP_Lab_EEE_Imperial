"""Check the saved DDPM without running notebook setup, training, or exports.

Loads only function/class definitions from selected notebook cells. Results go
to diagnostics/; the notebook, checkpoint, and original figures are untouched.
"""
import ast
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics"
DEVICE = torch.device("cpu")
CHANNELS = 3
IMG_SIZE = 32
BETA_START = 1e-4
BETA_END = 2e-2
torch.set_num_threads(4)


def load_definitions():
    notebook = json.loads((ROOT / "Practical_Project.ipynb").read_text(encoding="utf-8"))
    for index in (15, 29, 31, 35, 37, 40):
        tree = ast.parse("".join(notebook["cells"][index]["source"]))
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, f"Practical_Project.ipynb:cell_{index}", "exec"), globals())


@torch.no_grad()
def p_sample_clipped(model, x_t, step, schedule):
    """Experimental bounded-x0 sampler; do not clamp the noisy state x_t.

    Reference: OpenAI improved-diffusion p_mean_variance / process_xstart:
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
    """
    t = torch.full((x_t.shape[0],), step, dtype=torch.long, device=x_t.device)
    beta = extract(schedule["betas"], t, x_t.shape)
    alpha = extract(schedule["alphas"], t, x_t.shape)
    abar = extract(schedule["alpha_bars"], t, x_t.shape)
    abar_prev = extract(schedule["alpha_bars_prev"], t, x_t.shape)
    eps = model(x_t, t)
    x0_hat = ((x_t - (1 - abar).sqrt() * eps) / abar.sqrt()).clamp(-1, 1)
    if step == 0:
        return x0_hat
    mean = (beta * abar_prev.sqrt() / (1 - abar) * x0_hat
            + alpha.sqrt() * (1 - abar_prev) / (1 - abar) * x_t)
    variance = extract(schedule["posterior_variance"], t, x_t.shape)
    return mean + variance.sqrt() * torch.randn_like(x_t)


def stats(x):
    return {
        "mean": x.mean().item(), "std": x.std().item(),
        "min": x.min().item(), "max": x.max().item(),
        "below_minus_one": (x < -1).float().mean().item(),
        "above_one": (x > 1).float().mean().item(),
        "black_pixel_fraction": (x <= -1).all(dim=1).float().mean().item(),
        "rgb_mean": x.mean(dim=(0, 2, 3)).tolist(),
        "finite": bool(torch.isfinite(x).all()),
    }


def load_clean_batch(count=16):
    batch = []
    for path in sorted((ROOT / "pokemon").glob("*.png"))[:count]:
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", image.size, "WHITE")
            white.paste(rgba, (0, 0), rgba)
            batch.append(image_to_tensor(white.convert("RGB")))
    return torch.stack(batch)


@torch.no_grad()
def validate_math(schedule, model, clean):
    evaluation_schedule = schedule
    # Compare algebra in float64: float32 cancellation in 1-alpha_bar near
    # t=0 otherwise produces ~8e-5 differences between equivalent formulas.
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        schedule = make_linear_schedule(len(schedule["betas"]), device=DEVICE)
    finally:
        torch.set_default_dtype(original_dtype)
    torch.manual_seed(123)
    x0 = clean[:4].double()
    t = torch.tensor([0, 1, 500, 999])
    noise = torch.randn_like(x0)
    xt, returned_noise = q_sample(x0, t, schedule, noise=noise)
    abar = extract(schedule["alpha_bars"], t, x0.shape)
    recovered = (xt - (1 - abar).sqrt() * noise) / abar.sqrt()
    torch.testing.assert_close(returned_noise, noise)
    torch.testing.assert_close(recovered, x0, atol=1e-9, rtol=1e-9)

    class Oracle(nn.Module):
        def forward(self, x, times):
            a = extract(schedule["alpha_bars"], times, x.shape)
            return (x - a.sqrt() * x0) / (1 - a).sqrt()

    for step in (0, 1, 500, 999):
        times = torch.full((len(x0),), step, dtype=torch.long)
        noisy, _ = q_sample(x0, times, schedule, noise=noise)
        b = extract(schedule["betas"], times, x0.shape)
        a = extract(schedule["alphas"], times, x0.shape)
        ab = extract(schedule["alpha_bars"], times, x0.shape)
        prev = extract(schedule["alpha_bars_prev"], times, x0.shape)
        if step == 0:
            expected = x0
        else:
            expected = b * prev.sqrt() / (1 - ab) * x0 + a.sqrt() * (1 - prev) / (1 - ab) * noisy
            torch.manual_seed(900 + step)
            expected += extract(schedule["posterior_variance"], times, x0.shape).sqrt() * torch.randn_like(noisy)
        torch.manual_seed(900 + step)
        original = p_sample(Oracle(), noisy, step, schedule)
        torch.testing.assert_close(original, expected, atol=1e-9, rtol=1e-9)
        torch.manual_seed(900 + step)
        bounded = p_sample_clipped(Oracle(), noisy, step, schedule)
        torch.testing.assert_close(bounded, expected, atol=1e-9, rtol=1e-9)

    schedule = evaluation_schedule
    evaluation = []
    torch.manual_seed(123)
    for step in (0, 10, 100, 300, 500, 700, 999):
        t = torch.full((len(clean),), step, dtype=torch.long)
        xt, eps = q_sample(clean, t, schedule)
        prediction = model(xt, t)
        ab = schedule["alpha_bars"][step]
        x0_hat = (xt - (1 - ab).sqrt() * prediction) / ab.sqrt()
        evaluation.append({"step": step, "epsilon_mse": F.mse_loss(prediction, eps).item(),
                           "epsilon_bias": (prediction - eps).mean(dim=(0, 2, 3)).tolist(),
                           "x0_reconstruction_mse": F.mse_loss(x0_hat, clean).item()})
    return evaluation


@torch.no_grad()
def run_chain(sampler, model, schedule, seed=42, batch_size=4):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    x = torch.randn(batch_size, CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    rows = [{"step": "initial", **stats(x)}]
    frames = [x[0].detach().cpu().clone()]
    labels = ["initial"]
    started = time.perf_counter()
    for step in reversed(range(len(schedule["betas"]))):
        x = sampler(model, x, step, schedule)
        if step % 100 == 0 or step == 999:
            row = {"step": step, **stats(x)}
            rows.append(row)
            frames.append(x[0].detach().cpu().clone())
            labels.append(str(step))
            print(json.dumps({"sampler": sampler.__name__, "seed": seed, "elapsed_s": round(time.perf_counter()-started, 1), **row}), flush=True)
    return {"seed": seed, "batch_size": batch_size, "trajectory": rows,
            "per_sample_final": [stats(sample.unsqueeze(0)) for sample in x]}, frames, labels, x.detach().cpu()


def main():
    load_definitions()
    checkpoint_path = ROOT / "pokemon_ddpm_student.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    steps = checkpoint["num_diffusion_steps"]
    schedule = make_linear_schedule(steps, device=DEVICE)
    model = TinyDenoiser().eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    clean = load_clean_batch()
    evaluation = validate_math(schedule, model, clean)
    print("Forward/oracle reverse checks passed; checkpoint loaded strictly.", flush=True)
    print(json.dumps({"clean_data": stats(clean), "known_forward_noise_evaluation": evaluation}), flush=True)
    report = {"torch": torch.__version__, "device": str(DEVICE),
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "epochs": len(checkpoint["loss_history"]), "loss_first": checkpoint["loss_history"][0],
              "loss_last": checkpoint["loss_history"][-1], "diffusion_steps": steps,
              "terminal_alpha_bar": schedule["alpha_bars"][-1].item(),
              "math_checks": "passed: forward reconstruction and oracle reverse t=0,1,500,999",
              "known_forward_noise_evaluation": evaluation, "runs": {}}
    results = []
    for name, sampler in (("original", p_sample), ("bounded_x0", p_sample_clipped)):
        result, frames, labels, final = run_chain(sampler, model, schedule)
        report["runs"][name] = result
        results.append((name, frames, labels, final))
        (OUT / "denoising_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, len(results[0][1]), figsize=(22, 4.5))
    for row, (name, frames, labels, _) in enumerate(results):
        for col, (frame, label) in enumerate(zip(frames, labels)):
            axes[row, col].imshow(tensor_to_img(frame))
            axes[row, col].set_title(label, fontsize=9)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(name)
    fig.suptitle("Same checkpoint and random seed; only the reverse sampler changes")
    fig.tight_layout()
    fig.savefig(OUT / "denoising_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(9, 5))
    for row, (name, _, _, final) in enumerate(results):
        for col, sample in enumerate(final):
            axes[row, col].imshow(tensor_to_img(sample))
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(name)
    fig.tight_layout()
    fig.savefig(OUT / "denoising_final_samples.png", dpi=150, bbox_inches="tight")
    print("Saved diagnostics/denoising_metrics.json and comparison PNGs", flush=True)


if __name__ == "__main__":
    main()
