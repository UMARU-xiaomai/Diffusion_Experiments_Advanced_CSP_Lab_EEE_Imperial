"""Use the notebook's GPU and batch_size=1 with an explicit fresh seed."""
import json
import matplotlib.pyplot as plt
import torch
import check_denoising as d

d.load_definitions()
checkpoint = torch.load(d.ROOT / "pokemon_ddpm_student.pt", map_location="cpu", weights_only=True)
model = d.TinyDenoiser().eval().cuda()
model.load_state_dict(checkpoint["model_state_dict"])
schedule = d.make_linear_schedule(checkpoint["num_diffusion_steps"], device="cuda")
report = {"device": "cuda", "batch_size": 1, "seed": 42, "runs": {}}
fig, axes = plt.subplots(2, 12, figsize=(22, 4.5))
for row, (name, sampler) in enumerate((("original", d.p_sample), ("bounded_x0", d.p_sample_clipped))):
    result, frames, labels, final = d.run_chain(sampler, model, schedule, batch_size=1)
    report["runs"][name] = result
    for col, (frame, label) in enumerate(zip(frames, labels)):
        axes[row, col].imshow(d.tensor_to_img(frame))
        axes[row, col].set_title(label, fontsize=9)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        if col == 0:
            axes[row, col].set_ylabel(name)
fig.suptitle("GPU, batch size 1, seed 42: same checkpoint; only the reverse sampler changes")
fig.tight_layout()
fig.savefig(d.OUT / "gpu_single_comparison.png", dpi=150, bbox_inches="tight")
(d.OUT / "gpu_single_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
