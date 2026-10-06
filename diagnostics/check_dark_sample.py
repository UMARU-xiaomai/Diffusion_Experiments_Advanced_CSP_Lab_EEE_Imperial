"""Bounded seed search to reproduce a dark sample using the saved weights."""
import contextlib
import io
import json
import matplotlib.pyplot as plt
import torch
import check_denoising as d

d.load_definitions()
checkpoint = torch.load(d.ROOT / "pokemon_ddpm_student.pt", map_location="cpu", weights_only=True)
model = d.TinyDenoiser().eval().cuda()
model.load_state_dict(checkpoint["model_state_dict"])
schedule = d.make_linear_schedule(checkpoint["num_diffusion_steps"], device="cuda")
tested = []
best = None
for seed in range(12):
    with contextlib.redirect_stdout(io.StringIO()):
        result, frames, labels, final = d.run_chain(d.p_sample, model, schedule, seed=seed, batch_size=1)
    row = {"seed": seed, **result["trajectory"][-1]}
    tested.append(row)
    print(json.dumps(row), flush=True)
    if best is None or row["black_pixel_fraction"] > best[0]:
        best = (row["black_pixel_fraction"], seed, result, frames, labels)
    if row["black_pixel_fraction"] >= 0.5:
        break
_, seed, original, original_frames, labels = best
with contextlib.redirect_stdout(io.StringIO()):
    clipped, clipped_frames, _, _ = d.run_chain(d.p_sample_clipped, model, schedule, seed=seed, batch_size=1)
report = {"device": "cuda", "batch_size": 1, "selection": "darkest of up to 12 seeds, stop at black_fraction >= 0.5",
          "seeds_tested": tested, "selected_seed": seed, "original": original, "bounded_x0": clipped}
(d.OUT / "dark_sample_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
fig, axes = plt.subplots(2, len(labels), figsize=(22, 4.5))
for row, (name, frames) in enumerate((("original", original_frames), ("bounded_x0", clipped_frames))):
    for col, (frame, label) in enumerate(zip(frames, labels)):
        axes[row, col].imshow(d.tensor_to_img(frame))
        axes[row, col].set_title(label, fontsize=9)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        if col == 0:
            axes[row, col].set_ylabel(name)
fig.suptitle(f"Dark-sample diagnostic (selected seed {seed}): same weights and noise; only sampler changes")
fig.tight_layout()
fig.savefig(d.OUT / "dark_sample_comparison.png", dpi=150, bbox_inches="tight")
print(json.dumps({"selected_seed": seed, "original_final": original["trajectory"][-1],
                  "bounded_x0_final": clipped["trajectory"][-1]}), flush=True)
