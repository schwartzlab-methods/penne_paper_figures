import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from dataset import ShaneSeqCellTypeDataset
from model import GeneExpPredVisiumHD
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel
from _feature_extractors import init_spaghetti, pre_processing_phikon
from torchvision import transforms, utils
from PIL import Image
import pandas as pd

# ---------------------------
# Utilities
# ---------------------------
def normalize_arr(x):
    x = x - x.min()
    if x.max() != 0:
        x = x / x.max()
    return x

def upsample_patch_map(patch_map, target_size, interpolation='bilinear'):
    """
    patch_map: np array (ph, pw)
    target_size: (H, W)
    returns numpy HxW
    """
    t = torch.from_numpy(patch_map).unsqueeze(0).unsqueeze(0).float()  # 1,1,ph,pw
    up = F.interpolate(t, size=target_size, mode=interpolation, align_corners=False)
    return up.squeeze().cpu().numpy()

def apply_colormap_on_image(org_img, heatmap, colormap=cm.jet, alpha=0.5):
    """
    org_img: numpy HxW or HxWx3 in [0,1]
    heatmap: HxW in [0,1]
    returns overlay HxWx3 in [0,1]
    """
    if org_img.ndim == 2:
        org_img_rgb = np.stack([org_img]*3, axis=-1)
    else:
        org_img_rgb = org_img
    colored = colormap(heatmap)[:, :, :3]  # drop alpha from cmap
    overlay = alpha * colored + (1 - alpha) * org_img_rgb
    overlay = np.clip(overlay, 0, 1)
    return overlay

def show_heatmap_overlay(heatmap, overlay, orig, figsize=(10,4), name=""):
    plt.figure(figsize=figsize)
    plt.subplot(1,3,1)
    plt.imshow(orig)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(heatmap, cmap='hot')
    plt.title("Patch heatmap")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"/home/zf2dong/scratch/temp/gating/grad_cam_vit_gate_result_{name}.png")
    plt.close()

# ---------------------------
# Grad-CAM-like function for CLS-only downstream
# ---------------------------
def grad_cam_vit_gate_cls(
    vit_model,
    downstream_model,
    img_tensor,
    gate_selection_fn,
    patch_module,
    target_size=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    vit_model: HuggingFace ViTModel (or similar). Should accept img_tensor and produce outputs used by downstream_model.
    downstream_model: callable that accepts a CLS/features and returns predictions OR internally uses vit outputs; 
                      must connect computational graph so backward() on gate flows back to vit.
    img_tensor: torch tensor (1, C, H, W) preprocessed exactly as used in inference
    gate_selection_fn: function(model_out) -> scalar tensor (the gate scalar to backprop from).
                       Alternatively, it can call vit_model+downstream_model internally and return a scalar.
    patch_module: module to hook that produces patch tokens or last token representations. For HuggingFace ViT try:
                  model.vit.embeddings.patch_embeddings or model.embeddings.patch_embeddings (experiment).
                  We will try to accept outputs shaped (B, N, D) or (B, D, H, W).
    Returns: heatmap (H,W numpy 0..1), overlay (H,W,3 0..1), orig_image (H,W,3)
    """

    saved = {}
    def forward_hook(module, input, output):
        out = output
        # normalize shapes to (B, N, D)
        if out is None:
            return
        if isinstance(out, tuple):
            out = out[0]
        if out.ndim == 4:
            # B, D, Hp, Wp -> B, Hp*Wp, D
            B, D, Hp, Wp = out.shape
            out = out.view(B, D, Hp*Wp).permute(0,2,1).contiguous()
        elif out.ndim == 3:
            # assume B, N, D
            pass
        else:
            raise RuntimeError(f"patch_module output has unexpected ndim {out.ndim}")
        out.retain_grad()
        saved['patch_embeddings'] = out

    hook = patch_module.register_forward_hook(forward_hook)

    # zero grads
    vit_model.zero_grad()
    downstream_model.zero_grad()

    # Forward: we let user's gate_selection_fn perform the forward pass that triggers hooks
    gate_scalar = gate_selection_fn(vit_model, downstream_model, img_tensor)  # must return scalar tensor with grad
    if gate_scalar is None:
        hook.remove()
        raise RuntimeError("gate_selection_fn returned None. It must return a scalar torch tensor from the model.")
    if gate_scalar.dim() != 0 and not (gate_scalar.dim()==1 and gate_scalar.shape[0]==1):
        gate_scalar = gate_scalar.reshape(-1).mean()

    # Backward
    gate_scalar.backward()

    if 'patch_embeddings' not in saved:
        hook.remove()
        raise RuntimeError("Patch embeddings not captured. Make sure patch_module points to correct ViT submodule.")

    patch_emb = saved['patch_embeddings']   # (B, N, D)
    patch_grad = patch_emb.grad             # (B, N, D)
    if patch_grad is None:
        hook.remove()
        raise RuntimeError("patch_embeddings.grad is None. The gate scalar didn't backprop to patch embeddings.")

    # Importance per patch: use L2 norm over embedding dim (alternatives: mean of grad, grad*act, etc.)
    importance = torch.norm(patch_grad[0], dim=-1).detach().cpu().numpy()  # (N,)

    # Infer patch grid
    N = importance.shape[0]
    s = int(np.sqrt(N))
    if s*s == N:
        ph, pw = s, s
    else:
        # fallback: try to use token layout in ViT (many ViT use square grid); else treat as 1xN
        ph, pw = 1, N

    patch_map = importance.reshape(ph, pw)
    patch_map = normalize_arr(patch_map)

    # Upsample to image size
    if target_size is None:
        _, _, H, W = img_tensor.shape
        target_size = (H, W)
    heatmap = upsample_patch_map(patch_map, target_size)

    # Prepare original image for overlay (unnormalize if needed)
    img_np = img_tensor.detach().cpu().squeeze().numpy()
    if img_np.shape[0] == 1:
        orig = img_np[0]
        orig_viz = normalize_arr(orig)
    elif img_np.shape[0] == 3:
        orig_viz = np.transpose(img_np, (1,2,0))
        orig_viz = normalize_arr(orig_viz)
    else:
        orig_viz = normalize_arr(np.mean(img_np, axis=0))

    overlay = apply_colormap_on_image(orig_viz, heatmap, alpha=0.5)

    hook.remove()
    vit_model.zero_grad()
    downstream_model.zero_grad()
    return heatmap, overlay, orig_viz

# ---------- Example usage ----------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "/home/zf2dong/ric/results/aim_2_results/aim2/big_bio_1translator_10.0_marker_domain_three_stage_10xvisium_filtered_gated_acrosscell_pcm_celltype_hubert_5cosine/model/stage_3/version_0/checkpoints/epoch=29-step=4350.ckpt"
    gate_correlation_csv_path = "/home/zf2dong/ric/results/aim_2_results/aim2/big_bio_1translator_10.0_marker_domain_three_stage_10xvisium_filtered_gated_acrosscell_pcm_celltype_hubert_5cosine/val/pcm/shane_visualization/all_exp_visualization/top_gate_features_per_cluster.csv"
    dataset = ShaneSeqCellTypeDataset("/datasets/schwartz-lab/richard/data/internal/shane_aim2_validation/images/Mixed_NIR/")
    gate_correlation = pd.read_csv(gate_correlation_csv_path)
    all_cluster_ids = gate_correlation['Cluster'].unique().tolist()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    extractor = AutoModel.from_pretrained("owkin/phikon-v2").train().to(device)
    image_processor = pre_processing_phikon()
    feature_extractor = (image_processor, extractor)
    converter = init_spaghetti("/home/zf2dong/ric/results/spaghetti_results/models/ssim/croppatch_cycleGAN_pcm_he_lr0.005_rgb_large_ssim/version_0/checkpoints/epoch=99-step=59400.ckpt").to(device).train()
    # prep model
    model = GeneExpPredVisiumHD.load_from_checkpoint(model_path, num_genes = 18085, 
                                converter = converter, feature_extractor = feature_extractor).to(device)
    model.train()

    for name, module in extractor.named_modules():
        if "patch" in name.lower() and "embed" in name.lower():
            for param in module.parameters():
                param.requires_grad = True
            print(f"[Grad-CAM] Enabled gradients for: {name}")

    # get the patch embedding from ViT
    class ExtractorWrapper(torch.nn.Module):
        def __init__(self, extractor, image_processor, converter):
            super().__init__()
            self.extractor = extractor
            self.image_processor = image_processor
            self.converter = converter

        def forward(self, x):
            return self.extractor(self.image_processor(self.converter(x)))

    class DownStreamWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model.compute_gate(x.last_hidden_state[:, 0, :].view(1, -1), input_feature=True)

    def gate_selection_fn(vit, model, img_tensor, desired_gate_idx):
        vit.train()
        model.train()
        
        vit_out = vit(img_tensor)  # outputs.last_hidden_state: B, N+1, D
        preds = model(vit_out)  # should set downstream_model.last_gates inside forward

        return preds[-1,0,desired_gate_idx]

    final_extractor = ExtractorWrapper(extractor, image_processor, converter)
    downstream = DownStreamWrapper(model)

    # run grad-cam
    for img, label in tqdm(loader):
        img_tensor_all = img.to(model.device)
        img_tensor_all = img_tensor_all.squeeze(0) #remove the default batch dimension
        image_name = label[1][0].split("/")[-1].split(".")[0]
        for idx, each_img_tensor in tqdm(enumerate(img_tensor_all)):
            each_img_tensor = each_img_tensor.to(device)
            img_tensor = each_img_tensor.unsqueeze(0)
            all_heatmaps = []
            all_overlays = []
            for desired_gate_idx in range(960):
                heatmap, overlay, orig_viz = grad_cam_vit_gate_cls(
                    vit_model=final_extractor,
                    downstream_model=downstream,
                    img_tensor=img_tensor,
                    gate_selection_fn=lambda vit, model, img_tensor: gate_selection_fn(vit, model, img_tensor, desired_gate_idx),
                    patch_module=final_extractor.extractor.embeddings.patch_embeddings,
                    target_size=(img_tensor.shape[2], img_tensor.shape[3]),
                    device=device
                )
                all_heatmaps.append(heatmap)
                all_overlays.append(overlay)
            # take the average dim per cluster from csv
            for cluster_id in all_cluster_ids:
                gate_indices = gate_correlation[gate_correlation['Cluster']==cluster_id]['Gate Dimension'].values
                avg_heatmap = np.mean([all_heatmaps[i] for i in gate_indices], axis=0)
                avg_overlay = np.mean([all_overlays[i] for i in gate_indices], axis=0)
                show_heatmap_overlay(avg_heatmap, avg_overlay, orig_viz, name=f"{image_name}_img{idx}_cluster{cluster_id}")


