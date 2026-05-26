# Enable CatVTON-MaskFree weights

CatVTON-MaskFree is the original authors' improved version, trained on VITON-HD (16k pairs at 512px). It overlays attention-layer weights on top of base CatVTON.

**License:** CC-BY-NC-SA-4.0 + gated. Non-commercial use only. You must accept the gate form before downloading.

## One-time setup

### 1. Accept the gate

- Go to https://huggingface.co/zhengchong/CatVTON-MaskFree
- Sign in to HuggingFace (create account if needed)
- Fill out the gate form on the page (name, country, affiliation, agree to terms)
- Wait for the "You have been granted access" confirmation (usually instant)

### 2. Get an HF access token

- Go to https://huggingface.co/settings/tokens
- Create new token → **Read** scope → copy

### 3. Set env vars on the backend host

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
export CATVTON_MASKFREE_VARIANT=vitonhd-16k-512
```

Variants you can choose:
| Variant | Trained on | Best for |
|---|---|---|
| `vitonhd-16k-512` | VITON-HD 16k @ 512px | **Upper body, jackets, shirts** ← recommended |
| `dresscode-16k-512` | DressCode 16k @ 512px | Broader garment types (incl. dresses, pants) |
| `mix-48k-1024` | 48k mixed @ 1024px | Highest quality, slower, more VRAM |

### 4. Restart the backend

```bash
cd tryon_backend
python server.py
```

Expected log lines on startup:
```
Loading CatVTON model from zheng-chong/CatVTON …
Fetching MaskFree weights: zhengchong/CatVTON-MaskFree/vitonhd-16k-512/attention/model.safetensors
MaskFree overlay applied — N layers updated, M unmatched (base CatVTON weights retained).
CatVTON direct ready — 6-step DDIM. Garment-latent concat try-on.
```

If you see:
```
MaskFree download failed (...). Have you accepted the gate ... ?
```
Re-check steps 1-3. The backend keeps running on base CatVTON, so nothing breaks.

## To turn it off

```bash
unset CATVTON_MASKFREE_VARIANT
```

Restart — backend reverts to base CatVTON.
