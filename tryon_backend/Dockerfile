FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY server.py model.py ./

# Pre-download models at build time so first request is fast
RUN python -c "\
from diffusers import AutoPipelineForInpainting; \
import torch; \
AutoPipelineForInpainting.from_pretrained(\
    'diffusers/stable-diffusion-xl-1.0-inpainting-0.1', \
    torch_dtype=torch.float16, variant='fp16' \
)"

EXPOSE 8000

CMD ["python", "server.py"]
