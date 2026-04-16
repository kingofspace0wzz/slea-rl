FROM verlai/verl:sgl055.latest
# FROM lmsysorg/sglang:v0.4.6.post5-cu124

# 1. Install system dependencies (git is required for cloning tau-bench)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Create workspace directory
RUN mkdir -p /app/workspace


# 3. Install Python dependencies
RUN pip install gymnasium==0.29.1
RUN pip install stable-baselines3==2.6.0
RUN pip install alfworld && alfworld-download -f

# 4. Install vllm
RUN pip install vllm==0.11.0
# RUN pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

RUN pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir -v

# 5. Login to wandb
RUN wandb login 6e6f377b9c85f1a12f1322909e6e38195e1572d9

# 6. Set the final working directory
WORKDIR /app/workspace