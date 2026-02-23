# Use a base image with uv pre-installed
FROM ghcr.io/astral-sh/uv:bookworm-slim

# Install system dependencies required for matplotlib/mplfinance (fonts, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy dependency definitions
COPY pyproject.toml uv.lock* ./

# Install dependencies (uv will download the required Python version here)
RUN uv sync --no-install-project

# Copy source code
COPY src/ src/

# Run the application
CMD ["uv", "run", "src/main.py"]