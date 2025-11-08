# Tic-Tac-Toe Transformer Fullstack
A full-featured tic-tac-toe application that uses a Transformer model powered by TensorFlow.js.

## Key Features

- 🤖 **Transformer model** built with TensorFlow.js that learns how to play.
- 🚀 **Optimized for Apple M2** processors.
- 🎮 **Two opponent modes**: the ML model or a minimax algorithm.
- 🤖 **Autoplay mode** where the model competes against the bot.
- 🔄 **WebSocket communication** for live training and inference.
- 📊 **Training dashboard** with detailed statistics.

## Tech Stack

- **Frontend**: Vue.js 3, Vite.
- **Backend**: Node.js, WebSocket (`ws`).
- **Machine Learning**: TensorFlow.js (Node.js backend).
- **Performance**: Worker threads for dataset generation.

## Installation

### Option 1: Docker (recommended) 🐳

The easiest way to run the project without dependency issues.

## Quick Start

### ⚠️ Important: GPU vs CPU mode

**`npm start`** runs the project **locally** (CPU mode). GPU acceleration is unavailable without Docker.

**For GPU mode** use:
```bash
npm run docker:up
# or
npm run start:gpu
```

### Local run (CPU mode)

```bash
# Install dependencies (first time)
npm install

# Start the entire project locally (CPU)
npm start

# Stop all processes and containers
npm stop

# Restart the project
npm restart
```

This spins up:
- Server on port 8080 (WebSocket) — **CPU mode**  
- Client on port 5173 (Vite dev server)

**Notes:** 
- `npm start` — local launch, **CPU only**
- `npm run docker:up` or `npm run start:gpu` — Docker launch with **GPU support**
- `npm stop` stops every Docker container and local process related to the project
- `npm restart` stops everything and starts it again

### Docker run

```bash
# Make sure Docker Desktop is running
docker ps

# GPU build (requires NVIDIA GPU and nvidia-container-toolkit)
docker-compose -f docker-compose.gpu.yml up --build

# Or CPU build
docker-compose up --build
```

Services will be available at:
- Server: `ws://localhost:8080` (WebSocket)
- Client: `http://localhost:5173`

See also: [QUICK_DOCKER_START.md](QUICK_DOCKER_START.md) or [DOCKER_SETUP.md](DOCKER_SETUP.md)

## Verifying GPU usage

Check server logs on startup:
```
[TFJS] Using tfjs-node-gpu backend (CUDA support)
[TFJS] Backend: tensorflow (gpu)
[TrainTTT3] GPU acceleration: ENABLED ✓
```

If GPU is unavailable:
```
[TFJS] WARNING: Backend is not GPU! Check NVIDIA/CUDA/cuDNN installation.
[TrainTTT3] GPU acceleration: DISABLED ✗
```

**All TensorFlow.js operations automatically run on the GPU when `@tensorflow/tfjs-node-gpu` is available**, including tensors, `model.fit()`, and `model.predict()`.

### Option 2: Local installation

#### Requirements

- **Node.js v18.x or v20.x** (LTS releases)
- **Visual Studio Build Tools 2022** with the “Desktop development with C++” workload (Windows)
- **CUDA Toolkit** (optional, for GPU/CUDA support)

```bash
# Install dependencies
npm install

# Start server and client together
npm start

# Or separately:
npm run server  # Server at ws://localhost:8080
npm run client  # Client at http://localhost:5173
```

**Tip:** Switch to the Docker option if installing TensorFlow.js on Windows causes issues.

### CPU and GPU support

The project automatically detects and uses:
- **CPU (x86_64)** via `@tensorflow/tfjs-node`
- **CUDA GPU** via `@tensorflow/tfjs-node-gpu` (if available)

Inspect the server logs to confirm the backend that is active.

## Usage

1. **Train the model** — press “Train” to start from scratch.
2. **Reset the model** — use “Clear Model” to delete the saved state.
3. **Choose opponent** — switch between “Model” and “Algorithm”.
4. **Autoplay** — enable “Auto Play” to watch the model play against the bot.

## Project structure

```
.
├── client/          # Vue.js frontend
│   ├── src/
│   │   ├── App.vue   # Root component
│   │   └── main.js
│   └── package.json
├── server/          # Node.js backend
│   ├── src/
│   │   ├── model_transformer.mjs  # Transformer model
│   │   ├── dataset.mjs            # Dataset generation
│   │   ├── tic_tac_toe.mjs        # Game logic and minimax
│   │   └── tf.mjs                 # TensorFlow.js setup
│   ├── server.mjs   # WebSocket server
│   └── service.mjs  # Business logic
└── package.json
```

## M2 optimizations

- Dataset generation in parallel via worker threads.
- Alpha-beta pruning in the minimax algorithm.
- Tuned batch size and TensorFlow.js settings.
- Native TensorFlow bindings for extra performance.

## License

Distributed under the [MIT License](LICENSE). You may use, copy, modify, and distribute the code as long as you retain the copyright notice and license text.
