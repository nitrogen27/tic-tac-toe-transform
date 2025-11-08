## Tic-Tac-Toe Transformer Fullstack

An interactive tic-tac-toe application where a Vue 3 frontend, Node.js backend, and TensorFlow.js Transformer model train together, compete, and visualize learning progress. Optimized for Apple M2 but compatible with standard CPU/GPU environments.

---

### Highlights

- 🤖 Train the Transformer model live while playing.
- 🎯 Two opponent modes: the ML model or a minimax engine with alpha-beta pruning.
- 🔄 WebSocket channel between client and server for real-time training and inference.
- 🧪 Fully automated “model vs bot” mode for rapid dataset collection.
- 📊 Dashboard that visualizes training metrics and game statistics.

---

### Technology Stack

- **Frontend**: Vue.js 3 + Vite, Composition API, Tailwind CSS (optional).
- **Backend**: Node.js 18/20, WebSocket (`ws`), worker threads for dataset generation.
- **ML**: `@tensorflow/tfjs-node` and `@tensorflow/tfjs-node-gpu`, custom Transformer architecture.
- **Infrastructure**: Docker / Docker Compose, NVIDIA GPU support via `nvidia-container-toolkit`.

---

### Project Architecture

```
.
├── client/               # Vite + Vue 3 frontend
│   └── src/
│       ├── App.vue        # Root component with game modes
│       └── components/    # UI elements, metrics visualizations
├── server/               # Node.js WebSocket server
│   └── src/
│       ├── server.mjs     # Entry point, session management
│       ├── service.mjs    # Business logic, move orchestration
│       ├── model_transformer.mjs  # Model definition and training
│       ├── dataset.mjs    # Data generation and augmentation
│       └── tic_tac_toe.mjs # Game logic + minimax
└── docker-compose*.yml    # CPU/GPU launch templates
```

---

### Running the Project

**Local (CPU):**

```bash
npm install
npm start        # starts the server on :8080 and the frontend on :5173
```

**Docker:**

```bash
# CPU build
docker-compose up --build

# GPU build (requires NVIDIA GPU + nvidia-container-toolkit)
docker-compose -f docker-compose.gpu.yml up --build
```

Watch the server logs: when the GPU is connected you’ll see `Using tfjs-node-gpu backend`.

---

### Model Lifecycle

1. **Data generation** — worker threads create move batches (manual games, autoplay, minimax simulations).
2. **Training** — the `TrainTTT3` model trains with TensorFlow.js and streams updates to the frontend.
3. **Inference** — the client sends board state to the server, the model responds with the best move.
4. **Visualization** — the UI displays move history, win probabilities, and training metrics.

---

### Debugging & Monitoring

- Enable verbose logs with `DEBUG=train:*` before starting the server.
- Use `TF_CPP_MIN_LOG_LEVEL=0` to inspect TensorFlow backend details.
- In Docker GPU mode, verify container activity with `nvidia-smi`.
- Inspect WebSocket traffic via the DevTools Network tab (Frames).

---

### Roadmap

- [ ] Add a “model vs minimax” tournament mode with adjustable difficulty.
- [ ] Export the trained model and integrate it with a mobile client.
- [ ] Support multiple users and private matches via authentication.
- [ ] Deploy to the cloud (Render/Heroku/Fly) with automated Docker builds.
- [ ] Set up CI/CD pipelines for test and lint runs.

---

### Helpful Links

- Repository: `https://gitlab.com/Nitrogenn/tic-tac-toe-transform`
- Quick start: `README.md`
- Docker guides: `QUICK_DOCKER_START.md`, `DOCKER_SETUP.md`
- Discussions & bug reports: GitLab Issues

Need more docs? Create additional Wiki pages and reference them in “Helpful Links”.

