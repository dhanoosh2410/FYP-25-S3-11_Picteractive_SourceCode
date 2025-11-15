CSIT-25-S3-02: Show and Tell with a Computer

Team: FYP-25-S3-11

Picteractive –  run instructions (client/server split)

NOTE: 
-"Take a photo" feature on "What's this" page works only if permission for camera is given. (Works on chrome)
-First boot takes longer to install requirements
-All features work but might take longer to process 
-Ensure to run the project from the correct folder before running the commands. Refer to the file structure end of this text.
-Make sure the packages from requirement.txt are installed correctly.

1) Terminal 1 – API (backend)

#From the repo root, create and activate a virtual environment (Python 3.10+):
'''
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r server\requirements.txt

#Start the FastAPI server:
uvicorn server.main:app --reload --port 8000
'''

2)Terminal 2 – Frontend (client)

#From the repo root, go into the client folder and install dependencies (first time only):

'''
cd client
npm install

#Start the Vite dev server:
npm run dev
'''
Environment variables

The app reads .env from the repo root.

Important keys:
VITE_API_BASE=http://localhost:8000


3) Verify everything is running


(terminal 1)
API health check: open http://localhost:8000/api/health

(terminal 2)
Frontend: open http://localhost:5173  in your browser.





File Structure:
FYP-25-S3-11_Picteractive_SourceCode/
├─ README.txt                        # How to run backend + frontend
├─ .env                              # Environment config (API base URL, DB URL, etc.)
├─ index.html                        # Root HTML page served by Vite
│
├─ src/                              # Main React app source (used at runtime + in tests)
│  ├─ main.jsx                       # React entrypoint
│  ├─ App.jsx                        # All major UI/pages & routing
│  ├─ App.css                        # App-level styling
│  ├─ index.css                      # Tailwind/base styles
│  ├─ fonts.css                      # Font definitions
│  └─ assets/                        # Images used in UI + tests (e.g. scene1.jpg)
│
├─ client/                           # Frontend tooling (Vite + npm) needed to run UI
│  ├─ package.json                   # Vite/React config + scripts (dev/build/api)
│  ├─ package-lock.json              # Locked npm deps (for reproducible installs)
│  ├─ vite.config.js                 # Vite config (serves repo root src/)
│  ├─ tailwind.config.js             # Tailwind CSS config
│  ├─ postcss.config.js              # PostCSS pipeline
│  ├─ eslint.config.js               # Lint rules for React code
│  ├─ index.html                     # Vite’s internal entry (points to ../../src)
│  └─ src/
│     └─ main.jsx                    # Bridge entry that imports ../../src/main.jsx
│
├─ server/                           # Backend API needed to run web app
│  ├─ main.py                        # FastAPI app: all endpoints + ML/logic wiring
│  ├─ dev_api.py                     # Dev entry (loads .env, runs uvicorn with reload)
│  ├─ auth_DB.py                     # User/auth/session + SQLite DB models
│  ├─ story_gen.py                   # Story generation logic (BLIP captions + planner)
│  ├─ quiz_gen.py                    # Quiz generation with FLAN/GPT-2
│  ├─ showtellpyTorch.py             # Flickr8k CLIP retrieval captioner (builds index)
│  ├─ csvd_filter.py                 # Colour-vision filter API (simulate/daltonize)
│  ├─ requirements.txt               # Python dependencies needed on any machine
│  ├─ flickr8k_index.npz             # Precomputed CLIP image index model (runtime)
│  ├─ __init__.py                    # Marks `server` as a package
│  └─ data/                          # Runtime DB + required datasets
│     ├─ app.db                      # SQLite DB for users, sessions, settings
│     ├─ scenes/                     # Saved stories/drawings (user content)
│     ├─ images/                     # Flickr8k images (used by retrieval captioner)
│     ├─ sketches/
│     │  └─ train/                   # Quick, Draw! training sketches (for sketch labels)
│     ├─ captions.txt                # Flickr8k captions (runtime + training input)
│     └─ items.json                  # Saved items metadata for the app
│
├─ model_training/                   # Model training (for report / reproducibility)
│  ├─ flickr8k_train_caption_model.py   # CNN+LSTM caption model training + prep modes
│  ├─ caption_editor.py              # Helper to edit/clean caption text
│  ├─ flickr8k_train_requirements.txt  # Extra deps for training script
│  └─ readme_model_training.txt      # How the caption model and training pipeline work
│
├─ yolov8n.pt                        # YOLOv8n weights (used for object detection)
│
├─ run_fyp_tests.py                  # End‑to‑end test runner for 50 FYP scenarios
└─ package-lock.json                 # (Root) legacy lockfile; not needed for current client
