Picteractive – Image Captioning + Story Generation Web App
==========================================================


1. How the captioning model works
---------------------------------

There are THREE main components involved in captioning:

(1) Flickr8k dataset and captions.txt
-------------------------------------

- The Flickr8k dataset provides:
  - ~8,000 images
  - 5 human-written captions per image

- The raw dataset usually contains a file like:
  - `Flickr8k.token`
    Format: "image_name.jpg#0<TAB>Caption text ..."

- For our app, we convert the original token file into a simpler
  `captions.txt` format:

    image_name.jpg<TAB>caption 1
    image_name.jpg<TAB>caption 2
    ...

- `server/data/Images/`:
  contains all Flickr8k images.

- `server/data/captions.txt`:
  used by both:
    - the CLIP retrieval captioner (for runtime descriptions),
    - and the training script (for the CNN+LSTM model).



(2) Training script for CNN+LSTM captioning model
-------------------------------------------------

For collaboration purposes, a separate training script is provided:

    flickr8k_train_caption_model.py

This script is used to train the model on the Flickr8k dataset, and HOW accuracy (BLEU) and training
curves are computed.


3. Training the model and generating captions.txt
-------------------------------------------------

You can use the script `flickr8k_train_caption_model.py` in two modes:

A) Prepare captions.txt from the original Flickr8k.token
--------------------------------------------------------

1. Place the original token file at, for example:
   - `server/data/Flickr8k.token`

2. From the project root (where the script is located), run:

   (Windows PowerShell)
   > .\.venv\Scripts\activate
   > python flickr8k_train_caption_model.py ^
       --mode prepare ^
       --raw_tokens server/data/Flickr8k.token ^
       --out_captions server/data/captions.txt

   (Linux/macOS)
   $ source .venv/bin/activate
   $ python flickr8k_train_caption_model.py \
       --mode prepare \
       --raw_tokens server/data/Flickr8k.token \
       --out_captions server/data/captions.txt

3. This creates a `captions.txt` file in the format:

   image.jpg<TAB>Caption text

This is exactly the format required by `showtellpyTorch.py` for
the CLIP-based retrieval model used by the app at runtime.


B) Train a CNN+LSTM captioning model on Flickr8k
------------------------------------------------

The same script can train a simple encoder–decoder model:

- Encoder: ResNet-18 (pre-trained on ImageNet, last layer removed)
- Decoder: LSTM that generates captions word-by-word
- Vocabulary: built from all words in `captions.txt` plus special tokens:
    <pad>, <start>, <end>, <unk>
- Loss function: cross-entropy over next-token prediction
- Optimiser: Adam
- Metrics: Training loss, Validation loss, BLEU-4
- Outputs:
    - models/flickr8k_cnn_lstm.pt          (trained weights)
    - models/flickr8k_vocab.json           (vocabulary)
    - models/flickr8k_history.json         (loss + BLEU per epoch)
    - models/flickr8k_training_curves.png  (graph of loss + BLEU)


Steps:

1. Activate the virtual environment
-----------------------------------

Windows (PowerShell):
> py -3.11 -m venv .venv
> .\.venv\Scripts\activate

Linux/macOS:
$ python3 -m venv .venv
$ source .venv/bin/activate


2. Install dependencies
-----------------------

(From project root, with venv activated)

> pip install -r requirements.txt

Also ensure NLTK has the tokenizer:
> python -c "import nltk; nltk.download('punkt')"


3. Ensure Flickr8k images and captions are in place
---------------------------------------------------

- Place all Flickr8k images in:
    server/data/Images/

- Make sure you have a captions file:
    server/data/captions.txt

You can create this from the Flickr8k.token file using:

> python flickr8k_train_caption_model.py --mode prepare ^
    --raw_tokens server/data/Flickr8k.token ^
    --out_captions server/data/captions.txt


4. Run training
---------------

Example (Windows):

> python flickr8k_train_caption_model.py ^
    --mode train ^
    --images_dir server/data/Images ^
    --captions_file server/data/captions.txt ^
    --out_dir models ^
    --epochs 10 ^
    --batch_size 64

Example (Linux/macOS):

$ python flickr8k_train_caption_model.py \
    --mode train \
    --images_dir server/data/Images \
    --captions_file server/data/captions.txt \
    --out_dir models \
    --epochs 10 \
    --batch_size 64

The script will:

- print per-epoch:
  - train_loss
  - val_loss
  - BLEU-4

- save:
  - models/flickr8k_cnn_lstm.pt
  - models/flickr8k_vocab.json
  - models/flickr8k_history.json
  - models/flickr8k_training_curves.png

You can use `flickr8k_training_curves.png` in your report as the
"accuracy graph", and mention the final BLEU-4 as the model accuracy.


4. Running the backend API
--------------------------

Assuming the package structure:

- server/
    - __init__.py
    - main.py        (FastAPI app)
    - auth_DB.py
    - showtellpyTorch.py
    - story_gen/...

and your working directory is the project root.

1. Activate virtual environment (see section 3.1).

2. Start the backend:

Windows:
> uvicorn server.main:app --reload --port 8000

Linux/macOS:
$ uvicorn server.main:app --reload --port 8000

The API will be available at:
- http://127.0.0.1:8000

Useful endpoints:
- `POST /api/caption`         – generate description for an image
- `POST /api/story`           – generate story from caption + keywords
- `GET /` or `/health`        – basic health check (if defined)


5. Running the React frontend
-----------------------------

Assuming the React app (Vite) is located in the project root
(where `package.json` is).

1. Install Node dependencies:

   $ npm install

2. Start the dev server:

   $ npm run dev

By default, Vite runs on:
- http://127.0.0.1:5173

The frontend is configured to call the backend at:
- http://127.0.0.1:8000

Make sure the FastAPI backend is running before using the UI.


6. Summary for the report
-------------------------

- The deployed web app uses:
  - BLIP captioner (pre-trained) for a base caption.
  - CLIP + Flickr8k retrieval (showtellpyTorch.py) to retrieve similar
    images and stitch their human captions into a descriptive paragraph.
  - The final caption is normalised to start with:
    "The image shows ..."

- The training script:
  - demonstrates how a separate CNN+LSTM caption model was trained on
    the same Flickr8k dataset,
  - generates BLEU-4 scores and training/validation loss,
  - saves a PNG graph and JSON history that can be used as evidence
    of the training and accuracy evaluation process.

This satisfies the requirement to:
- show how the model is trained,
- show how captions.txt is generated from the dataset,
- and show accuracy (graph) and BLEU scores for the caption model.
