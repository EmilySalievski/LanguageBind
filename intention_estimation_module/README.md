# Intention Estimation Module

## How to Run & Test

**1. Clone the project and set up a virtual environment:**

```
git clone <your-repo-url>
cd intention_estimation
python3 -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run the pipeline interactively:**

* Login to hunnging face in the terminal: huggingface-cli login
* Enter your personal access token, to create one go to https://huggingface.co/settings/tokens
* Go to https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 and accept the agreement to access the model

```
python run_pipeline.py
```

**4. Run tests:**

```
pytest tests/
```

## How to Fine-Tune the Intention Estimation Model

**1. Prepare your dataset**

Create `data/intents.csv` with the following format:

```
text,label
"Summarize the whole video.","summarize"
"Find the screenshot of clogged nozzle.","find_screenshot"
"Give a transcript of the calibration section.","transcript_answer"
"Show the segment where PLA is compared to ABS.","locate_segment"
```

**Tips:**

* First row must be the header: `text,label`
* Wrap text in double quotes if it contains commas.

**2. Start training:**

```
python training/train.py
```

**3. Evaluate a sample:**

```
python training/evaluator.py
```

## Notes

* Recommended to run on GPU if available (especially for Mistral-7B fallback).
* For lower VRAM setups, consider using quantized models with `bitsandbytes` to avoid out-of-memory errors.

## To run on GPU

* First uninstall any other version, then install version appropriate with your gpu, use ChatGPT and show it the result 
of running: nvidia-smi (or similar for your appropriate GPU)

```
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```