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
```

**2. Install Requirements:**

```
python -m pip install -r requirements.txt
```

**3. Consent to use Mistral from Huggingface**

* Login to hugging face in the terminal: 

```
huggingface-cli login
```

* Enter your personal access token, to create one go to https://huggingface.co/settings/tokens
* Go to https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 and accept the agreement to access the model

---

### IMPORTANT: Checking GPU Usage and Running the Project

1. **Check GPU availability**

   Open PowerShell or terminal and run:

   ```powershell
   nvidia-smi
   ```
   
   * **Find the CUDA version** at the top of the output
   * Note: For non-NVIDIA GPUs, the project will fallback to CPU automatically.


2. **Install dependencies for your GPU/CPU**

   Visit PyTorch to get the correct installation command: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

   * OS: Windows
   * Package: Pip
   * Language: Python
   * CUDA: Select your installed version (matches `nvidia-smi`) or None for CPU

   Copy the generated command and run it in your terminal or virtual environment. The command should look something like this:

   ```
   pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
   ```

5. **Check GPU usage in Python (optional)**

   ```python
   import torch

   if torch.cuda.is_available():
       print(f"Using GPU: {torch.cuda.get_device_name(0)}")
   else:
       print("No NVIDIA GPU found, using CPU")
   ```

---

**4. Prepare your dataset:**

Create `data/ground_truth.jsonl` with the following format:

```
{"prompt": "What objects are visible in the video?", "task_type": "object_detection", "output_modality": "text", "complexity": "simple", "temporal_context": "none", "spatial_context": "none"}
{"prompt": "At what time does the person start running?", "task_type": "temporal_localisation", "output_modality": "text", "complexity": "temporal", "temporal_context": "during", "spatial_context": "none"}
{"prompt": "Is there a car in front of the building?", "task_type": "object_detection", "output_modality": "text", "complexity": "simple", "temporal_context": "none", "spatial_context": "in_front"}
{"prompt": "Why did the cat jump off the table?", "task_type": "question_answering", "output_modality": "text", "complexity": "causal", "temporal_context": "none", "spatial_context": "none"}
```

**5. Start training:**

```
python training/train.py
```

**6. Run the pipeline:**

```
python run_pipeline.py
```

**7. Run tests:**

* To be done... :)
