# Crop Management Backend

FastAPI backend foundation for the crop diagnosis and advisory platform.

## Run locally

1. Create a virtual environment and install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set `MONGODB_URI` and `MONGODB_DATABASE`.
   The application loads this file at startup. A process manager or shell
   profile can also provide the same variables.

3. Start the API from the `backend` directory:

   ```powershell
   uvicorn app.main:app --reload
   ```

## Initial endpoints

### Health

```powershell
curl http://127.0.0.1:8000/api/v1/health
```

The health endpoint reports whether MongoDB is configured and reachable. It does
not hide database connectivity failures.

### Create a diagnosis

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/diagnoses `
  -F "image=@C:\path\to\leaf.jpg" `
  -F "crop=tomato" `
  -F "farmer_id=farmer-001" `
  -F "context_json={}"
```

The endpoint stores diagnosis metadata and returns `pending_model_inference`.
After the tomato CNN checkpoint is trained and available, the same endpoint
accepts tomato images, runs CNN inference, stores the disease and confidence,
and returns `completed`. Non-tomato crops are rejected with
`unsupported_crop`. If the checkpoint is missing, the backend still starts,
but diagnosis inference returns HTTP 503 and no diagnosis document is created.

### Retrieve a diagnosis

```powershell
curl http://127.0.0.1:8000/api/v1/diagnoses/<diagnosis_id>
```

## Tomato CNN

The first model uses a torchvision ResNet18 transfer-learning classifier with
the approved ten-class PlantVillage tomato mapping. The training script
expects the approved release arranged as:

```text
data/train/<Tomato class>/*.jpg
data/test/<Tomato class>/*.jpg
```

From the `backend` directory, install the dependencies and train only after
the dataset has been prepared locally:

```powershell
pip install -r requirements.txt
python -m ml.train `
  --train-root C:\path\to\plantvillage\data\train `
  --test-root C:\path\to\plantvillage\data\test `
  --epochs 10 `
  --output ml\artifacts\tomato_resnet18.pt
```

Training creates a grouped validation split from the provided training split,
using the filename prefix before `___` as the related-image group, and leaves
the provided test split for final evaluation. It does not use the rejected
BaxaDE, IP102, or Mendeley datasets.

Evaluate the trained checkpoint with:

```powershell
python -m ml.evaluate `
  --test-root C:\path\to\plantvillage\data\test `
  --checkpoint ml\artifacts\tomato_resnet18.pt `
  --report ml\reports\tomato-evaluation.json
```

The API loads the checkpoint once at startup. It never retrains on startup.
