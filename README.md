# NOAA NMFS Optics spicy_ir_seal_detector Model Deployment (Ultralytics YOLO)

Welcome! This repository is a starting template for deploying custom **Ultralytics YOLO** Computer Vision models into the Optics SI Airflow ecosystem on Google Cloud. 

Our infrastructure requires models to run inside isolated Docker containers, expose an HTTP endpoint, and communicate with Google Cloud Storage (GCS). We have pre-written the heavy lifting for you. This repository expects you to provide a `.pt` weights file, and it will automatically handle downloading inputs, running inference using Ultralytics, and formatting the output into csv format. 

---

## 🟢 Phase 1: Local Setup & Testing

Before deploying to the cloud, you should bake your custom model into the container and test it locally.

### 🏁 Step 1: Add Weights

1. Clone this repositorty to your local machine (or Google Cloud Workstation).
2. **CRITICAL:** You must place your default YOLO weights file in the root of the repository and name it exactly `model.pt`. 

```bash
git clone https://github.com/eem1/optics-models-ultralytics-detection-ir.git
cd optics-models-ultralytics-detection_ir
# Copy your weights in... e.g. yolo11s_IR_2025_best.pt
cp /path/to/your/best.pt ./model.pt
```

### 💻 Step 2: Test Locally

We recommend using the Google Cloud workstations for testing, as they already have `docker` and `gcloud` installed.

**1. Authenticate with Google Cloud**
Ensure you have local credentials so the container can download test files from GCS:
```bash
gcloud auth application-default login
```

```bash
chmod +r ~/.config/gcloud/application_default_credentials.json
```

**2. Build the Docker Container**
```bash
docker build -t optics-yolo-ir-despeckle-normalize-model:latest .
```

**3. Run the Container**
*(This maps your local GCP credentials into the container so it can access buckets)*
```bash
docker run -p 8080:8080 \
  -v ~/.config/gcloud:/tmp/.config/gcloud \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/.config/gcloud/application_default_credentials.json \
  -e GOOGLE_CLOUD_PROJECT=ggn-nmfs-osi-dev-1 \
  optics-yolo-ir-despeckle-normalize-model:latest
```

If run the container from your local Windows termial:

```bash
docker run -p 8080:8080 `
  -v ${env:APPDATA}\gcloud:/tmp/.config/gcloud `
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/.config/gcloud/application_default_credentials.json `
  -e GOOGLE_CLOUD_PROJECT=ggn-nmfs-osi-dev-1 `
  optics-yolo-ir-despeckle-normalize-model:latest
```

**4. Send a Test Request**

With your container running, open a new terminal and send a JSON payload to test it. 

***(Ensure you have updated the GCS paths in your test payload to point to actual media files you have access to).***


```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d @test_payloads/yolo_ir_test_payload_example.json
```

---

#### ⚙️ Configuration & Features

Note: See `test_payloads/yolo_ir_test_payload_example.json` on how to pass Model normalization params to your model.

This template supports several advanced features through the JSON Airflow payload.

#### YOLO Inference Options (Kwargs Passthrough)
Any key-value pairs you place inside the `"options"` object of your config will be passed directly to the `YOLO.predict()` method as `**kwargs`. This means you can control confidence, IOU, image size, and more, directly from the Airflow UI without changing code.

```json
"config": {
    "options": {
        "conf": 0.25,
        "iou": 0.45,
        "imgsz": 1280
    }
}
```

#### Dynamic Weights Override
While the container is built with a default `model.pt` baked in, you can instruct it to download a different set of weights at runtime by providing a GCS URI in the `"weights"` key.

```json
"config": {
    "weights": "gs://my-bucket/path/to/experimental_weights.pt",
    "options": { ... }
}
```
**⚠️ Performance Warning:** Using dynamic weights provides great flexibility for A/B testing, but it has performance tradeoffs. Downloading large `.pt` files from GCS at runtime will increase the latency of the job startup and consume more network bandwidth. For highly scaled production jobs, baking the weights into the container image is preferred.


#### KWCOCO Support 
**Note: The current implementation supports CSV format**. For KWCOCO format, prefer to the document [NOAA NMFS Optics Model Deployment Template](https://github.com/csbrown-noaa/optics-models-ultralytics-detection)

This template natively handles both images and videos. All outputs are formatted into the **KWCOCO (Kitware COCO)** JSON specification.

*   **For Images:** Standard COCO image and annotation records are created.
*   **For Videos:** The template automatically registers the video in the KWCOCO `"videos"` array. As YOLO processes the video frame-by-frame, it registers each frame in the `"images"` array with a `"video_id"` and `"frame_index"`, ensuring downstream tools can reconstruct the temporal tracking data. Bounding boxes are automatically converted from YOLO's `xyxy` format to COCO's `[x, y, width, height]`.

### 💻 Step 3: Deloy Your Model to Google Cloud
**1. Register Your Model to the Artifact Registry**

```batch
docker tag optics-yolo-ir-despeckle-normalize-model:latest us-central1-docker.pkg.dev/ggn-nmfs-osi-dev-1/nmfs-dev-uc1-docker-repository/optics-yolo-ir-despeckle-normalize-model:latest 

docker push us-central1-docker.pkg.dev/ggn-nmfs-osi-dev-1/nmfs-dev-uc1-docker-repository/optics-yolo-ir-despeckle-normalize-model:latest

```

**2. Register Your Model to Google Cloud Airflow**

To run the pipeline in Airflow you need to register your model to the gcs [model_runtime_definitions.json](gs://ggn-nmfs-osi-dev-1-data/configs/model_runtime_definitions.json)

Since the `optics-yolo-ir-despeckle-normalize-model` is already registered, the following steps are essential when you need to update the model name or docker image.

Download the gcs [model_runtime_definitions.json](gs://ggn-nmfs-osi-dev-1-data/configs/model_runtime_definitions.json)

Search for `optics-yolo-ir-despeckle-normalize-model` in the JSON file and update its configruration.

**3. Trigger the Pipeline via Airflow**

Note: See `dag_files/nmfs-optics-yolo-ir-despeckle-normalize-config.yaml` on how to pass Model normalization params to your model. Upload this file to GCS as it'll be an input config for Airflow.

Go to Google Cloud Console, project ID `ggn-nmfs-osi-dev-1`, search Bar `Managed Airflow`, select `composer-env1`, `Open Airflow UI`, select `nmfs-optics-pipeline-longrunning-dag`, Trigger DAG arrow

In the form, fill out the required fields:

Model Type: `optics-yolo-ir-despeckle-normalize-model`

YAML Config File: `gs://bucket/your_folder/configs/nmfs-optics-yolo-ir-despeckle-normalize-config.yaml`

Output Folder: `your_folder/your-output-folder/`

Click "Trigger" button

Wait until the job complete and check your output folder.