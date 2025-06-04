# Docker Image Deployment to GCP Artifact Registry

## Project and Registry Information
- **GCP Project ID:** diabetes-pred-461721
- **Artifact Registry Repository:** diabetes-pred
- **Location:** us-central1
- **Format:** Docker

## Authenticate Docker with GCP
```bash
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev

docker pull us-central1-docker.pkg.dev/diabetes-pred-461721/diabetes-pred/predictor-image:latest

docker build -t local-image-name .

docker tag local-image-name us-central1-docker.pkg.dev/diabetes-pred-461721/diabetes-pred/new-image:tag

docker push us-central1-docker.pkg.dev/diabetes-pred-461721/diabetes-pred/new-image:tag
