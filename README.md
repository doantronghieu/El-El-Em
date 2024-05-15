# Large Languge Model-powered applications

## Indexing Flow

![Indexing Flow](./diagrams/Indexing%20Flow.jpeg)

## RAG Flow

![RAG Flow](./diagrams/RAG%20Flow.jpeg)

## Agent Flow

![Agent Flow](./diagrams/Agent%20Flow.jpeg)

## Docker

```bash
cd Apps
docker build -t image-fastapi-langchain:latest -f deploy/docker_k8s/docker-files/Dockerfile.FastApi-LangChain .
docker run -d --name container-fastapi-langchain -p 8000:8000 image-fastapi-langchain:latest
docker run --name container-fastapi-langchain -p 8000:8000 image-fastapi-langchain:latest

docker exec -ti container-fastapi-langchain bash

```
