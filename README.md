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

docker compose up -d

---
docker run --name container-fastapi-langchain -p 8000:8000 image-fastapi-langchain:latest

docker exec -ti fastapi-langchain bash

```

### For me

```bash
docker build -t doantronghieu/image-fastapi-langchain:latest -f deploy/docker_k8s/docker-files/Dockerfile.FastApi-LangChain .
docker build -t doantronghieu/image-streamlit:latest -f deploy/docker_k8s/docker-files/Dockerfile.Streamlit .

docker push doantronghieu/image-fastapi-langchain:latest
docker push doantronghieu/image-streamlit:latest

docker run -d --name container-fastapi-langchain -p 8000:8000  doantronghieu/image-fastapi-langchain:latest
docker run -d --name container-streamlit -p 8051:8051 doantronghieu/image-streamlit:latest
```
