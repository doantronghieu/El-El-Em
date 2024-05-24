# Large Languge Model-powered applications

## Indexing Flow

![Indexing Flow](./diagrams/Indexing%20Flow.jpeg)

## RAG Flow

![RAG Flow](./diagrams/RAG%20Flow.jpeg)

## Agent Flow

![Agent Flow](./diagrams/Agent%20Flow.jpeg)

## Docker

```bash
docker build -t doantronghieu/llm-fastapi:latest -f deploy/docker_k8s/docker-files/Dockerfile.fastapi .
docker build -t doantronghieu/llm-streamlit:latest -f deploy/docker_k8s/docker-files/Dockerfile.streamlit .

docker push doantronghieu/llm-fastapi:latest
docker push doantronghieu/llm-streamlit:latest

---

# Test
docker run -p 8000:8000 doantronghieu/llm-fastapi:latest

docker exec -ti fastapi bash

docker compose up -d
docker compose down

---

# For AWS EC2 x86
docker build --platform linux/amd64 -t doantronghieu/llm-fastapi:amd64 -f deploy/docker_k8s/docker-files/Dockerfile.fastapi .
docker build --platform linux/amd64 -t doantronghieu/llm-streamlit:amd64 -f deploy/docker_k8s/docker-files/Dockerfile.streamlit .

docker push doantronghieu/llm-fastapi:amd64
docker push doantronghieu/llm-streamlit:amd64

---

eksctl create cluster -f deploy/docker_k8s/eks-cluster.yaml
eksctl delete cluster --wait --disable-nodegroup-eviction -f deploy/docker_k8s/eks-cluster.yaml 
```
