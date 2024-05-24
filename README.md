# Large Languge Model-powered applications

## Indexing Flow

![Indexing Flow](./diagrams/IndexingFlow.jpeg)

## RAG Flow

![RAG Flow](./diagrams/RAGFlow.jpeg)

## Agent Flow

![Agent Flow](./diagrams/AgentFlow.jpeg)

## System Architecture

![System Architecture](./diagrams/SystemDesign.jpeg)

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

# For AWS EC2 arm
docker pull doantronghieu/llm-fastapi:latest
docker pull doantronghieu/llm-streamlit:latest

---

eksctl create cluster -f deploy/docker_k8s/eks-cluster.yaml --dry-run

eksctl create cluster -f deploy/docker_k8s/eks-cluster.yaml
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

eksctl upgrade cluster --config-file deploy/docker_k8s/eks-cluster.yaml

eksctl delete cluster --wait --disable-nodegroup-eviction -f deploy/docker_k8s/eks-cluster.yaml 

```
