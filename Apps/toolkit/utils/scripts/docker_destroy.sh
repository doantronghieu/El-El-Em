#!/bin/bash

docker stop $(docker ps -q)
docker rm $(docker ps -aq)
docker rmi $(docker images -q)

docker system prune -a -f 
docker container prune -f 
docker buildx prune -f
docker network prune -f 