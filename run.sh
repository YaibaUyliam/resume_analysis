docker run --rm --privileged --gpus "device=0" --shm-size="8g" --name resume_analysis \
-p 9001:9001 \
yaibawiliam/resume_analysis:$1



# --runtime=nvidia