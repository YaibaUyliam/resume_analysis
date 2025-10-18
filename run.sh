docker run --rm --privileged --gpus "device=0" --shm-size="16g" --name resume_analysis \
-p 9001:9001 \
yaibawiliam/resume_analysis:$1


# docker run --rm --gpus all --shm-size="8g" --name resume_analysis -p 9001:9001 yaibawiliam/resume_analysis:$1
# --runtime=nvidia