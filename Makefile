
### -------------------------- Docker commands ----------------------------- ###

IDU=$(shell id -u)
IDG=$(shell id -g)

docker_clean:
	docker kill assignement_running || true

docker_build:
	docker build -t assignement_running --build-arg USER_ID=$IDU --build-arg GROUP_ID=$IDG .

docker_run_cpu:
	docker run --rm -it  --shm-size 4G --cpu-shares 4 -v /home/gregoryscafarto/ecovadis:/workspace -p 8888:8888 \
--name assignement_running assignement_running

docker_run:
	docker run --rm -it -d --shm-size 4G --gpus all -v $(shell pwd):/workspace -p 8888:8888 \
--name assignement_running_v1 assignement_running #--runtime=nvidia

docker_jupyter_cpu:
	docker run --rm -it -d --shm-size 4G --cpu-shares 4 -v $(shell pwd):/workspace -p 8888:8888 \
--name assignement_running assignement_running "bash" "-c" "make jupyter"

docker_jupyter:
	docker run --rm -d -it --shm-size 4G --gpus all -v $(shell pwd):/workspace -p 8888:8888 \
--name assignement_running assignement_running "bash" "-c" "make jupyter"
# --runtime=nvidia

run-flask:
	cd endpoints && nohup flask run  &
	
run-UI:	
	cd UI && nohup streamlit run streamlit_main.py &

clear_app:
	chmod  +x clear.sh
	./clear.sh


download-file:
	curl -L -o models/roberta_sentiment_classification "https://drive.google.com/uc?export=download&id=1qdQNi17UOp4P0qq_qb7u40C97yLWHjQL"





jupyter:
	jupyter-notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=''