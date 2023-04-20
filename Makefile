mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
dataset_dir := $(mkfile_dir)data/visdrone/MOT/
output_dir := $(mkfile_dir)data/visdrone/COCO/
train_output := $(mkfile_dir)trainings/

# As the install needs GPU and it can't be done in Dockerfile
# We do it in this step after.
build:
	docker buildx build -t motrv2:tmp --target=final-form .
	docker run --gpus all -v $(mkfile_dir):/motrv2 --name tmp_container_motrv2 motrv2:tmp \
		-c "python /motrv2/motmodels/ops/setup.py build --build-base=motmodels/ops/ install"
	docker commit tmp_container_motrv2 motrv2:dev
	docker rm tmp_container_motrv2

run:
	docker run --rm -it --entrypoint zsh -v $(HOME):$(HOME) -v /data:/data --ipc=host \
		--workdir=$(mkfile_dir) --gpus device=0 motrv2:dev

launch-jupyter:
	jupyter-notebook --allow-root --ip 0.0.0.0 --port 6565