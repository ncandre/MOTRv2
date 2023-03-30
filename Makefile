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
		-c "python /motrv2/models/ops/setup.py build --build-base=models/ops/ install"
	docker commit tmp_container_motrv2 motrv2:dev
	docker rm tmp_container_motrv2

run:
	docker run --rm -it --entrypoint zsh -v $(HOME):$(HOME) -v /data:/data --ipc=host \
		--workdir=$(mkfile_dir) --gpus device=3 -p 6565:6565 motrv2:dev

launch-jupyter:
	jupyter-notebook --allow-root --ip 0.0.0.0 --port 6565

download-models:
	mkdir $(mkfile_dir)models
	cd $(mkfile_dir)models && \
	wget https://vision.in.tum.de/webshare/u/meinhard/trackformer_models_v1.zip && \
	unzip trackformer_models_v1.zip && \
	rm trackformer_models_v1.zip

# Make sure that you've downloaded the models in ./models and prepared the snakeboard example.
# ffmpeg -i data/snakeboard/snakeboard.mp4 -vf fps=30 data/snakeboard/%06d.png
# Check the readme for more info
demo-snakeboard:
	python src/track.py with \
		dataset_name=DEMO \
		data_root_dir=data/snakeboard \
		output_dir=data/snakeboard \
		write_images=pretty

download-mot17:
	cd $(mkfile_dir)data && \
	wget https://motchallenge.net/data/MOT17.zip && \
	unzip MOT17.zip && \
	rm MOT17.zip
	python src/generate_coco_from_mot.py

train-mot17:
	python src/train.py with \
		mot17 \
		deformable \
		multi_frame \
		tracking \
		output_dir=/home/marbelot/Desktop/PP_trackformer/trainings/mot17_deformable_multi_frame


# Download visDrone multi object tracking:
# ICI: https://s3.console.aws.amazon.com/s3/buckets/earthcube-projects?region=eu-central-1&prefix=1-customer_project/FMV/raw/datasets/visdrone/&showversions=false
download-visdrone:
	mkdir -p $(dataset_dir)train
	aws s3 sync s3://earthcube-projects/1-customer_project/FMV/raw/datasets/visdrone/VisDrone2019-MOT-train/VisDrone2019-MOT-train/ \
		$(dataset_dir)train
	mkdir -p $(dataset_dir)test
	aws s3 sync s3://earthcube-projects/1-customer_project/FMV/raw/datasets/visdrone/VisDrone2019-MOT-val/ \
		$(dataset_dir)test

transform-visdrone:
	python $(mkfile_dir)visdrone_to_coco_format.py -i $(dataset_dir)train -o $(output_dir)
	python $(mkfile_dir)visdrone_to_coco_format.py -i $(dataset_dir)test -o $(output_dir)

# Make sure to update the dataset path given in cfgs/train_visdrone.yaml to match your local config
# before launching the cmd.
train-visdrone:
	python src/train.py with \
		visdrone-vehicle-mot \
		deformable \
		multi_frame \
		tracking \
		output_dir=$(train_output)visdrone-vehicle

compress-zip:
	cd $(mkfile_dir) && zip -r ../trackformer_visdrone.zip . -x data/\* models/\* docs/\* .\*


# Video .mp4 must be turned into a list of jpg image. You can use
# ffmpeg -i <my_video.mp4> -vf fps=30 -ss 10 -to 15 %06d.png
# -ss the start time in seconds | -to the end time in seconds.
# (https://stackoverflow.com/questions/40088222/ffmpeg-convert-video-to-images)
# And after to turn a list of image into a video
# ffmpeg -framerate 30 -i %06d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
# (https://askubuntu.com/questions/610903/how-can-i-create-a-video-file-from-a-set-of-jpg-images)
# To reduce the quality of the video
#  
# (https://unix.stackexchange.com/questions/28803/how-can-i-reduce-a-videos-size-with-ffmpeg)
demo-visdrone:
	python src/track.py with \
		dataset_name=DEMO \
		data_root_dir=/data/stot/datasets_trackformer/fmv/frames/desert_car_2_pexel_m5 \
		output_dir=/home/nathan.candre/other/trackformer_results/videos/temp \
		write_images=pretty
		obj_detect_checkpoint_file=/home/nathan.candre/platypus/libs/research/research/sota_testing/trackformer/trainings/visdrone-vehicle/checkpoint.pth
