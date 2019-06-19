# So that docker can see the webcam
echo "Setting envivonment variables for the webcam" 
xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# Download the VOC dataset for INT8 Calibration 
DATA_DIR=VOCdevkit
if [ -d "$DATA_DIR" ]; then
	echo "$DATA_DIR has already been downloaded"
else
	echo "Downloading VOC dataset"
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	tar -xf VOCtest_06-Nov-2007.tar
fi

# Build the dockerfile for the engironment 
if [ ! -z $(docker images -q object_detection_webcam:latest) ]; then 
	echo "Dockerfile has already been built"
else
	echo "Building docker image" 
	docker build -f dockerfiles/Dockerfile --tag=object_detection_webcam .
fi

# Start the docker container
echo "Starting docker container" 
docker run --runtime=nvidia -it -v `pwd`:/mnt --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH object_detection_webcam 
