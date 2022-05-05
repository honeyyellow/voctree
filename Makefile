#change acording to your OpenCV installation
#OPENCV_DIR = ~/opencv/installed/3.1.0
OPENCV_DIR = /usr/local

CC=nvcc # g++ 
CFLAGS= -arch=sm_75 # -c -Wall -O3
LIBS=-L $(OPENCV_DIR)/lib \
	-lopencv_core \
	-lopencv_highgui \
	-lopencv_imgproc \
	-lopencv_imgcodecs \
	-lopencv_features2d \
	-lopencv_xfeatures2d \
	-lopencv_video \
	-lopencv_videoio \
	-lopencv_flann \
	-lopencv_calib3d

SDIR = src
ODIR = .
EXEC = vt

SOURCES=FileHelper.cu \
	Configuration.cu \
	Catalog.cu \
	FileManager.cu \
	Database.cu \
	KeyPointPersistor.cu \
	ExtKmeans.cu \
	FeatureMethod.cu \
	MatPersistor.cu \
	Matching.cu \
	ShootSegmenter.cu \
	VocTree.cu \
	KMeans.cu \
	Server.cu \
	main.cu


OBJECTS=$(SOURCES:.cu=.o)


INC=-I $(SDIR)/ \
	-I $(OPENCV_DIR)/include/ \
	-I $(OPENCV_DIR)/include/opencv \
	-I $(OPENCV_DIR)/include/opencv2/


OBJS = $(addprefix $(ODIR)/, $(OBJECTS))


all: $(OBJS) $(EXEC) clean_objs

$(ODIR)/%.o: $(SDIR)/%.cu 
	$(CC) $(INC) $(CFLAGS) -o $@ $<

$(EXEC): $(OBJS)
	$(CC) -O3 -o $(ODIR)/$@ $^ $(LIBS)

clean: clean_objs
	rm $(ODIR)/$(EXEC)

clean_objs:
	rm $(OBJS)
