CXX = g++
CC = g++

CXXFLAGS = -std=c++11
CXXFLAGS += -Wall
CXXFLAGS += `pkg-config --cflags --libs opencv` 
CXXFLAGS += -I /home/oscar/Dokument/caffe/caffe/include

PROGS = capture_and_save_video featuredetector 

#capture_video: capture_video.o

PROGNAME = coddetector

coddetector: 
	$(CC) $(CXXFLAGS) classification.cpp -o $(PROGNAME)

#all: $(PROGS)

#$(PROGS): $(PROGS).cpp
#	$(CC) $(CXXFLAGS) $(PROGS).cpp -o $(PROGNAME) 
#clean:
#	rm -f *.o $(PROGS)

.PHONY: coddetector clean
