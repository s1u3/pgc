NVCC = nvcc
NVCCFLAGS = -O2 -arch=sm_120

TARGET = neural_network

all: $(TARGET)

$(TARGET): neural_network.cu
	$(NVCC) $(NVCCFLAGS) neural_network.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.exe *.obj *.lib *.exp
