CC = nvcc
INCLUDES = -I.

# the build target executable:
TARGET = prog

all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CC) $(TARGET).o -o $@ 

$(TARGET).o: $(TARGET).cu
	$(CC) $(INCLUDES) $(TARGET).cu -c -o $@

clean:
	$(RM) $(TARGET) $(TARGET).o
	
