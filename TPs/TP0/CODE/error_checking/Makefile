CC = nvcc
INCLUDES = -I../common

# the build target executable:
EXE = exemple1.exe exemple2.exe exemple3.exe

all: $(EXE)

%.exe: %.o
	$(CC) $^ -o $@ 

%.o: %.cu
	$(CC) $(INCLUDES) $^ -c -o $@

clean:
	$(RM) $(EXE) *.o
	
