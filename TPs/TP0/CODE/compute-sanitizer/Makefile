CC = nvcc -lineinfo
INCLUDES = 

# the build target executable:
EXE = memcheck.exe initcheck.exe 

all: $(EXE)

%.exe: %.o
	$(CC) $^ -o $@ 

%.o: %.cu
	$(CC) $(INCLUDES) $^ -c -o $@

clean:
	$(RM) $(EXE) *.o
	
