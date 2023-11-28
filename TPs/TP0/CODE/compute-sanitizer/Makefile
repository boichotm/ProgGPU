CC = nvcc -lineinfo
INCLUDES = -I../common 

# the build target executable:
EXE = memcheck.exe initcheck.exe out_of_bound.exe 

all: $(EXE)

%.exe: %.o
	$(CC) $^ -o $@ 

%.o: %.cu
	$(CC) $(INCLUDES) $^ -c -o $@

clean:
	$(RM) $(EXE) *.o
	
