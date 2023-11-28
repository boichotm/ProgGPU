CC=gcc
CUDA_CC=nvcc
CFLAGS_OMP=-O3 -Wall -fopenmp -DUSE_OMP
CFLAGS=-O3 -Wall
EXE1=pi.exe
EXE2=pi_omp.exe
# EXE3=pi_task.exe
EXE3=pi_cuda.exe
EXE4=pi_cuda_shared.exe
EXE5=pi_cuda_shared_2.exe
EXE6=pi_cuda_shared_3.exe
EXE7=pi_cuda_shared_4.exe

all : $(EXE1) $(EXE2) $(EXE3) $(EXE4) $(EXE5) $(EXE6) $(EXE7) 

$(EXE1) : pi_sequentiel.o
	$(CC) $(CFLAGS) -o $@ $<

$(EXE2) : pi_omp.o
	$(CC) $(CFLAGS_OMP) -o $@ $<

$(EXE3) : pi_cuda.cu
	$(CUDA_CC) -O3 -o $@ $<

$(EXE4) : pi_cuda_shared.cu
	$(CUDA_CC) -O3 -o $@ $<

$(EXE5) : pi_cuda_shared_2.cu
	$(CUDA_CC) -O3 -o $@ $<

$(EXE6) : pi_cuda_shared_3.cu
	$(CUDA_CC) -O3 -o $@ $<

$(EXE7) : pi_cuda_shared_4.cu
	$(CUDA_CC) -O3 -o $@ $<

%_omp.o : %_omp.c
	$(CC) $(CFLAGS_OMP) -c -o $@ $<

%_task.o : %_task.c
	$(CC) $(CFLAGS_OMP) -c -o $@ $<

%.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean :
	rm -f *.o $(EXE1) $(EXE2) $(EXE3) $(EXE4) $(EXE5) $(EXE6) $(EXE7) 

proper :
	rm -f *.o
