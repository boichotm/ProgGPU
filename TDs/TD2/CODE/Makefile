CC=gcc
CFLAGS_OMP=-O3 -Wall -fopenmp -DUSE_OMP
CFLAGS=-O3 -Wall
EXE1=pi.exe
EXE2=pi_omp.exe

all : $(EXE1) $(EXE2)

$(EXE1) : pi.o
	$(CC) $(CFLAGS) -o $@ $<

$(EXE2) : pi_omp.o
	$(CC) $(CFLAGS_OMP) -o $@ $<

%_omp.o : %_omp.c
	$(CC) $(CFLAGS_OMP) -c -o $@ $<

%.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean :
	rm -f *.o $(EXE1) $(EXE2)

proper :
	rm -f *.o
