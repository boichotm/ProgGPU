CC=clang -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
CFLAGS=-O3 -Wall
LDFLAGS=-lm -lrt

EXEC=dgemm.exe
SRC=$(wildcard *.c)
OBJ=$(SRC:.c=.o)

all : $(EXEC)

$(EXEC) : $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o : %.c
	$(CC) $(CFLAGS) -o $@ -c $<

proper :
	rm -f *.o

clean : proper
	rm -f $(EXEC)
