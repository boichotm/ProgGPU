CC=clang -fopenmp
CFLAGS=-g -Wall
LDFLAGS=-lm -lrt

EXEC=spmv.exe
SRC=$(wildcard *.c)
OBJ=$(SRC:.c=.o)

all : $(EXEC)

$(EXEC) : main.o CSRMatrix.o EllMatrix.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

main_seq.o : main.c
	$(CC) -D SEQ $(CFLAGS) -o $@ -c $<

%.o : %.c
	$(CC) $(CFLAGS) -o $@ -c $<

proper :
	rm -f *.o

clean : proper
	rm -f $(EXEC)
