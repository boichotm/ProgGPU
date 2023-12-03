CC=gcc
CFLAGS=-O3 -Wall
LDFLAGS=-lm
EXE=tp2.exe

all : $(EXE)

$(EXE) : tp2.o
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) 

%.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $<

run : $(EXE)
	./$< 10240

proper : 
	rm -f *.o

clean :
	rm -f *.o $(EXE)
