CFLAGS=-Wall -Ofast -march=native -fopenmp
LDFLAGS=-lm -fopenmp

sort: sort.o
	gcc -o sort sort.o -lm $(LDFLAGS)

sort.o: sort.c
	gcc -g -c $(CFLAGS) sort.c

clean:
	rm -f ./sort *.o
