CFLAGS = -O3 -march=native  -D_GNU_SOURCE -pg
LDFLAGS = -lm -Mprof=lines 
CC = gcc
VECREPORT = -O3 -march=native  -D_GNU_SOURCE -ftree-vectorize
GPROF = -pg

LIKWID = -mfma -DLIKWID_PERFMON -I$LIKWID_PATH/include -L$LIKWID_PATH/lib -llikwid -lpthread -lm

ACC = pgcc -acc -Minfo -O3

OBJ = utils.o optimised-sparsemm.o basic-sparsemm.o
HEADER = utils.h

.PHONY: clean help check

all: sparsemm

help:
	@echo "Available targets are"
	@echo "  clean: Remove all build artifacts"
	@echo "  check: Perform a simple test of your optimised routines"
	@echo "  sparsemm: Build the sparse matrix-matrix multiplication binary"

clean:
	-rm -f sparsemm test $(OBJ)

check: sparsemm
	./sparsemm CHECK

sparsemm: sparsemm.c $(OBJ)
	$(ACC) -o $@ $< $(LDFLAGS) $(OBJ) 

# test: test.c $(OBJ)
# 	$(CC) $(CFLAGS) -o $@ $< $(OBJ) $(LDFLAGS)

test: test.c $(OBJ)
	$(ACC) -o $@ $< $(LDFLAGS) $(OBJ)

%.o: %.c $(HEADER)
	$(ACC) -c -o $@ $<

performance: sparsemm.c $(OBJ)
	$(CC) $(CFLAGS) -o sparsemm $< $(OBJ) $(LDFLAGS) $(LIKWID)

gprof: sparsemm.c $(OBJ)
	$(CC) $(CFLAGS) -o sparsemm $< $(OBJ) $(LDFLAGS) $(GPROF)

vecreport: sparsemm.c $(OBJ)
	$(CC) $(VECREPORT) -o sparsemm $< $(OBJ) $(LDFLAGS)

