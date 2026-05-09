# Build the C kernel for substrate-LLM inner loop.
# Phase C of original architecture spec — replaces the numba JIT
# trainer with a single shared object (no venv/numba dependency).
#
# Usage:
#   cd brain/tasks/llm
#   make                  # build _inner_loop.so
#   make clean            # remove the .so

CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fPIC -shared
TARGET  := _inner_loop.so
SOURCES := _inner_loop.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

.PHONY: all clean
