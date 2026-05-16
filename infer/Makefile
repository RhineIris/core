# LCM Inference Engine — Makefile
#
# Build targets:
#   make              — Release build (liblcm.a, liblcm.so)
#   make debug        — Debug build
#   make contracts    — Contract-checked build
#   make clean        — Remove build artifacts
#   make test         — Build and run unit tests
#
# Configurable:
#   make LCM_D=256    — Vector dimension (default: 128, matches step_19000)
#
# Required: C99 compiler (gcc or clang), libm

CC       ?= gcc
CFLAGS   ?= -std=c99 -Wall -Wextra -Wpedantic -Werror
OPTFLAGS ?= -O2 -march=native -ffast-math
LDFLAGS  ?= -lm

# Vector dimension — override with LCM_D=256 for old checkpoints
LCM_D    ?= 128
CFLAGS   += -DLCM_D=$(LCM_D)

SRCS     = hyp.c lattice.c gvalue.c danger.c engine.c lcm_api.c
OBJS     = $(SRCS:.c=.o)

# ── Release build (default) ──────────────────────────────────────────────────
all: CFLAGS += $(OPTFLAGS) -DNDEBUG -fPIC
all: liblcm.a liblcm.so

# ── Debug build ──────────────────────────────────────────────────────────────
debug: CFLAGS += -g -O0 -DDEBUG -DLCM_USE_CONTRACTS -fPIC
debug: liblcm.a liblcm.so

# ── Contract-checked build ───────────────────────────────────────────────────
contracts: CFLAGS += $(OPTFLAGS) -DLCM_USE_CONTRACTS -fPIC
contracts: liblcm.a liblcm.so

liblcm.a: $(OBJS)
	$(AR) rcs $@ $(OBJS)

liblcm.so: $(OBJS)
	$(CC) -shared $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c lcm.h lcm_api.h
	$(CC) $(CFLAGS) -c -o $@ $<

# ── Test target ──────────────────────────────────────────────────────────────
test: CFLAGS += -g -O0 -DDEBUG -DLCM_USE_CONTRACTS
test: test_runner.c liblcm.a
	$(CC) $(CFLAGS) -o test_runner test_runner.c -L. -llcm $(LDFLAGS)
	./test_runner

# ── Clean ───────────────────────────────────────────────────────────────────
clean:
	rm -f *.o *.a *.so test_runner

.PHONY: all debug contracts clean test
