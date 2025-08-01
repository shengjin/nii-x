# =============================================================================
# MPI Langevin + Gaussian 混合采样项目 Makefile
# =============================================================================

# 编译器
CC    := mpicc
CXX   := mpic++

# 编译选项
CFLAGS   := -Wall -Wextra -O2 -I/usr/include/adolc
CXXFLAGS := -Wall -Wextra -O2 -I/usr/include/adolc

# 链接选项
LDFLAGS  := -lm -ladolc

# C 源文件列表
C_SRCS := \
    main.c        \
    mpi_init.c    \
    mpi_batch.c   \
    mpi_stack.c   \
    mpi_flow.c    \
    data_loader.c \
    alloc.c       \
    rand_func.c   \
    readin.c      \
    user_prior.c

# C++ 源文件列表
CPP_SRCS := \
    user_logll_grad.cpp

# 目标文件
C_OBJS   := $(C_SRCS:.c=.o)
CPP_OBJS := $(CPP_SRCS:.cpp=.o)
OBJS     := $(C_OBJS) $(CPP_OBJS)

# 最终可执行文件
TARGET := a.out

# 默认目标
all: $(TARGET)

# 链接
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

# C 文件编译规则
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# C++ 文件编译规则
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理
clean:
	rm -f $(OBJS) $(TARGET) test.dat

.PHONY: all clean

