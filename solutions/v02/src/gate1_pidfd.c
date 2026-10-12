#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>

int scor_pidfd_open(int pid) {
    return (int)syscall(SYS_pidfd_open, pid, 0);
}
