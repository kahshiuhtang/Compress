#ifndef TEMPLATE_POSIX_SOCKS_H
#define TEMPLATE_POSIX_SOCKS_H
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include <fcntl.h>

int open_nb_socket(const char* addr, const char* port);

#endif