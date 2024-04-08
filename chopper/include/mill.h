#ifndef MILL_H
#define MILL_H

#include "templates/posix_sockets.h"
#include <mqtt.h>
#include <pthread.h>
/**
 * @brief The function that would be called whenever a PUBLISH is received.
 *
 * @note This function is not used in this example.
 */
void publish_callback(void** unused, struct mqtt_response_publish *published);

/**
 * @brief The client's refresher. This function triggers back-end routines to
 *        handle ingress/egress traffic to the broker.
 *
 * @note All this function needs to do is call \ref __mqtt_recv and
 *       \ref __mqtt_send every so often. I've picked 100 ms meaning that
 *       client ingress/egress traffic will be handled every 100 ms.
 */
void* client_refresher(void* client);

/**
 * @brief Safelty closes the \p sockfd and cancels the \p client_daemon before \c exit.
 */
void exit_example(int status, int sockfd, pthread_t *client_daemon);

/**
 * @brief Create a publisher client node
 *
 * Similar functionality to the actual syslog daemon
 */
int create_publisher(int argc, const char *argv[]);

/**
 * @brief Create a subscriber client node
 *
 * Similar functionality to the actual syslog daemon
 */

int create_subscriber(int argc, const char *argv[]);


/**
 * @brief Listen for traffic on certain ports
 *
 * 
 */

int sniff_packets(int **ports);

#define USAGE(program_name, retcode) do{ \
fprintf(stderr, "USAGE: %s %s\n", program_name, \
"[-h] [-p|-s] [-c CONFIG] [-l LOGPATH]\n" \
"    -h       Help: displays this help menu.\n" \
"    -p       Publisher: create a publisher node\n" \
"    -s       Subscriber: create a subscriber\n" \
"    -c       Config: Specify the location of a config file\n" \
"    -l       Log: Specify the location of a log file to read or write from"); \
exit(retcode); \
} while(0)

#endif