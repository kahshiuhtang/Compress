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
void client_exit(int status, int sockfd, pthread_t *client_daemon);

/**
 * @brief Create a publisher client node
 *
 * Similar functionality to the actual syslog daemon
 */
int create_publisher(const char *_serv_address, const char * _server_port, const char * _topic);

/**
 * @brief Create a subscriber client node
 *
 * Similar functionality to the actual syslog daemon
 */

int create_subscriber(const char *_serv_address, const char * _server_port, const char * _topic);


/**
 * @brief Listen for traffic on certain ports
 *
 * 
 */

int sniff_packets(int **ports);

#define USAGE(program_name, retcode) do{ \
fprintf(stderr, "USAGE: %s %s\n", program_name, \
"[-h] [-y|-z] [-c CONFIG] [-l LOGPATH] [-a ADDRESS] [-p PORT] [-t TOPIC]\n" \
"    -h       Help: displays this help menu.\n" \
"    -y       Publisher: create a publisher node\n" \
"    -z       Subscriber: create a subscriber\n" \
"    -c       Config: Specify the location of a config file\n" \
"    -a       Address: Specify server address\n" \
"    -p       Port: Specify server port\n" \
"    -t       Topic: Specify topic to subscribe/publish to"); \
exit(retcode); \
} while(0)

#endif