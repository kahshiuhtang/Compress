/**
 * @file
 * A simple program to that publishes the current time whenever ENTER is pressed.
 */
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#include <mqtt.h>
#include "client.h"



/**
 * A simple program to that publishes the current time whenever ENTER is pressed.
 */
int create_publisher(const char *_serv_address, const char * _server_port, const char * _topic)
{
    const char* addr = _serv_address;
    const char* port = _server_port;
    const char* topic = _topic;

    if (addr == NULL)  {
        addr = "test.mosquitto.org";
    }
    if (port == NULL) {
        port = "1883";
    }

    if (topic == NULL) {
        topic = "datetime";
    }

    /* open the non-blocking TCP socket (connecting to the broker) */
    int sockfd = open_nb_socket(addr, port);

    if (sockfd == -1) {
        perror("Failed to open socket: ");
        client_exit(EXIT_FAILURE, sockfd, NULL);
    }

    /* setup a client */
    struct mqtt_client client;
    uint8_t sendbuf[2048]; /* sendbuf should be large enough to hold multiple whole mqtt messages */
    uint8_t recvbuf[1024]; /* recvbuf should be large enough any whole mqtt message expected to be received */
    mqtt_init(&client, sockfd, sendbuf, sizeof(sendbuf), recvbuf, sizeof(recvbuf), publish_callback);
    /* Create an anonymous session */
    const char* client_id = NULL;
    /* Ensure we have a clean session */
    uint8_t connect_flags = MQTT_CONNECT_CLEAN_SESSION;
    /* Send connection request to the broker. */
    mqtt_connect(&client, client_id, NULL, NULL, 0, NULL, NULL, connect_flags, 400);

    /* check that we don't have any errors */
    if (client.error != MQTT_OK) {
        fprintf(stderr, "error: %s\n", mqtt_error_str(client.error));
        client_exit(EXIT_FAILURE, sockfd, NULL);
    }

    /* start a thread to refresh the client (handle egress and ingree client traffic) */
    pthread_t client_daemon;
    if(pthread_create(&client_daemon, NULL, client_refresher, &client)) {
        fprintf(stderr, "Failed to start client daemon.\n");
        client_exit(EXIT_FAILURE, sockfd, NULL);

    }

    /* start publishing the time */
    printf("%s is ready to begin publishing the time.\n", addr);
    printf("Press ENTER to publish the current time.\n");
    printf("Press CTRL-D (or any other key) to exit.\n");
    while(fgetc(stdin) == '\n') {
        /* get the current time */
        time_t timer;
        time(&timer);
        struct tm* tm_info = localtime(&timer);
        char timebuf[26];
        strftime(timebuf, 26, "%Y-%m-%d %H:%M:%S", tm_info);

        /* print a message */
        char application_message[256];
        snprintf(application_message, sizeof(application_message), "The time is %s", timebuf);
        printf("%s published : \"%s\"", addr, application_message);

        /* publish the time */
        mqtt_publish(&client, topic, application_message, strlen(application_message) + 1, MQTT_PUBLISH_QOS_0);

        /* check for errors */
        if (client.error != MQTT_OK) {
            fprintf(stderr, "error: %s\n", mqtt_error_str(client.error));
            client_exit(EXIT_FAILURE, sockfd, &client_daemon);
        }
    }
    pthread_cancel(client_daemon);
    printf("%s disconnecting from %s\n", addr, addr);
    sleep(1);
    client_exit(EXIT_SUCCESS, sockfd, &client_daemon);
    return 0;
}

int create_subscriber(const char *_serv_address, const char * _server_port, const char * _topic)
{
    const char* addr = _serv_address;
    const char* port = _server_port;
    const char* topic = _topic;

    if (addr == NULL)  {
        addr = "test.mosquitto.org";
    }
    if (port == NULL) {
        port = "1883";
    }

    if (topic == NULL) {
        topic = "datetime";
    }

    /* open the non-blocking TCP socket (connecting to the broker) */
    int sockfd = open_nb_socket(addr, port);

    if (sockfd == -1) {
        perror("Failed to open socket: ");
        client_exit(EXIT_FAILURE, sockfd, NULL);
    }

    struct mqtt_client client;
    uint8_t sendbuf[2048]; 
    uint8_t recvbuf[1024];
    mqtt_init(&client, sockfd, sendbuf, sizeof(sendbuf), recvbuf, sizeof(recvbuf), publish_callback);

    const char* client_id = NULL;
    uint8_t connect_flags = MQTT_CONNECT_CLEAN_SESSION;
    mqtt_connect(&client, client_id, NULL, NULL, 0, NULL, NULL, connect_flags, 400);

    if (client.error != MQTT_OK) {
        fprintf(stderr, "error: %s\n", mqtt_error_str(client.error));
        client_exit(EXIT_FAILURE, sockfd, NULL);
    }

    pthread_t client_daemon;
    if(pthread_create(&client_daemon, NULL, client_refresher, &client)) {
        fprintf(stderr, "Failed to start client daemon.\n");
        client_exit(EXIT_FAILURE, sockfd, NULL);
        return -1;
    }

    mqtt_subscribe(&client, topic, 0);

    printf("%s listening for '%s' messages.\n", addr, topic);
    printf("Press CTRL-D to exit.\n\n");

    while(fgetc(stdin) != EOF){

    }

    /* disconnect */
    printf("%s disconnecting from %s\n", addr, addr);
    sleep(1);

    /* exit */
    client_exit(EXIT_SUCCESS, sockfd, &client_daemon);
    return 0;
}
