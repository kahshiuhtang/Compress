#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <syslog.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/wait.h>

#include "lumber.h"
#include "mqtt.h"
#include "client.h"

lmb_daemon *running_daemon;
void _send_messages();

int lmb_print(int priority, const char *format, ...){
    va_list arg;
    int done;
    va_start (arg, format);
    if(running_daemon == NULL){
        fprintf(stderr, "lmb_error: no daemon process started");
        return -1;
    }
    if(running_daemon->write_fd == -1){
        fprintf(stderr, "lmb_error: no log destination set");
        return -1;
    }
    char *log_msg = malloc(MAX_LOG_MSG_LENGTH * sizeof(char));
    done = sprintf(log_msg, format, arg);
    if(done){
        fprintf(stderr, "lmb_error: unable to format log.");
        return -1;
    }
    if(running_daemon->write_fd <= 0){
        fprintf(stderr, "lmb_error: unavailable write fd.");
        return -1;
    }
    write(running_daemon->write_fd, log_msg, MAX_LOG_MSG_LENGTH * sizeof(char));
    free(log_msg);
    va_end (arg);
    return done;
}

int lmb_setup(char* setup_file){
    if(setup_file == NULL){
        running_daemon = malloc(sizeof(lmb_daemon));
        running_daemon->pid = 0;
        running_daemon->write_fd = 0;
        fprintf(stderr, "lmb: created daemon.\n");
    }
    return 0;
}

int lmb_start(){
    if(running_daemon != NULL){
        fprintf(stderr, "lmb_error: no process setup.");
        return -1;
    }else{
        int p[2];
        if(pipe(p) < 0) {
            return -1;
        }
        // pipefd[0] refers to the read end of the pipe.  pipefd[1] refers to the write end of the pipe. 
        int pid;
        if((pid = fork()) < 0){
            fprintf(stderr, "lmb_error: fork failed: %s\n", strerror(errno));
	        running_daemon->pid = 0;
	        close(p[0]); close(p[1]);   
            abort();
        }else if(pid == 0){
            setpgid(0, 0);
            close(p[1]);
            dup2(p[0], STDIN_FILENO); //stdin becomes p[1]
            _send_messages(NULL,NULL,NULL);
        }else{
            close(p[1]);
            running_daemon->pid = pid;
            running_daemon->write_fd = p[0];
        }
    }
    return 0;
}


int lmb_shutdown(){
    if(running_daemon != NULL){
        fprintf(stderr, "lmb_error: no process started.");
        return -1;
    }
    kill(running_daemon->pid, SIGKILL);
    int status;
    waitpid(running_daemon->pid, &status, 0);
    free(running_daemon);
    if (WIFEXITED(status)){
		fprintf(stderr, "lmb: log daemon EXIT code is %d\n", WEXITSTATUS(status));
	}
    status = close(running_daemon->write_fd);
    if(status){
        fprintf(stderr, "lmb_error: issue closing write file desciptor.");
        return -1;
    }
    return 0;
}

void _send_messages(char *server_address, char *server_port, char* server_topic){
    struct mqtt_client client;
    const char* addr = server_address;
    const char* port = server_port;
    const char* topic = server_topic;

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

    pthread_t client_daemon;
    if(pthread_create(&client_daemon, NULL, client_refresher, &client)) {
        fprintf(stderr, "Failed to start client daemon.\n");
        client_exit(EXIT_FAILURE, sockfd, NULL);
        return;
    }
    running_daemon->tid = client_daemon;

    while(fgetc(stdin) == '\n') {
        /* get the current time */
        time_t timer;
        time(&timer);
        struct tm* tm_info = localtime(&timer);
        char timebuf[26];
        strftime(timebuf, 26, "%Y-%m-%d %H:%M:%S", tm_info);

        char application_message[256];
        snprintf(application_message, sizeof(application_message), "The time is %s", timebuf);

        /* publish the time */
        mqtt_publish(&client, topic, application_message, strlen(application_message) + 1, MQTT_PUBLISH_QOS_0);

        /* check for errors */
        if (client.error != MQTT_OK) {
            fprintf(stderr, "error: %s\n", mqtt_error_str(client.error));
            pthread_cancel(client_daemon);
            client_exit(EXIT_FAILURE, sockfd, &client_daemon);
            return;
        }
    }
    pthread_cancel(client_daemon);
    pthread_join(client_daemon, NULL);
    printf("%s disconnecting from %s\n", addr, addr);
    sleep(1);
    client_exit(EXIT_SUCCESS, sockfd, &client_daemon);
    return;
}

void __listen_message(char *server_address, char *server_port, char* server_topic){
    const char* addr = server_address;
    const char* port = server_port;
    const char* topic = server_topic;

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
        return;
    }
    running_daemon->tid = client_daemon;

    /* subscribe */
    mqtt_subscribe(&client, topic, 0);

    int idx = 0;
    int c;
    char buffer[MAX_LOG_MSG_LENGTH];
    while((c = fgetc(stdin)) != EOF){
        if(idx == MAX_LOG_MSG_LENGTH){
            syslog(LOG_NOTICE, buffer);
            idx = 0;
        }else{
            buffer[idx] = c;
            idx++;
        }
    }

    pthread_cancel(client_daemon);
    pthread_join(client_daemon, NULL);
    printf("\n%s disconnecting from %s\n", addr, addr);
    sleep(1);

    client_exit(EXIT_SUCCESS, sockfd, &client_daemon);
}

