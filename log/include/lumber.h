#ifndef LUMBER_H
#define LUMBER_H

#define PUBLISHER 0x1
#define SUBSCRIBER 0x2

#define MAX_LOG_MSG_LENGTH 256
/*
*/
typedef struct lmb_daemon{
    int write_fd;
    __pid_t pid;
    pthread_t tid;
    pthread_mutex_t mux;
} lmb_daemon;

/**
 * @brief Allow caller to log a message with a priority
 *
 * Similar functionality to the actual syslog daemon
 */
int lmb_print(int priority, const char *format, ...);

/**
 * @brief Setup the daemon to run 
 *
 * Either a json file is passed in as the filepath or two 2D arrays of options and values is passed once.
 */
int lmb_setup(char * setup_file);

/**
 * @brief Start the lumber daemon
 *
 */
int lmb_start();

/**
 * @brief Shutdown and deallocate the lumber daemon
 *
 */
int lmb_shutdown();

#endif