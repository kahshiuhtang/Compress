#ifndef LUMBER_H
#define LUMBER_H

#define PUBLISHER 0x1
#define SUBSCRIBER 0x2

int write_fd;
int read_fd;

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
int lmb_setup(char * filepath, char ** options, char ** values);

/**
 * @brief Start the lumber daemon
 *
 */
int lmb_start();

#endif