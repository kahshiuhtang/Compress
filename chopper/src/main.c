#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "lumber.h"
#include "client.h"

int main(int argc,  char **argv){
    int help_flag = 0;
    int publisher_flag = 0;
    int subscriber_flag = 0;
    char * config_path = NULL;
    char * log_file_path = NULL;
    char * server = NULL;
    char * port = NULL;
    char * topic = NULL;
    int index;
    int c;

    opterr = 0;

    while ((c = getopt (argc, argv, "hyzc:f:a:p:t:")) != -1)
        switch (c)
        {
        case 'h':
            help_flag = 1;
            break;
        case 'y':
            publisher_flag = 1;
            break;
        case 'z':
            subscriber_flag = 1;
            break;
        case 'c':
            config_path = optarg;
            break;
        case 'l':
            log_file_path = optarg;
            break;
        case 'a':
            server = optarg;
            break;
        case 'p':
            port = optarg;
            break;
        case 't':
            topic = optarg;
            break;
        case '?':
            USAGE(*argv, EXIT_FAILURE);
            return 1;
        default:
            abort ();
        }
    if(help_flag){
        USAGE(*argv, EXIT_SUCCESS);
        return 1;
    }
    if(publisher_flag && subscriber_flag){
       printf("Node cannot be a publisher and subscriber at once.");
       USAGE(*argv, EXIT_FAILURE); 
    }
    if(!publisher_flag && !subscriber_flag){
        printf("Node has to either be a publisher or subscriber"); 
        USAGE(*argv, EXIT_FAILURE); 
    }
    if(log_file_path){
        FILE *file = fopen(log_file_path, "r");
        if (file == NULL) {
            printf("Failed to open log file.");
            return EXIT_FAILURE;
        }
        fclose(file);
    }
    if(config_path){
        FILE *file = fopen(config_path, "r");
        if (file == NULL) {
            printf("Failed to open config file.");
            return EXIT_FAILURE;
        }
        fclose(file);

    }
    if(publisher_flag){
        create_publisher(server, port, topic);
    }
    if(subscriber_flag){
        create_subscriber(server, port, topic);
    }
    return 0;
    //sniff_packets();
}