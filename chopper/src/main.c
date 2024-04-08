#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "lumber.h"
#include "mill.h"
int main(int argc,  char **argv){
    int help_flag = 0;
    int publisher_flag = 0;
    int subscriber_flag = 0;
    char * config_path = NULL;
    char * log_file_path = NULL;
    int index;
    int c;

    opterr = 0;


    while ((c = getopt (argc, argv, "hpsc:f:")) != -1)
        switch (c)
        {
        case 'h':
            help_flag = 1;
            break;
        case 'p':
            publisher_flag = 1;
            break;
        case 's':
            subscriber_flag = 1;
            break;
        case 'c':
            config_path = optarg;
            break;
        case 'l':
            log_file_path = optarg;
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
    for (index = optind; index < argc; index++)
        printf ("Non-option argument %s\n", argv[index]);
    return 0;
    //sniff_packets();
}