#include "lumber.h"
#include <stdarg.h>
#include <stdio.h>
int lmb_print(int priority, const char *format, ...){
    va_list arg;
    int done;

    va_start (arg, format);
    done = vfprintf (stdout, format, arg);
    va_end (arg);

    return done;
}