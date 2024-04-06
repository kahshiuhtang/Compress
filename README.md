# LUMBERJACK

### MQTT-backed log distributor

## Summary

When taking Cloud Computing in school, we often have to debug the output of multiple servers. Especially when you are only allowed to use a terminal, it is a massive pain to try and debug errors.

Goals for the project are as follows:

1) Create a basic API that allows users to write logs 

2) Behind the scnes of the API, generate a log message, based on the options given to the log, 

3) Create a MQTT client that can publish to a MQTT server. The MQTT client must be aware of which server to publish its messages to. The MQTT server should not be expected to anything but publish messages to subscribers.

4) Have a MQTT client that will decode and call syslog to write to a log

5) Create a CUDA-based string compressor that will quickly compress and decompress large log files

6) Create a CUDA-based string search tool to look through logs.

7) Listen to a port to sniff out any packets that come in. This is more of a debugging tool that may be useful.


Here is an idea of the structure of the log message can be.

```
[ServerSourceID] [BatchID] [Timestamp] [Message]
```

A large inspiration for this project was Elasticsearch. I think its a very good idea and piece of software but to set it up was a massive pain. It also ate up all my cloud credits. Because of this, some goals for Lumberjack are to be easily scalable, easily configurable, and low resource.

### Papers

I want to read some papers to get more knowledge on the things that I covered above. Here are the ones I will focus on.

[Hadoop](https://15799.courses.cs.cmu.edu/fall2013/static/papers/vldb09-861.pdf)

[HDFS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5496972)

[Efficient Parallel Scan Algos for GPUs](https://mgarland.org/files/papers/nvr-2008-003.pdf)

[Boyer Moore](https://dl.acm.org/doi/pdf/10.1145/359842.359859)

[Weiner](https://cpsc.yale.edu/sites/default/files/files/technical-reports/TR17%20Linear%20Pattern%20Matching%20ALgorithms.pdf)

[DEFLATE](https://www.ietf.org/rfc/rfc1951.txt)

[Burrows Wheeler](http://www.eecs.harvard.edu/~michaelm/CS222/burrows-wheeler.pdf)

[Kafka](https://notes.stephenholiday.com/Kafka.pdf)