# LUMBERJACK

### Logging tool to consolidate logging into one central hub

## Summary

I want to make a log storing system that will take logs from multiple servers and place them onto one centralized server

I think logs should be split by where they came from and what "process" they come from. Logs will come in and be redirected to whatever DB by a message broker, if it is nessecary to store them long-term. Otherwise, logs will go into a log file, preferably by batch.

To give the best chance at parallel searching, the format of these messages should be in the lines of 

```
[ServerSourceID] [BatchID] [Timestamp] [Message]
```

Some goals for lumberjack are to be easily scalable, easily configurable, and low resource.

I have not completed doing research for this project, but I have some ideas in mind. I will use MQTT as the messaging protocol, store data to be backed up in MongoDB servers that are the master to a slave DB like Cassandra. I'm going to use my NVidia GPU for string search through logs in hopes of parallel performance boost, but I will mainly use C and/or Rust.

## Project Goals

1) Send logging messages into one central server

2) Create small compiler language for .conf file

3) Use MQTT to publish messages and send to subscribers

4) Implement fast searching algorithm through large file using GPU & CPU

5) Create logging system that is easily readable and 


### Papers

[Hadoop](https://15799.courses.cs.cmu.edu/fall2013/static/papers/vldb09-861.pdf)

[HDFS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5496972)

[Efficient Parallel Scan Algos for GPUs](https://mgarland.org/files/papers/nvr-2008-003.pdf)

[Boyer Moore](https://dl.acm.org/doi/pdf/10.1145/359842.359859)

[Weiner](https://cpsc.yale.edu/sites/default/files/files/technical-reports/TR17%20Linear%20Pattern%20Matching%20ALgorithms.pdf)

[DEFLATE](https://www.ietf.org/rfc/rfc1951.txt)

[Burrows Wheeler](http://www.eecs.harvard.edu/~michaelm/CS222/burrows-wheeler.pdf)

[Kafka](https://notes.stephenholiday.com/Kafka.pdf)