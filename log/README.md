[MQTT Library](https://github.com/LiamBindle/MQTT-C)

## Installing MQTT Server

```bash

$ sudo apt install mosquitto

```

Log Structure:

In order to be able to search through logs faster, it is important to be able to have some sort of structure in the log messages

Here is the current state:

[SERVER][PORT][PROCESS][SERVERITY][TYPE][MESSAGE]

There may be an easier way to send messages in groups, but I have not figured that out. This should allow for an easier way of sorting through files based on Server/Port/Process. It seems useful to keep a strict format of messages and fill in any blanks with spaces, but that mat increase message size too much.

One TYPE of message that is included is messages about incoming requests on certain ports. I find when debugging microservices, I often run into the issue of needing to print out the messages I receive, such as the request parameters and body.