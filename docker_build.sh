#!/usr/bin/env bash

docker build --build-arg UID=$UID -t rew_gen:v1.0 .
