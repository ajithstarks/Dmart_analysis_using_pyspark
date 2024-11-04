#!/bin/bash

export SPARK_HOME=/home/ajith/spark
export PATH=$SPARK_HOME/bin:$PATH

export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='notebook'

pyspark

