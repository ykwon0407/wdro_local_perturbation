#!/bin/bash
cd ~/deep/project/dro-id/
mkdir results
cd experiments/
find -name '*.txt' -exec cp --parents {} -at ../results/ \;
