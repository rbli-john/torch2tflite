#!/bin/bash

python3 -m torch2tflite.converter  \
    --torch-path tests/mobilenetv2_model.pt  \
    --tflite-path output/mobilenetv2.tflite  \
    --sample-file /Users/rbli/codebases/GitHub/android-demo-app/HelloWorldApp/app/src/main/assets/image.jpg  \
    --target-shape 224 224 3