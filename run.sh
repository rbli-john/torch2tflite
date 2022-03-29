#!/bin/bash

# Converts traced MyModule
python3 -m torch2tflite.converter  \
    --torch-path tmp/my_module_script.pt  \
    --use-jit \
    --tflite-path output/my_module_script.tflite  \
    --sample-file /Users/rbli/codebases/GitHub/android-demo-app/HelloWorldApp/app/src/main/assets/image.jpg  \
    --target-shape 30 30 3

## Converts traced MyModule
#python3 -m torch2tflite.converter  \
#    --torch-path tests/my_module_traced.pt  \
#    --use_jit \
#    --tflite-path output/my_module_traced.tflite  \
#    --sample-file /Users/rbli/codebases/GitHub/android-demo-app/HelloWorldApp/app/src/main/assets/image.jpg  \
#    --target-shape 30 30 3

## Converts MyModule
#python3 -m torch2tflite.converter  \
#    --torch-path tests/my_module.pt  \
#    --tflite-path output/my_module.tflite  \
#    --sample-file /Users/rbli/codebases/GitHub/android-demo-app/HelloWorldApp/app/src/main/assets/image.jpg  \
#    --target-shape 30 30 3

#python3 -m torch2tflite.converter  \
#    --torch-path tests/mobilenetv2_model.pt  \
#    --tflite-path output/mobilenetv2.tflite  \
#    --sample-file /Users/rbli/codebases/GitHub/android-demo-app/HelloWorldApp/app/src/main/assets/image.jpg  \
#    --target-shape 224 224 3