# Generate Bounding Box

C functions to generate the bounding box from a feature map.

## Prerequisites
This requires library *Tensorlight* to execute tensor computation.
Check [Tensorlight](https://github.com/zhaozhixu/TensorLight) for installation guide.

## Build for test
Use `make` in this folder to compile the shared library `libgen_bbox.so`,
then do the test.

## Usage
Copy `gen_bbox.h` and `gen_bbox.c` in your project to use the functions.

## Example
```
#include "gen_bbox.h"

int16_t *feature_map;
int img_width, img_height; /* original image size */
float bbox[4]; /* bounding box */
struct pre_alloc_tensors *tensors;

tensors = preprocess(); /* create necessary tensors */

feature_map = ...... /* compute feature_map */
feature_express(feature_map, img_width, img_height, tensors, bbox); /* get bbox */

postprocess(tensors); /* release tensors */

```
