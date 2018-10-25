# Generate Bounding Box

C functions to generate the bounding box from a feature map.

## Prerequisites
This requires library *TensorLight* to execute tensor computation.
Check [TensorLight](https://github.com/zhaozhixu/TensorLight) for installation guide.

## Build for test
Use `make` in this folder to compile test program then do the test.

## Usage
Copy `gen_bbox.h` and `gen_bbox.c` in your project to use the functions.

## Example
```
#include "gen_bbox.h"

float *feature_map;
int img_width, img_height; /* original image size */
float bbox[4]; /* bounding box */
struct pre_alloc_tensors *tensors;

tensors = (struct pre_alloc_tensors *)gb_preprocess(); /* create necessary tensors */

feature_map = ...... /* compute feature_map */
gb_getbbox(feature_map, img_width, img_height, tensors, bbox); /* get bbox */

gb_postprocess(tensors); /* release tensors */

```
