#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

#include "../gen_bbox_gpu.h"

static const int C_FULL = 48;
static const int W_FULL = 24;
static const int H_FULL = 12;
static const int C_VALID = 45;
static const int W_VALID = 20;
static const int H_VALID = 12;
static const int ANCHORS_PER_GRID = 9;
static const int IMG_H = 360;
static const int IMG_W = 640;
static const float ANCHOR_SHAPE[] = { 229., 137., 48., 71., 289., 245.,
                                      185., 134., 85., 142., 31., 41.,
                                      197., 191., 237., 206., 63., 108. };

float *prepare_anchors(const float *anchor_shape, int width, int height,
                       int H, int W, int B)
{
     float center_x[W], center_y[H];
     float *anchors;
     int i, j, k;

     anchors = malloc(sizeof(float) * H * W * B * 4);
     for (i = 1; i <= W; i++)
          center_x[i-1] = i * width / (W + 1.0);
     for (i = 1; i <= H; i++)
          center_y[i-1] = i * height / (H + 1.0);

     int h_vol = W * B * 4;
     int w_vol = B * 4;
     int b_vol = 4;
     for (i = 0; i < H; i++) {
          for (j = 0; j < W; j++) {
               for (k = 0; k < B; k++) {
                    anchors[i*h_vol+j*w_vol+k*b_vol] = center_x[j];
                    anchors[i*h_vol+j*w_vol+k*b_vol+1] = center_y[i];
                    anchors[i*h_vol+j*w_vol+k*b_vol+2] = anchor_shape[k*2];
                    anchors[i*h_vol+j*w_vol+k*b_vol+3] = anchor_shape[k*2+1];
               }
          }
     }
     return anchors;
}

int main(int argc, char **argv)
{
     int num_ele = C_FULL * W_FULL * H_FULL;
     int feature_tmp;
     int8_t feature[C_FULL * W_FULL * H_FULL];
     float bbox[4];
     float *anchors;
     const char *file;
     FILE *fp;
     struct pre_alloc_data *data;
     clock_t start, end;

     if (argc != 2) {
          printf("%s FEATURE_FILE\n", argv[0]);
          exit(EXIT_SUCCESS);
     }

     file = argv[1];
     fp = fopen(file, "r");
     if (!fp) {
          printf("cannot open %s: %s\n", file, strerror(errno));
          exit(EXIT_FAILURE);
     }

     for (int i = 0; i < num_ele; i++) {
          fscanf(fp, "%x", &feature_tmp);
          feature[i] = (int8_t)feature_tmp;
     }
     fclose(fp);

     anchors = prepare_anchors(ANCHOR_SHAPE, IMG_W, IMG_H,
                               H_VALID, W_VALID, ANCHORS_PER_GRID);
     data = gbd_preprocess();
     start = clock();
     gbd_getbbox(data, feature, anchors, bbox);
     end = clock();
     gbd_postprocess(data);
     free(anchors);

     printf("[ ");
     for (int i = 0; i < 4; i++) {
          printf("%.2f ", bbox[i]);
     }
     printf("]\n");
     printf("time: %.3fms\n", (float)(end - start)/CLOCKS_PER_SEC*1000);
}
