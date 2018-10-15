#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

#include "../gen_bbox.h"

#define CONVOUT_C  144
#define CONVOUT_H  23
#define CONVOUT_W  40

int main(int argc, char **argv)
{
     int num_ele = CONVOUT_C*CONVOUT_H*CONVOUT_W;
     float feature_float[CONVOUT_C*CONVOUT_H*CONVOUT_W] = {0};
     int16_t feature_i16[CONVOUT_C*CONVOUT_H*CONVOUT_W] = {0};
     float bbox[4];
     const char *file;
     char *filecp, cmd[100];
     int img_w, img_h, i, status;
     FILE *fp;
     struct pre_alloc_tensors *tensors;
     clock_t start, end;

     if (argc != 4) {
          printf("%s FEATURE_FILE IMAGE_WIDTH IMAGE_HEIGHT\n", argv[0]);
          exit(EXIT_SUCCESS);
     }

     file = argv[1];
     img_w = atoi(argv[2]);
     img_h = atoi(argv[3]);
     filecp = malloc(sizeof(char) * (strlen(file)+100));
     strcpy(filecp, file);
     sprintf(filecp+strlen(filecp), ".%d", getpid());

     sprintf(cmd, "cp %s %s", file, filecp);
     status = system(cmd);
     if (status < 0)
          exit(EXIT_FAILURE);
     memset(cmd, 0, sizeof(char)*100);

     sprintf(cmd, "sed -i 's,\\],,g' %s", filecp);
     status = system(cmd);
     if (status < 0)
          exit(EXIT_FAILURE);
     memset(cmd, 0, sizeof(char)*100);

     sprintf(cmd, "sed -i 's,\\[,,g' %s", filecp);
     status = system(cmd);
     if (status < 0)
          exit(EXIT_FAILURE);
     memset(cmd, 0, sizeof(char)*100);

     fp = fopen(filecp, "r");
     if (!fp) {
          printf("cannot open %s: %s\n", filecp, strerror(errno));
          exit(EXIT_FAILURE);
     }

     for (i = 0; i < num_ele; i++) {
          fscanf(fp, "%f", &feature_float[i]);
          feature_i16[i] = (int16_t)feature_float[i];
     }
     fclose(fp);

     sprintf(cmd, "rm -f %s", filecp);
     status = system(cmd);
     if (status < 0)
          exit(EXIT_FAILURE);
     memset(cmd, 0, sizeof(char)*100);

     free(filecp);

     tensors = gb_preprocess();
     start = clock();
     gb_getbbox(feature_i16, img_w, img_h, tensors, bbox);
     end = clock();
     gb_postprocess(tensors);

     printf("predict:\n[ ");
     for (i = 0; i < 4; i++) {
          printf("%.2f ", bbox[i]);
     }
     printf("]\n");
     printf("time: %.3fms\n", (float)(end - start)/CLOCKS_PER_SEC*1000);
}
