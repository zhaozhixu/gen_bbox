/* Author: Zhao Zhixu */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static const int C_FULL = 48;
static const int W_FULL = 24;
static const int H_FULL = 12;
static const int C_VALID = 45;
static const int W_VALID = 20;
static const int H_VALID = 12;
static const int W_PER_GROUP = 3;
static const int W_GROUPS = 8;
static const int C_PER_GROUP = 8;
static const int C_GROUPS = 6;
static const int ANCHORS_PER_GRID = 9;
static const int BBOX_SIZE = 4;
static const int IMG_H = 360;
static const int IMG_W = 640;
static const float E = 2.718281828;

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static void *alloc(size_t size)
{
    void *p;

    p = malloc(size);
    if (p == NULL)
        fprintf(stderr, "malloc(%lu) failed", size);

    return p;
}

static int get_index(const int *ids, int ndim, const int *dims)
{
    int i, id;
    for (i = 0, id = ids[0]; i < ndim-1; i++)
        id = dims[i+1] * id + ids[i+1];
    return id;
}

static void get_coords(int id, int *ids, int ndim, const int *dims)
{
    for (int i = ndim-1; i >= 0; i--) {
        ids[i] = id % dims[i];
        id /= dims[i];
    }
}

static int hwc_to_out_index(const int *ids)
{
    int w = ids[1];
    int c = ids[2];
    int w_ids[2];
    int c_ids[2];
    int *out_ids;
    int index;

    get_coords(w, w_ids, 2, (int[]){W_GROUPS, W_PER_GROUP});
    get_coords(c, c_ids, 2, (int[]){C_GROUPS, C_PER_GROUP});

    out_ids = (int[]){ids[0], w_ids[1], c_ids[0], w_ids[0], c_ids[1]};
    index = get_index(out_ids, 5, (int[]){H_FULL, W_PER_GROUP, C_GROUPS,
                W_GROUPS, C_PER_GROUP});

    return index;
}

static void out_index_to_hwc(int index, int *hwc)
{
    int out_ids[5];

    get_coords(index, out_ids, 5, (int[]){H_FULL, W_PER_GROUP, C_GROUPS,
                W_GROUPS, C_PER_GROUP});
    hwc[0] = out_ids[0];
    hwc[1] = out_ids[3] * W_PER_GROUP + out_ids[1];
    hwc[2] = out_ids[2] * C_PER_GROUP + out_ids[4];
}

struct pre_alloc_data {
    int *conf_indexes;
    int *bbox_indexes;
    int anchors_num;
};

struct pre_alloc_data *gbd_preprocess(void)
{
    struct pre_alloc_data *data;
    int anchors_num = W_VALID * H_VALID * ANCHORS_PER_GRID;

    data = (struct pre_alloc_data *)alloc(sizeof(struct pre_alloc_data));
    data->conf_indexes = (int *)alloc(sizeof(int) * anchors_num);
    data->bbox_indexes = (int *)alloc(sizeof(int) * anchors_num * BBOX_SIZE);
    data->anchors_num = anchors_num;

    int WA = W_VALID * ANCHORS_PER_GRID;
    int WAB = W_VALID * ANCHORS_PER_GRID * BBOX_SIZE;
    int AB = ANCHORS_PER_GRID * BBOX_SIZE;
    for (int i = 0; i < H_VALID; i++) {
        for (int j = 0; j < W_VALID; j++) {
            for (int k = 0; k < ANCHORS_PER_GRID; k++) {
                data->conf_indexes[i*WA+j*ANCHORS_PER_GRID+k]
                    = hwc_to_out_index((int[]){i, j, k});
                for (int l = 0; l < BBOX_SIZE; l++) {
                    data->bbox_indexes[i*WAB+j*AB+k*BBOX_SIZE+l]
                        = hwc_to_out_index((int[]){i, j,
                                    k*BBOX_SIZE+l+ANCHORS_PER_GRID});
                }
            }
        }
    }
    return data;
}

void gbd_postprocess(struct pre_alloc_data *data)
{
    free(data->conf_indexes);
    free(data->bbox_indexes);
    free(data);
}

static int find_max_index(struct pre_alloc_data *data, int8_t *feature)
{
    int max_id = -1;
    int8_t max = INT8_MIN;
    int8_t conf;

    for (int i = 0; i < data->anchors_num; i++) {
        conf = feature[data->conf_indexes[i]];
        if (max <= conf) {
            max = conf;
            max_id = i;
        }
    }
    return max_id;
}

static float safe_exp(float w)
{
    if (w < 1)
        return expf(w);
    return w * E;
}

static void transform_bbox(int8_t *bbox_delta, float *anchor, float *result)
{
    float delta_x = (float)bbox_delta[0] / 4;
    float delta_y = (float)bbox_delta[1] / 4;
    float delta_w = (float)bbox_delta[2] / 4;
    float delta_h = (float)bbox_delta[3] / 4;
    float anchor_x = anchor[0];
    float anchor_y = anchor[1];
    float anchor_w = anchor[2];
    float anchor_h = anchor[3];

    float cx = anchor_x + delta_x * anchor_w;
    float cy = anchor_y + delta_y * anchor_h;
    float w = anchor_w * safe_exp(delta_w);
    float h = anchor_h * safe_exp(delta_h);
    float xmin = MIN(MAX(cx-w*0.5, 0), IMG_W-1);
    float ymin = MIN(MAX(cy-h*0.5, 0), IMG_H-1);
    float xmax = MAX(MIN(cx+w*0.5, IMG_W-1), 0);
    float ymax = MAX(MIN(cy+h*0.5, IMG_H-1), 0);
    result[0] = xmin;
    result[1] = xmax;
    result[2] = ymin;
    result[3] = ymax;
}

void gbd_getbbox(struct pre_alloc_data *data, int8_t *feature, float *anchors,
                 float *bbox)
{
    int max_id;
    int hwc[3];
    int8_t bbox_delta[4];
    int anchor_id;
    float anchor[4];

    max_id = find_max_index(data, feature);
    bbox_delta[0] = feature[data->bbox_indexes[max_id * BBOX_SIZE]];
    bbox_delta[1] = feature[data->bbox_indexes[max_id * BBOX_SIZE + 1]];
    bbox_delta[2] = feature[data->bbox_indexes[max_id * BBOX_SIZE + 2]];
    bbox_delta[3] = feature[data->bbox_indexes[max_id * BBOX_SIZE + 3]];

    out_index_to_hwc(data->conf_indexes[max_id], hwc);
    anchor_id = get_index(hwc, 3, (int[]){H_VALID, W_VALID, ANCHORS_PER_GRID});
    anchor[0] = anchors[anchor_id * BBOX_SIZE];
    anchor[1] = anchors[anchor_id * BBOX_SIZE + 1];
    anchor[2] = anchors[anchor_id * BBOX_SIZE + 2];
    anchor[3] = anchors[anchor_id * BBOX_SIZE + 3];
    transform_bbox(bbox_delta, anchor, bbox);
}
