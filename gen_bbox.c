#include <tl_tensor.h>
#include <math.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static const int INPUT_C = 3;
static const int INPUT_H = 368;
static const int INPUT_W = 640;

static const int CONVOUT_C = 144;
static const int CONVOUT_H = 23;
static const int CONVOUT_W = 40;

static const int CLASS_SLICE_C = 99;
static const int CONF_SLICE_C = 9;
static const int BBOX_SLICE_C = 36;

static const int TOP_N_DETECTION = 64;
static const float NMS_THRESH = 0.4;
static const float PROB_THRESH = 0.3;
static const float PLOT_PROB_THRESH = 0.4;

static const int ANCHORS_PER_GRID = 9;
static float ANCHOR_SHAPES[] = {229, 137, 48, 71, 289, 245,
                                185, 134, 85, 142, 31, 41,
                                197, 191, 237, 206, 63, 108};

static const float E = 2.718281828;

struct pre_alloc_tensors {
     tl_tensor *anchors;
     tl_tensor *feature;
     tl_tensor *conf_feature, *bbox_feature;
     tl_tensor *conf_max, *conf_maxidx;
     tl_tensor *bbox_int16, *bbox_float, *anchor;
     int volumn_slice_feature;
};

struct pre_alloc_tensors *gb_preprocess(void)
{
     struct pre_alloc_tensors *tensors;

     tensors = (struct pre_alloc_tensors *)tl_alloc(sizeof(struct pre_alloc_tensors));
     int dims_feature[] = {CONVOUT_C,CONVOUT_H,CONVOUT_W};
     tensors->feature = tl_tensor_create(NULL, 3, dims_feature, TL_INT16, 0);
     int dims_conf_feature[] = {CONF_SLICE_C,CONVOUT_H,CONVOUT_W};
     tensors->conf_feature = tl_tensor_create(NULL, 3, dims_conf_feature, TL_INT16, 0);
     int dims_bbox_feature[] = {BBOX_SLICE_C,CONVOUT_H,CONVOUT_W};
     tensors->bbox_feature = tl_tensor_create(NULL, 3, dims_bbox_feature, TL_INT16, 0);
     int dims_zeros_conf_max[] = {1};
     tensors->conf_max = tl_tensor_zeros(1, dims_zeros_conf_max, TL_INT16);
     int dims_zeros_conf_maxidx[] = {1};
     tensors->conf_maxidx = tl_tensor_zeros(1, dims_zeros_conf_maxidx, TL_INT32);
     int dims_zeros_bbox_int16[] = {1,4};
     tensors->bbox_int16 = tl_tensor_zeros(2, dims_zeros_bbox_int16, TL_INT16);
     int dims_zeros_bbox_float[] = {1,4};
     tensors->bbox_float = tl_tensor_zeros(2, dims_zeros_bbox_float, TL_FLOAT);
     int dims_zeros_anchor[] = {1,4};
     tensors->anchor = tl_tensor_zeros(2, dims_zeros_anchor, TL_FLOAT);

     tensors->volumn_slice_feature = 1;
     for (int i = 1; i < tensors->feature->ndim; i++)
          tensors->volumn_slice_feature *= tensors->feature->dims[i];

     int dims_create_anchor[] = {ANCHORS_PER_GRID,2};
     tl_tensor *anchor_shapes = tl_tensor_create(ANCHOR_SHAPES, 2, dims_create_anchor, TL_FLOAT, 0);
     tl_tensor *all_anchor_shapes = tl_tensor_repeat(anchor_shapes, CONVOUT_H*CONVOUT_W);
     int dims_reshape_anchor[] = {CONVOUT_H,CONVOUT_W,ANCHORS_PER_GRID,2};
     tl_tensor_reshape_src(all_anchor_shapes, 4, dims_reshape_anchor);

     tl_tensor *center_x_conv = tl_tensor_arange(1, CONVOUT_W+1, 1, TL_FLOAT);
     tl_tensor *center_x_input = tl_tensor_elew_param(center_x_conv, (float)INPUT_W/((float)CONVOUT_W+1), NULL, TL_MUL);
     tl_tensor *center_x_input_all = tl_tensor_repeat(center_x_input, CONVOUT_H*ANCHORS_PER_GRID);
     int dims_reshape_center_x[] = {ANCHORS_PER_GRID,CONVOUT_H,CONVOUT_W};
     tl_tensor_reshape_src(center_x_input_all, 3, dims_reshape_center_x);
     int dims_trans_center_x[] = {1,2,0};
     tl_tensor *center_x_input_all_trans = tl_tensor_transpose(center_x_input_all, NULL, dims_trans_center_x, NULL);
     int dims_reshape_center_x_trans[] = {CONVOUT_H,CONVOUT_W,ANCHORS_PER_GRID,1};
     tl_tensor_reshape_src(center_x_input_all_trans, 4, dims_reshape_center_x_trans);

     tl_tensor *center_y_conv = tl_tensor_arange(1, CONVOUT_H+1, 1, TL_FLOAT);
     tl_tensor *center_y_input = tl_tensor_elew_param(center_y_conv, (float)INPUT_H/((float)CONVOUT_H+1), NULL, TL_MUL);
     tl_tensor *center_y_input_all = tl_tensor_repeat(center_y_input, CONVOUT_W*ANCHORS_PER_GRID);
     int dims_reshape_center_y[] = {ANCHORS_PER_GRID,CONVOUT_W,CONVOUT_H};
     tl_tensor_reshape_src(center_y_input_all, 3, dims_reshape_center_y);
     int dims_trans_center_y[] = {2,1,0};
     tl_tensor *center_y_input_all_trans = tl_tensor_transpose(center_y_input_all, NULL, dims_trans_center_y, NULL);
     int dims_reshape_center_y_trans[] = {CONVOUT_H,CONVOUT_W,ANCHORS_PER_GRID,1};
     tl_tensor_reshape_src(center_y_input_all_trans, 4, dims_reshape_center_y_trans);

     tl_tensor *concat1 = tl_tensor_concat(center_x_input_all_trans, center_y_input_all_trans, NULL, 3);
     tl_tensor *concat2 = tl_tensor_concat(concat1, all_anchor_shapes, NULL, 3);

     tensors->anchors = concat2;
     tl_tensor_free(anchor_shapes);
     tl_tensor_free_data_too(all_anchor_shapes);
     tl_tensor_free_data_too(center_x_conv);
     tl_tensor_free_data_too(center_x_input);
     tl_tensor_free_data_too(center_x_input_all);
     tl_tensor_free_data_too(center_x_input_all_trans);
     tl_tensor_free_data_too(center_y_conv);
     tl_tensor_free_data_too(center_y_input);
     tl_tensor_free_data_too(center_y_input_all);
     tl_tensor_free_data_too(center_y_input_all_trans);
     tl_tensor_free_data_too(concat1);

     return tensors;
}

void gb_postprocess(struct pre_alloc_tensors *tensors)
{
     tl_tensor_free_data_too(tensors->anchors);
     tl_tensor_free(tensors->bbox_feature);
     tl_tensor_free(tensors->conf_feature);
     tl_tensor_free_data_too(tensors->conf_max);
     tl_tensor_free_data_too(tensors->conf_maxidx);
     tl_tensor_free_data_too(tensors->bbox_int16);
     tl_tensor_free_data_too(tensors->bbox_float);
     tl_tensor_free_data_too(tensors->anchor);
     tl_tensor_free(tensors->feature);
}

static float safe_exp(float w)
{
     if (w < 1)
          return expf(w);
     return w * E;
}

static void transform_bbox(float *bbox_delta, float *anchor, float *result,
                           int img_width, int img_height)
{
     float x_scale = 1.0 * img_width / INPUT_W;
     float y_scale = 1.0 * img_height / INPUT_H;
     float delta_x = bbox_delta[0];
     float delta_y = bbox_delta[1];
     float delta_w = bbox_delta[2];
     float delta_h = bbox_delta[3];
     float anchor_x = anchor[0];
     float anchor_y = anchor[1];
     float anchor_w = anchor[2];
     float anchor_h = anchor[3];
     float cx = (anchor_x + delta_x * anchor_w) * x_scale;
     float cy = (anchor_y + delta_y * anchor_h) * y_scale;
     float w = anchor_w * safe_exp(delta_w) * x_scale;
     float h = anchor_w * safe_exp(delta_h) * y_scale;
     float xmin = MIN(MAX(cx-w*0.5, 0), img_width-1);
     float ymin = MIN(MAX(cy-h*0.5, 0), img_height-1);
     float xmax = MAX(MIN(cx+w*0.5, img_width-1), 0);
     float ymax = MAX(MIN(cy+h*0.5, img_height-1), 0);
     result[0] = xmin;
     result[1] = ymin;
     result[2] = xmax;
     result[3] = ymax;
}

void gb_getbbox(int16_t *feature, int img_width, int img_height,
                struct pre_alloc_tensors *tensors, float *result)
{
     tensors->feature->data = feature;
     tensors->conf_feature->data = &((int16_t*)tensors->feature->data)[CLASS_SLICE_C*tensors->volumn_slice_feature];
     tensors->bbox_feature->data = &((int16_t*)tensors->feature->data)[(CLASS_SLICE_C+CONF_SLICE_C)*tensors->volumn_slice_feature];
     int dims_reshape_bbox[] = {ANCHORS_PER_GRID,4,CONVOUT_H,CONVOUT_W};
     tl_tensor_reshape_src(tensors->bbox_feature, 4, dims_reshape_bbox);

     int dims_reshape_conf1[] = {tensors->conf_feature->len, 1};
     tl_tensor_reshape_src(tensors->conf_feature, 2, dims_reshape_conf1);
     tl_tensor_maxreduce(tensors->conf_feature, tensors->conf_max, tensors->conf_maxidx, 0);
     int dims_reshape_conf2[] = {ANCHORS_PER_GRID,1,CONVOUT_H,CONVOUT_W};
     tl_tensor_reshape_src(tensors->conf_feature, 4, dims_reshape_conf2);

     int coords[4], index, coords_anchor[4];
     tl_tensor_coords(tensors->conf_feature, *(int32_t*)tensors->conf_maxidx->data, coords);
     index = tl_tensor_index(tensors->bbox_feature, coords);
     ((int16_t*)tensors->bbox_int16->data)[0] = ((int16_t*)tensors->bbox_feature->data)[index];
     coords_anchor[0] = coords[2];
     coords_anchor[1] = coords[3];
     coords_anchor[2] = coords[0];
     coords_anchor[3] = coords[1];
     index = tl_tensor_index(tensors->anchors, coords_anchor);
     ((float*)tensors->anchor->data)[0] = ((float*)tensors->anchors->data)[index];

     coords[1] += 1;
     index = tl_tensor_index(tensors->bbox_feature, coords);
     ((int16_t*)tensors->bbox_int16->data)[1] = ((int16_t*)tensors->bbox_feature->data)[index];
     coords_anchor[0] = coords[2];
     coords_anchor[1] = coords[3];
     coords_anchor[2] = coords[0];
     coords_anchor[3] = coords[1];
     index = tl_tensor_index(tensors->anchors, coords_anchor);
     ((float*)tensors->anchor->data)[1] = ((float*)tensors->anchors->data)[index];

     coords[1] += 1;
     index = tl_tensor_index(tensors->bbox_feature, coords);
     ((int16_t*)tensors->bbox_int16->data)[2] = ((int16_t*)tensors->bbox_feature->data)[index];
     coords_anchor[0] = coords[2];
     coords_anchor[1] = coords[3];
     coords_anchor[2] = coords[0];
     coords_anchor[3] = coords[1];
     index = tl_tensor_index(tensors->anchors, coords_anchor);
     ((float*)tensors->anchor->data)[2] = ((float*)tensors->anchors->data)[index];

     coords[1] += 1;
     index = tl_tensor_index(tensors->bbox_feature, coords);
     ((int16_t*)tensors->bbox_int16->data)[3] = ((int16_t*)tensors->bbox_feature->data)[index];
     coords_anchor[0] = coords[2];
     coords_anchor[1] = coords[3];
     coords_anchor[2] = coords[0];
     coords_anchor[3] = coords[1];
     index = tl_tensor_index(tensors->anchors, coords_anchor);
     ((float*)tensors->anchor->data)[3] = ((float*)tensors->anchors->data)[index];

     tl_tensor_convert(tensors->bbox_int16, tensors->bbox_float, TL_FLOAT);
     transform_bbox((float*)tensors->bbox_int16->data, (float*)tensors->anchor->data, result, img_width, img_height);
}
