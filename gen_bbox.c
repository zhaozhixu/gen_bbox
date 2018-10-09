#include <tl_tensor.h>
#include <math.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* convinent macro to define an array */
#define ARR(type, varg...) (type[]){varg}

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
     tl_tensor *conf_transpose, *bbox_transpose;
     tl_tensor *conf_workspace, *bbox_workspace;
     tl_tensor *conf_max, *conf_maxidx;
     tl_tensor *bbox_int16, *bbox_float, *anchor;
     tl_tensor *bbox_final;
};

struct pre_alloc_tensors *preprocess(void)
{
     struct pre_alloc_tensors *tensors;

     tensors = tl_alloc(sizeof(struct pre_alloc_tensors));
     tensors->feature = tl_tensor_create(NULL, 3, ARR(int,CONVOUT_C,CONVOUT_H,CONVOUT_W), TL_INT16);
     tensors->conf_feature = tl_tensor_zeros(3, ARR(int,CONF_SLICE_C,CONVOUT_H,CONVOUT_W), TL_INT16);
     tensors->bbox_feature = tl_tensor_zeros(3, ARR(int,BBOX_SLICE_C,CONVOUT_H,CONVOUT_W), TL_INT16);
     tensors->conf_transpose = tl_tensor_zeros(4, ARR(int,CONVOUT_H,CONVOUT_W,ANCHORS_PER_GRID,1), TL_INT16);
     tensors->bbox_transpose = tl_tensor_zeros(4, ARR(int,CONVOUT_H,CONVOUT_W,ANCHORS_PER_GRID,4), TL_INT16);
     tensors->conf_workspace = tl_tensor_zeros(1, ARR(int,tensors->conf_transpose->ndim*tensors->conf_transpose->len*2), TL_INT32);
     tensors->bbox_workspace = tl_tensor_zeros(1, ARR(int,tensors->bbox_transpose->ndim*tensors->bbox_transpose->len*2), TL_INT32);
     tensors->conf_max = tl_tensor_zeros(1, ARR(int,1), TL_INT16);
     tensors->conf_maxidx = tl_tensor_zeros(1, ARR(int,1), TL_INT32);
     tensors->bbox_int16 = tl_tensor_zeros(1, ARR(int,4), TL_INT16);
     tensors->bbox_float = tl_tensor_zeros(1, ARR(int,4), TL_FLOAT);
     tensors->anchor = tl_tensor_zeros(1, ARR(int,4), TL_FLOAT);
     tensors->bbox_final = tl_tensor_zeros(1, ARR(int,4), TL_FLOAT);

     tl_tensor *anchor_shapes = tl_tensor_create(ANCHOR_SHAPES, 2, ARR(int,ANCHORS_PER_GRID,2), TL_FLOAT);
     tl_tensor *all_anchor_shapes = tl_tensor_repeat(anchor_shapes, CONVOUT_H*CONVOUT_W);
     tl_tensor_reshape(all_anchor_shapes, 4, ARR(int,CONVOUT_H,CONVOUT_W,ANCHORS_PER_GRID,2));

     tl_tensor *center_x_conv = tl_tensor_arange(1, CONVOUT_W+1, 1, TL_FLOAT);
     tl_tensor *center_x_input = tl_tensor_elew_param(center_x_conv, (float)INPUT_W/((float)CONVOUT_W+1), NULL, TL_MUL);
     tl_tensor *center_x_input_all = tl_tensor_repeat(center_x_input, CONVOUT_H*ANCHORS_PER_GRID);
     tl_tensor_reshape(center_x_input_all, 3, ARR(int,ANCHORS_PER_GRID,CONVOUT_H,CONVOUT_W));
     tl_tensor *center_x_input_all_trans = tl_tensor_transpose(center_x_input_all, NULL, ARR(int,1,2,0), NULL);
     tl_tensor_reshape(center_x_input_all_trans, 4, ARR(int,CONVOUT_H,CONVOUT_W,ANCHORS_PER_GRID,1));

     tl_tensor *center_y_conv = tl_tensor_arange(1, CONVOUT_H+1, 1, TL_FLOAT);
     tl_tensor *center_y_input = tl_tensor_elew_param(center_y_conv, (float)INPUT_H/((float)CONVOUT_H+1), NULL, TL_MUL);
     tl_tensor *center_y_input_all = tl_tensor_repeat(center_y_input, CONVOUT_W*ANCHORS_PER_GRID);
     tl_tensor_reshape(center_y_input_all, 3, ARR(int,ANCHORS_PER_GRID,CONVOUT_W,CONVOUT_H));
     tl_tensor *center_y_input_all_trans = tl_tensor_transpose(center_y_input_all, NULL, ARR(int,2,1,0), NULL);
     tl_tensor_reshape(center_y_input_all_trans, 4, ARR(int,CONVOUT_H,CONVOUT_W,ANCHORS_PER_GRID,1));

     tl_tensor *concat1 = tl_tensor_concat(center_x_input_all_trans, center_y_input_all_trans, NULL, 3);
     tl_tensor *concat2 = tl_tensor_concat(concat1, all_anchor_shapes, NULL, 3);
     tl_tensor_reshape(concat2, 2, ARR(int,concat2->len/4,4));

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

void postprocess(struct pre_alloc_tensors *tensors)
{
     tl_tensor_free_data_too(tensors->anchors);
     tl_tensor_free_data_too(tensors->bbox_feature);
     tl_tensor_free_data_too(tensors->bbox_final);
     tl_tensor_free_data_too(tensors->bbox_transpose);
     tl_tensor_free_data_too(tensors->bbox_workspace);
     tl_tensor_free_data_too(tensors->conf_feature);
     tl_tensor_free_data_too(tensors->conf_transpose);
     tl_tensor_free_data_too(tensors->conf_workspace);
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

void feature_express(int16_t *feature, int img_width, int img_height,
                     struct pre_alloc_tensors *tensors, float *result)
{
     tensors->feature->data = feature;
     tl_tensor_slice(tensors->feature, tensors->conf_feature, 0, CLASS_SLICE_C, CONF_SLICE_C);
     tl_tensor_slice(tensors->feature, tensors->bbox_feature, 0, CLASS_SLICE_C+CONF_SLICE_C, BBOX_SLICE_C);
     tl_tensor_reshape(tensors->conf_feature, 4, ARR(int,ANCHORS_PER_GRID,1,CONVOUT_H,CONVOUT_W));
     tl_tensor_reshape(tensors->bbox_feature, 4, ARR(int,ANCHORS_PER_GRID,4,CONVOUT_H,CONVOUT_W));
     tl_tensor_transpose(tensors->conf_feature, tensors->conf_transpose, ARR(int,2,3,0,1), tensors->conf_workspace);;
     tl_tensor_transpose(tensors->bbox_feature, tensors->bbox_transpose, ARR(int,2,3,0,1), tensors->bbox_workspace);;
     tl_tensor_reshape(tensors->conf_transpose, 2, ARR(int,tensors->conf_transpose->len,1));
     tl_tensor_reshape(tensors->bbox_transpose, 2, ARR(int,tensors->bbox_transpose->len/4,4));
     tl_tensor_maxreduce(tensors->conf_transpose, tensors->conf_max, tensors->conf_maxidx, 0);
     tl_tensor_slice(tensors->bbox_transpose, tensors->bbox_int16, 0, *(int32_t*)tensors->conf_maxidx->data, 1);
     tl_tensor_slice(tensors->anchors, tensors->anchor, 0, *(int32_t*)tensors->conf_maxidx->data, 1);
     tl_tensor_convert(tensors->bbox_int16, tensors->bbox_float, TL_FLOAT);
     transform_bbox((float*)tensors->bbox_int16->data, (float*)tensors->anchor->data, result, img_width, img_height);
}