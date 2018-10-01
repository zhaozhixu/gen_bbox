#include <tl_tensor.h>

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
static const int ANCHOR_SIZE = 4;
static const int *ANCHOR_SHAPES = {229, 137, 48, 71, 289, 245,
                                   185, 134, 85, 142, 31, 41,
                                   197, 191, 237, 206, 63, 108};

static const float E = 2.718281828;

/* convinent macro to define an array */
#define ARR(type, varg...) (type[]){varg...}

struct pre_alloc_tensors {
     tl_tensor *anchors;
     tl_tensor *feature;
     tl_tensor *conf_feature, *bbox_feature;
     tl_tensor *conf_transpose, *bbox_transpose;
     tl_tensor *bbox_final;
};

struct pre_alloc_tensors *preprocess(void)
{
     struct pre_alloc_tensors *tensors;

     tensors = tl_alloc(sizeof(struct pre_alloc_tensors));
     tensors->feature = tl_tensor_create(NULL, 3, ARR(int,CONVOUT_C,CONVOUT_H,CONVOUT_W),
                                         TL_INT16);
     tensors->conf_feature = tl_tensor_zeros(3, ARR(int,CONF_SLICE_C,CONVOUT_H,CONVOUT_W),
                                             TL_INT16);
     tensors->bbox_feature = tl_tensor_zeros(3, ARR(int,BBOX_SLICE_C,CONVOUT_H,CONVOUT_W),
                                             TL_INT16);
     tensors->conf_transpose = tl_tensor_zeros(4, ARR(int,CONVOUT_H,CONVOUT_W,ANCHORS_PER_GRID,1),
                                               TL_INT16);
     tensors->bbox_transpose = tl_tensor_zeros(4, ARR(int,CONVOUT_H,CONVOUT_W,ANCHORS_PER_GRID,4),
                                               TL_INT16);
     tensors->bbox_final = tl_tensor_zeros(1, ARR(float,4), TL_FLOAT);
}

float *feature_express(int16_t *feature, int img_width, int img_height,
                       struct pre_alloc_tensors *tensors)
{
     int order;

}
