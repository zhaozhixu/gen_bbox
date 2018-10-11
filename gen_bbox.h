#ifndef _GEN_BBOX_H_
#define _GEN_BBOX_H_

struct pre_alloc_tensors;

struct pre_alloc_tensors *preprocess(void);
void feature_express(int16_t *feature, int img_width, int img_height,
                     struct pre_alloc_tensors *tensors, float *result);
void postprocess(struct pre_alloc_tensors *tensors);

#endif  /* _GEN_BBOX_H_ */
