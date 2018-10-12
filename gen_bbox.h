#ifndef _GEN_BBOX_H_
#define _GEN_BBOX_H_

struct pre_alloc_tensors;

#ifdef __cplusplus
#define GB_CPPSTART extern "C" {
#define GB_CPPEND }
GB_CPPSTART
#endif

struct pre_alloc_tensors *gb_preprocess(void);
void gb_getbbox(int16_t *feature, int img_width, int img_height,
                struct pre_alloc_tensors *tensors, float *result);
void gb_postprocess(struct pre_alloc_tensors *tensors);

#ifdef __cplusplus
GB_CPPEND
#endif

#endif  /* _GEN_BBOX_H_ */
