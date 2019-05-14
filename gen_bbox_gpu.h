#ifndef _GEN_BBOX_DPU_H_
#define _GEN_BBOX_DPU_H_

struct pre_alloc_data;

#ifdef __cplusplus
#define GBD_CPPSTART extern "C" {
#define GBD_CPPEND }
GBD_CPPSTART
#endif

/*
 * Allocate necessary data structures.
 */
struct pre_alloc_data *gbd_preprocess(void);

/*
 * Release 'data' allocated by gbd_preprocess().
 */
void gbd_postprocess(struct pre_alloc_data *data);

/*
 * 'data' should be returned by gbd_preprocess().
 * 'anchors' is of shape [H_VALID, W_VALID, ANCHORS_PER_GRID, BBOX_SIZE]
 * 'feature' is of shape [H_FULL, W_PER_GROUP, C_GROUP, W_GROUPS, W_PER_GROUP].
 * 'bbox' is the result bounding box, allocated by the caller.
 */
void gbd_getbbox(struct pre_alloc_data *data, int8_t *feature, float *anchors,
                 float *bbox);

#ifdef __cplusplus
GBD_CPPEND
#endif

#endif  /* _GEN_BBOX_DPU_H_ */
