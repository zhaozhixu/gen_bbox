.PHONY: all test clean

all: target test

target: gen_bbox.c tl_tensor.c tl_type.c tl_util.c
	@echo "Compiling..."
# @gcc -Wall -fPIC -shared -O3 -g gen_bbox.c -o libgen_bbox.so `pkg-config --libs --cflags tensorlight`
	@gcc -Wall -std=c99 -fPIC -shared -O2 -Wno-unused-result gen_bbox.c tl_tensor.c tl_type.c tl_util.c -o libgen_bbox.so -lm

test:
	@gcc -Wall -std=c99 -O2 -Wno-unused-result test/gen_bbox_test.c libgen_bbox.so -o test/gen_bbox_test
	@echo "test with test/convoutTensor_10001.txt "
	@echo "expect:"
	@echo "[ 312.12 168.34 368.83 225.64 ]"
	@echo "Python test result:"
	@python test/gen_bbox.py test/convoutTensor_10001.txt 640 360
	@echo "C test result:"
	@test/gen_bbox_test test/convoutTensor_10001.txt 640 360

clean:
	rm -f libgen_bbox.so test/gen_bbox_test
