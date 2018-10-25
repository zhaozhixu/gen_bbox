.PHONY: all test clean

all: target test

target: gen_bbox.c
	@echo "Compiling..."
# @gcc -Wall -fPIC -shared -O3 -g gen_bbox.c -o libgen_bbox.so `pkg-config --libs --cflags tensorlight`
	@gcc -Wall -std=c99 -O3 -Wno-unused-result test/gen_bbox_test.c gen_bbox.c -o test/gen_bbox_test `pkg-config --libs --cflags tensorlight`

test:
	@echo "test with test/convoutTensor_10001.txt "
	@echo "expect:"
	@echo "[ 312.12 168.34 368.83 225.64 ]"
# @echo "Python test result:"
# @python test/gen_bbox.py test/convoutTensor_10001.txt 640 360
	@echo "C test result:"
	@test/gen_bbox_test test/convoutTensor_10001.txt 640 360

clean:
	rm -f libgen_bbox.so test/gen_bbox_test
