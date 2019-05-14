.PHONY: all test clean

all: target test

target: gen_bbox_dpu.c
	@echo "Compiling..."
	@gcc -Wall -std=c99 -fPIC -shared -O2 -Wno-unused-result gen_bbox_dpu.c -o libgen_bbox_dpu.so -lm

test:
	@gcc -Wall -std=c99 -O2 -Wno-unused-result test/gen_bbox_test.c libgen_bbox_dpu.so -o test/gen_bbox_test
	@echo "test with test/test.txt "
	@echo "expect [xmin xmax ymin ymax]:"
	@echo "[ 317 354 129 200 ]"
	@echo "Python test result:"
	@python test/gen_bbox_test.py test/test.txt
	@echo "C test result:"
	@test/gen_bbox_test test/test.txt

clean:
	rm -f libgen_bbox_dpu.so test/gen_bbox_test
