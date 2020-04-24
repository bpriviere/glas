all: gen nn-export-test swig

gen:
	./gen.sh

nn-export-test:
	gcc -std=c99 nn.c main.c -lm -o nn-export-test

swig:
	swig -python nnexport.i
	gcc -std=c99 -shared -fPIC \
		-I/usr/include/python3.6 \
		nnexport_wrap.c \
		nn.c \
		-lm -lpython3.6m \
		-g \
		-o _nnexport.so

clean:
	rm -f generated_weights.c nn-export-test nnexport.py nnexport.pyc _nnexport.so __init__.pyc
