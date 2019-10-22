all: build/Protokoll.pdf

build/plot.pdf: programm.py | build
	TEXINPUTS=$$(pwd): python programm.py

build/Protokoll.pdf: build/plot.pdf

build/Protokoll.pdf: FORCE | build
	    TEXINPUTS=build: \
    	BIBINPUTS=build: \
    	max_print_line=1048576 \
	latexmk \
	  --lualatex \
      --output-directory=build \
      --interaction=nonstopmode \
      --halt-on-error \
    Protokoll.tex


build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: FORCE all clean