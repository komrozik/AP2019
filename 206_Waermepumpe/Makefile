all: build/main.pdf

# hier Python-Skripte:
build/plot_temp.pdf:plot.py programm.py header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python programm.py
	

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/main.pdf: build/plot_temp.pdf

build/main.pdf: FORCE | build
	  TEXINPUTS=build: \
	  BIBINPUTS=build: \
	  max_print_line=1048576 \
	latexmk -pvc\
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	main.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
