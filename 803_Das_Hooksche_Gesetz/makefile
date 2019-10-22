all : build/Protokoll.pdf

build/Protokoll.pdf: Protokoll.tex |build
	latexmk -pvc --lualatex --output-directory=build --interaction=nonstopmode --halt-on-error Protokoll.tex
	


build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: FORCE all clean