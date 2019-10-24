
#################################################
all: build/Protokoll.pdf

build/Protokoll.pdf: Protokoll.tex build/plot.pdf | build
	lualatex --output-directory=build --interaction=batchmode --halt-on-error Protokoll.tex
	lualatex --output-directory=build --interaction=batchmode --halt-on-error Protokoll.tex

build:
	mkdir -p build

clean:
	rm -rf build
