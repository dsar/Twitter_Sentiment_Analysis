FILE = report
TMP_DIR = tmp
GARBAGE = $(FILE).log $(FILE).nav $(FILE).out $(FILE).toc $(FILE).vrb $(FILE).snm $(FILE).bbl $(FILE).lof $(FILE).bbl $(FILE).blg $(FILE).glo $(FILE).ist $(FILE).lot *.aux $(FILE).synctex.gz
PDFLATEX = pdflatex --output-directory=$(TMP_DIR)
BIBTEX = bibtex 
PRINTER = lp
UNAME := $(shell uname)


all: $(FILE).tex clean
	mkdir -p $(TMP_DIR)
	$(PDFLATEX) $(FILE).tex
	$(BIBTEX) $(TMP_DIR)/$(FILE).aux
	$(PDFLATEX) $(FILE).tex
	$(PDFLATEX) $(FILE).tex
	mv $(TMP_DIR)/$(FILE).pdf .
ifeq ($(UNAME), Darwin)
	open -g  $(FILE).pdf
else
	evince $(FILE).pdf &
endif


i:
	@mkdir -p $(TMP_DIR)
	@$(PDFLATEX) $(FILE).tex 2>&1 > /dev/null
	@$(BIBTEX) $(TMP_DIR)/$(FILE).aux | grep --color=yes Warning || echo > /dev/null
	@$(PDFLATEX) $(FILE).tex 2>&1 > /dev/null
	@$(PDFLATEX) $(FILE).tex | grep --color=yes -A 1 Warning | grep -v "^--" | grep -v "^$$" || echo > /dev/null
	@mv $(TMP_DIR)/$(FILE).pdf .


clean:
	rm -f $(FILE).pdf && rm -rf $(TMP_DIR) && rm -rf $(GARBAGE) && rm -rf $(MERGED_DIR) && rm -f $(MERGED_FILE).pdf


p:	
	lpr -P$(PRINTER) $(FILE).pdf
	lpq -P$(PRINTER)
