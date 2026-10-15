REPORT_SRC = report.tex

clean:
	@rm -f *.aux *.log *.out *.pdf *.toc

help:
	@echo " make report to Build this report"
	@echo " make help to Display this help"
	
report: $(REPORT_SRC)
	pdflatex -pdf -shell-escape $(REPORT_SRC)
	
.PHONY: clean help report
