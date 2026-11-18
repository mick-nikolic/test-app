run: install
	cd src; conda run -n mlops python main.py

check: install
	ruff check src/.; rm -rf .ruff*
format: install
	ruff format --check src/.; ruff format --diff src/.; rm -rf .ruff*

clean:
	rm -rf src/config/__pycache__; rm -rf src/db/__pycache__; rm -rf src/model/__pycache__; rm -rf src/model/pipeline/__pycache__

install: environment.yml
	conda env create -f environment.yml



