PYTHON=python3
VENV=.venv
PIP=$(VENV)/bin/pip
PY=$(VENV)/bin/python

# Create virtual environment + install dependencies
install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Train the model using the prebuilt training CSV
run:
	jupyter nbconvert --to notebook --execute model.ipynb --output executed_model.ipynb
