# Define the python interpreter
PYTHON := python3.8
VENV := chp1
BIN := $(VENV)/bin

.PHONY: setup install lint test clean

# Create virtual environment and install dependencies
setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	@echo "Setup complete. Run 'source $(VENV)/bin/activate' to start."
	

# Install requirements
install:
	$(BIN)/pip install -r requirements.txt

# Run Linting (requires ruff or flake8)
lint:
	$(BIN)/pip install ruff
	$(BIN)/ruff check .

# Run Tests (requires pytest)
test:
	$(BIN)/pip install pytest
	$(BIN)/pytest tests/tests_main.py

# Clean up temporary files
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +