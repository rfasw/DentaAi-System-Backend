#!/usr/bin/env bash
set -o errexit

# Force Python 3.10
PYTHON_VERSION="3.10.0"
pyenv install --skip-existing $PYTHON_VERSION
pyenv global $PYTHON_VERSION

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn  # Ensure gunicorn is installed

# Additional build steps (if needed)
# python manage.py collectstatic --noinput