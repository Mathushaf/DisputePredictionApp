#!/bin/bash
export PYTHONUNBUFFERED=1
gunicorn UIwithflaskR1.app1:app --bind 0.0.0.0:$PORT
