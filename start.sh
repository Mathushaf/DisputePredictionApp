#!/bin/bash
gunicorn --workers=1 --timeout=600 --bind 0.0.0.0:10000 UIwithflaskR1.app1:app