#/bin/bash

# python3 api.py
uvicorn api:segformer --reload --host 0.0.0.0 --port 13035
