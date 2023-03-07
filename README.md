# Overview
FastAPI implementation of SegFormer

## I/O
Input is of image format:
```
{
  "image_binary": <bytes> # image, in bytes
}
```
Output is of image format:
``` 
{
  "image": <StreamingResponse> # SegFormer result
}
```

# Getting Started

## Prerequisites
- Docker
- docker-compose

## Usage
For development:
`docker-compose up`

## Parameters
* `CUDA_VISIBLE_DEVICES`: Exposes GPU to docker container.

## Dependencies
### Major Frameworks
- Python 3.6

### Minor Dependencies

# Technical Specification
- make sure the docker image, container and fastapi names are all the same

# References

* Original GitHub repo: https://github.com/NVlabs/SegFormer
