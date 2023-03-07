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

# Usage

Sample Example

<img src="https://github.com/richylyq/SegFormer-FastAPI/blob/master/src/results/original.jpg" width="300" height="300">

* `/solid` - Create Solid Mask

<img src="https://github.com/richylyq/SegFormer-FastAPI/blob/master/src/results/solid.png" width="300" height="300">

* `/semi` - Create Semi Mask

<img src="https://github.com/richylyq/SegFormer-FastAPI/blob/master/src/results/semi.png" width="300" height="300">

* `/get_class_names` - Get Class Names of segmentated results
```
[
  "ceiling",
  "lamp",
  "wall",
  "painting",
  "light",
  "curtain",
  "sofa",
  "cushion",
  "table",
  "blanket",
  "clock",
  "plate",
  "coffee table",
  "vase",
  "floor",
  "book",
  "tray",
  "ball",
  "ottoman",
  "bed "
]
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
