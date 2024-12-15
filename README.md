# Python Ray Tracer

## Overview
This project is a Python-based **Ray Tracer** that simulates the physics of light to create photorealistic 3D scene renderings. It leverages ray tracing techniques to calculate realistic lighting, shadows, reflections, and material properties by parsing scene definitions and outputting rendered images.

## Key Features
- **Camera Configuration**: Customizable position, orientation, and screen dimensions.
- **Material Properties**: Support for transparency, diffuse, specular, and reflective attributes.
- **Soft Shadows**: Realistic shadow effects using a grid of light rays.
- **Recursive Ray Tracing**: Handles reflections up to a configurable depth.
- **Scene File Support**: Defines scenes using a simple, text-based file format for objects, lights, and materials.


## Dependencies
This project requires the following Python libraries:
- `argparse`
- `Pillow` (for saving images)
- `numpy` (for vector math)

To install dependencies, run:
```bash
pip install Pillow numpy
```

## Scene File Format
Scene files define the layout and configuration of the environment. Comments start with `#`. Below are the supported keywords:

- **Camera Configuration**: 
  ```
  cam x y z lookAtX lookAtY lookAtZ upX upY upZ screenWidth screenDistance
  ```
- **Global Settings**:
  ```
  set backgroundColorR backgroundColorG backgroundColorB shadowIntensity maxRecursions
  ```
- **Material Definition**:
  ```
  mtl diffuseR diffuseG diffuseB specularR specularG specularB reflectionR reflectionG reflectionB transparency shininess
  ```
- **Geometric Objects**:
  - Sphere: 
    ```
    sph centerX centerY centerZ radius materialIndex
    ```
  - Infinite Plane:
    ```
    pln normalX normalY normalZ distance materialIndex
    ```
  - Cube: 
    ```
    box centerX centerY centerZ scale materialIndex
    ```
- **Lights**:
  ```
  lgt positionX positionY positionZ colorR colorG colorB specularIntensity shadowIntensity radius
  ```

## Usage
Run the program using the following command:
```bash
python main.py <scene_file> <output_image> [--width WIDTH] [--height HEIGHT]
```

### Arguments
- `<scene_file>`: Path to the scene definition file.
- `<output_image>`: Name of the output image file (e.g., `output.png`).
- `--width` (optional): Image width (default: 500).
- `--height` (optional): Image height (default: 500).

### Example
```bash
python main.py scenes/example_scene.txt output.png --width 800 --height 800
```


## Example Scene
two example scenes are provided in the scenes file. 
the out put rendered images are also provided as room.png and pool.png.

## Output
The program generates a rendered image of the scene and saves it as a PNG file.


## Authors
Hend Bishara
Nour Atieh


