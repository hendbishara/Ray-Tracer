import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

from intersection import Intersection
from ray import Ray
import time

EPS = 1e-6



def normalize(v):
    """Normalize a vector"""
    norm = np.linalg.norm(v)
    return v / norm 

def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(np.array(params[:3],dtype = 'float'), np.array(params[3:6],dtype = 'float'), np.array(params[6:9],dtype = 'float'), params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(np.array(params[:3],dtype = 'float'), params[3], params[4])
            elif obj_type == "mtl":
                material = Material(np.array(params[:3],dtype='float'), np.array(params[3:6],dtype='float'), np.array(params[6:9],dtype='float'), params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(np.array(params[:3],dtype = 'float'), params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(np.array(params[:3],dtype = float), params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(np.array(params[:3],dtype = 'float'), params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(np.array(params[:3],dtype = 'float'), np.array(params[3:6],dtype = 'float'), params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects

'''get a sortde list of all shapes that intersect with ray'''
def all_intersections(ray, shapes):
    intersections = [shape.intersect(ray) for shape in shapes if shape.intersect(ray) and shape.intersect(ray).t > EPS]
    return sorted(intersections, key=lambda inter: inter.t)

"""get normal of object in point"""
def calc_normal(obj, point):
    if isinstance(obj,InfinitePlane):
        n = obj.normal
    elif(isinstance(obj,Cube)):
        n = calc_cube_normal(obj,point)
    else:
        n = normalize(point-obj.position)
    
    return n

"""get cube normal in point"""
def calc_cube_normal(cube, point):
    # Calculate min and max corners of the cube
    half_length = cube.scale / 2
    center = cube.position
    cube_min = np.array([center[0] - half_length, center[1] - half_length, center[2] - half_length])
    cube_max = np.array([center[0] + half_length, center[1] + half_length, center[2] + half_length])

    # Determine which face of the cube the point lies on
    normal = np.zeros(3)
    
    if abs(point[0] - cube_min[0]) < EPS:
        normal = np.array([-1, 0, 0])
    elif abs(point[0] - cube_max[0]) < EPS:
        normal = np.array([1, 0, 0])
    elif abs(point[1] - cube_min[1]) < EPS:
        normal = np.array([0, -1, 0])
    elif abs(point[1] - cube_max[1]) < EPS:
        normal = np.array([0, 1, 0])
    elif abs(point[2] - cube_min[2]) < EPS:
        normal = np.array([0, 0, -1])
    elif abs(point[2] - cube_max[2]) < EPS:
        normal = np.array([0, 0, 1])

    return normal


"""get the fraction of light rays that intersect a point from the light's radius"""
def get_intensity(light,intersection,N,shapes):
   width = light.radius
   center_point = light.position
   normal_vec = normalize(intersection.point-center_point)
   # Compute the perpendicular basis vectors
   u, w = find_perpendicular_vectors(normal_vec)
    
   # Find the bottom-left corner of the grid
   corner = center_point - (width / 2) * u - (width / 2) * w
   
   cell_size = width / N
   
   if (N>2):
        # Define corners:
        center = center_point
        left_buttom = corner
        right_buttom = center_point - (width/2) * u + (width/2) * w
        top_left = center_point + (width / 2) * u - (width / 2) * w
        top_right = center_point + (width / 2) * u + (width / 2) * w
        
        corners = [center, left_buttom, right_buttom,top_left, top_right]
        
        corner_cnt_1 = 0
        corner_cnt_2 = 0
        # check if in all corners there is light or no light
        for c in corners:
            if check_condition(c, shapes, light, intersection.point):
                corner_cnt_1 +=1
            else:
                corner_cnt_2 +=1
        
        if corner_cnt_1 == len(corners):
            return 1
            
        if corner_cnt_2 == len(corners):
            return 0
   #if not go through the whole grid
   cnt = 0
   for i in range(int(N)):
        for j in range(int(N)):
            # Pick a random point within the cell
            random_point = corner + (i * cell_size + np.random.uniform(0, cell_size)) * u + (j * cell_size + np.random.uniform(0, cell_size)) * w
            if check_condition(random_point, shapes, light, intersection.point):
                cnt += 1
            
   return cnt / (N * N)

"""check if light ray directly hits point"""
def check_condition(point, shapes, light, target_point):
    shadow_ray = Ray(point, normalize(target_point - point), light)
    intersections = all_intersections(shadow_ray,shapes)
    intersection_to_light_dist = np.linalg.norm(point - target_point) #distance between point and light source
        
    if len(intersections) != 0:
        #check if the distance of intersection equal to the distance of the point from light ray source
        return abs(intersection_to_light_dist - intersections[0].t) < (0.00001)
    else:
        return False

"""returns two perpendicular verctors of the vector v, the two vectors define a perpendicular plane"""
def find_perpendicular_vectors(v):
    
    # Choose an arbitrary vector that is not aligned with v
    if v[0] != 0 or v[1] != 0:
        arbitrary_vector = np.array([0, 0, 1])
    else:
        arbitrary_vector = np.array([1, 0, 0])
    
    # Compute the first perpendicular vector
    u = normalize(np.cross(v, arbitrary_vector))
    # Compute the second perpendicular vector
    w = normalize(np.cross(u, v))
    
    return u, w




'''calculate the color of pixel in the screen'''
def pix_color(intersections, scene_settings,shapes, lights, materials, limit):
    background = np.copy(scene_settings.background_color)
    if  limit==0: #reached the bais case
        return background
    #get the shapes list till the untrasperante object for the ray color calculation
    for i in range(len(intersections)):
        mat = materials[intersections[i].shape.material_index-1]
        if mat.transparency == 0:
            intersections = intersections[0:i + 1]
            break
    
    color=np.copy(background)
    #iterate over all shapes that intersect with the ray
    while intersections:
        hit=intersections.pop()
        mat = materials[hit.shape.material_index-1]
        transparency = mat.transparency
        
        #get diffuse and spacular color with soft shadows
        color = compute_diffuse_specular(hit,scene_settings,shapes, lights, materials)
        
        #clculate the reflected ray
        normal=calc_normal(hit.shape,hit.point)
        reflection_ray = Ray(hit.point, -reflected_diretction(hit.ray.direction, normal),hit.shape)
        reflected_intersections = all_intersections(reflection_ray,shapes)
        
        #recursive call with the reflected ray
        reflection_color = pix_color(reflected_intersections, scene_settings,shapes, lights, materials, limit - 1)
        reflection_color *= mat.reflection_color
        #calculate the color using the given equation
        color = ((1 - transparency) * color + transparency * background) + reflection_color
        background = np.copy(color)

    return color

'''get the reflected ray direction'''
def reflected_diretction(w, n):
    return 2 * np.dot(w, n) * n - w

'''calculate the diffuse and specular color'''
def compute_diffuse_specular(intersect, scene_settings, shapes, lights, materials):
    diffuse = np.zeros(3, dtype="float")
    specular = np.zeros(3, dtype="float")
    material = materials[intersect.shape.material_index - 1]

    #iterate over each light 
    for light in lights:
        N = calc_normal(intersect.shape, intersect.point) 
        L = normalize(light.position - intersect.point)
        N_dot_L = np.dot(N, L)
        
        if N_dot_L > 0:
            #get the light intensety in the intersected point 
            intensity = get_intensity(light,intersect,scene_settings.root_number_shadow_rays,shapes)
            light_intensity = (1-light.shadow_intensity)+light.shadow_intensity*intensity
            V = normalize(intersect.ray.origin - intersect.point)
            R = reflected_diretction(L, N)
            #diffuse color
            diffuse += light.color * light_intensity * N_dot_L
            #specular color
            specular += light.color * light_intensity * light.specular_intensity * (np.dot(R, V) ** material.shininess)
    
    return (diffuse * material.diffuse_color + specular * material.specular_color)


'''divide objects to shapes lights and materilas'''
def shaps_lights_materials(objects):
    shapes=[]
    lights=[]
    materials=[]
    for i in range(len(objects)):
        if isinstance(objects[i],Cube) or isinstance(objects[i],Sphere) or isinstance(objects[i],InfinitePlane):
            shapes.append(objects[i])
        elif isinstance(objects[i],Light):
            lights.append(objects[i]) 
        else:
            materials.append(objects[i])
    return shapes , lights , materials

'''fix the camera crodenates to be perpendicular'''
def camera_fix(camera):
    direction = normalize(camera.look_at - camera.position)
    right = normalize(np.cross(direction, camera.up_vector))
    up_vector = normalize(np.cross(right, direction))
    camera.fix(direction,right,up_vector)

def save_image(image_array,out_name):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save(f"{out_name}")

'''main function that runs the ray tracer algorithem'''
def ray_trace(camera,scene_settings,shapes,lights,materials,width,height):
    ratio = camera.screen_width/width #ratio
    final_image = np.zeros((height, width, 3))
    #screen center for the camera to look at
    screen_center = camera.position + camera.screen_distance * camera.direction
    #for each pixel
    for i in range(height):
        y_offset = (i-height // 2) * ratio * camera.up_vector
        for j in range(width):
            x_offset = (j - width // 2) * ratio * camera.right
            direction = normalize((screen_center + x_offset - y_offset) - camera.position)
            #create the ray for it that starts from camera postion
            ray = Ray(camera.position, direction, camera)
            #get all the intesrcations with that ray
            intersections = all_intersections(ray, shapes)
            #calculate the color of the relevant pixel
            color = pix_color(intersections, scene_settings, shapes, lights, materials, scene_settings.max_recursions)
            if np.sum(color,axis=0)<EPS:
                pass
            final_image[i, width-j-1] = np.clip(color, 0, 1) * 255
    return final_image

def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()
    
    
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    shapes, lights, materials = shaps_lights_materials(objects)
    camera_fix(camera)
    
    width = args.width
    height = args.height
    
    final_image = ray_trace(camera,scene_settings,shapes,lights,materials,width,height)
    

    # Save the output image
    save_image(final_image,args.output_image)

if __name__ == '__main__':
    main()
