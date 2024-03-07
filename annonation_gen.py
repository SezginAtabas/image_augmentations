import blenderproc as bproc
import mathutils
import numpy as np
import os
import math

blend_path = ""
output_path = ""
hdr_path = ""

# how many times the scene is rendered
render_count = 1
# how many times randomizer is triggered
randomize_count = 1
# name of the target object that will be targed
target_name = "avocado"
cluster_name = "avs"

# params for randomizer
light_energy_range = [1, 2]
light_rotation_amount = [[-5, 5], [-5, 5], [-5, 5]]

target_rotation_amount = [[-30, 30], [-30, 30], [-30, 30]]
target_default_rotation = [0, -90, 0]

mat_specular_range = [0.2, 0.4]

bproc.init()

objs = bproc.loader.load_blend(blend_path)

target_objects = []
target_clusters = []

# filter the objects to get only the target objects
for obj in objs:
    obj_name = obj.get_name()
    if target_name in obj_name:
        target_objects.append(obj)
        obj.set_cp("category_id", 1)

    elif cluster_name in obj_name:
        target_clusters.append(obj)
        
# find the target material
materials = bproc.material.collect_all()
target_material = bproc.filter.one_by_attr(materials, "name", "Material.002")

# create light
light = bproc.types.Light()
light.set_type("SUN")
light.set_location([0.0, 0.0, 50.0])
light.set_rotation_euler([30, 5, 5])
light.set_energy(1)
        
bproc.world.set_world_background_hdr_img(hdr_path)
bproc.camera.set_resolution(720, 360)
#bproc.camera.set_intrinsics_from_blender_params(lens=2.8, lens_unit="MILLIMETERS")

target_default_rotation = list(map(math.radians, target_default_rotation))


# add the poi of all targets to poi list
poi_list = [bproc.object.compute_poi(target_objects), ]

# find the poi to look at for each target cluster
for cluster in target_clusters:
    
    # get all target objects in the cluster
    cluster_children = cluster.get_children()
    
    # calculate poi and add it to list
    poi_list.append(bproc.object.compute_poi(cluster_children))



# Camera settings for each POI
num_steps_per_poi = 25
camera_radius = 20
camera_height = 8
walk_magnitude = np.pi/32  # Maximum random rotation per step

for render_c in range(render_count):        
    bproc.utility.reset_keyframes()
    for i in range(randomize_count):
        
        for poi in poi_list:        
            # randomize object rotation
            for obj in target_objects:
                        
                # random angles
                rand_rotation = np.random.uniform(
                    [target_rotation_amount[0][0], target_rotation_amount[1][0], target_rotation_amount[2][0]],
                    [target_rotation_amount[0][1], target_rotation_amount[1][1], target_rotation_amount[2][1]]
                    )
                
                # convert to radians
                rand_rotation = list(map(math.radians, rand_rotation))
                
                # randomize objects rotation
                obj.set_rotation_euler([r1 + r2 for r1, r2 in zip(rand_rotation, target_default_rotation)])
                    
            # randomize material
                target_material.set_principled_shader_value("Specular", np.random.uniform(mat_specular_range[0], mat_specular_range[1]))
            
            # randomize light rotation
            current_light_rotation = light.get_rotation_euler()      
            light_random_rotation = np.random.uniform(
                [light_rotation_amount[0][0], light_rotation_amount[1][0], light_rotation_amount[2][0]],
                [light_rotation_amount[0][1], light_rotation_amount[1][1], light_rotation_amount[2][1]]
                )
            
            # convert to radians
            light_random_rotation = list(map(math.radians, light_random_rotation))     
            # set light rotation
            light.set_rotation_euler([ r1 + r2 for r1, r2 in zip(light_random_rotation, current_light_rotation)])
                
            # randomize light energy
            light.set_energy(np.random.uniform(light_energy_range[0], light_energy_range[1]))
            
            #----- random walk starts here -----#
            
            # translational random walk
            poi_drift = bproc.sampler.random_walk(
                total_length=num_steps_per_poi, dims=3, step_magnitude=0.005,
                window_size=5, interval=[-0.03, 0.03], distribution='uniform'
            )
            
            # Camera shaking (rotational random walk)
            camera_shaking_rot_angle = bproc.sampler.random_walk(
                total_length=num_steps_per_poi, dims=1, step_magnitude=walk_magnitude,
                window_size=5, interval=[-walk_magnitude * 4, walk_magnitude * 4], distribution='uniform'
            )
            
            camera_shaking_rot_axis = bproc.sampler.random_walk(
                total_length=num_steps_per_poi, dims=3, window_size=10, distribution='normal'
            )
            
            camera_shaking_rot_axis /= np.linalg.norm(camera_shaking_rot_axis, axis=1, keepdims=True)

            # Loop for camera poses around each POI
            for i in range(num_steps_per_poi):
                
                # Camera trajectory (circular path)
                angle = i / (num_steps_per_poi - 1) * 2 * np.pi
                location_cam = np.array([
                    camera_radius * np.cos(angle),
                    camera_radius * np.sin(angle),
                    camera_height
                ]) + poi + poi_drift[i]

                # Look at the POI with added random rotation from shaking
                look_at = poi + poi_drift[i]
                camera_rot_axis = camera_shaking_rot_axis[i]
                camera_rot_angle = camera_shaking_rot_angle[i]

                # Apply shaking to rotation matrix
                R_rand = np.array(mathutils.Matrix.Rotation(camera_rot_angle, 3, camera_rot_axis))
                rotation_matrix = bproc.camera.rotation_from_forward_vec(look_at - location_cam)
                rotation_matrix = R_rand @ rotation_matrix

                # Add camera pose with location and rotation
                cam2world_matrix = bproc.math.build_transformation_mat(location_cam, rotation_matrix)
                bproc.camera.add_camera_pose(cam2world_matrix)
         
    # activate normal rendering
    #bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(default_values={'category_id': 99}, map_by=["category_id", "instance", "name"])
    
    # render the whole pipeline
    data = bproc.renderer.render()

    # Write data to coco file
    bproc.writer.write_coco_annotations(os.path.join(output_path, 'coco_data'),
                                        instance_segmaps=data["instance_segmaps"],
                                        instance_attribute_maps=data["instance_attribute_maps"],
                                        colors=data["colors"],
                                        color_file_format="PNG")





















