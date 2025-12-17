# Data collector for Datasets  in HabitatSim
# Available for: Replica, HM3D, MP3D
# Single Scene setting and run, set the config to choose your desired dataset and save path

import os
import time
import hydra
import threading
import habitat_sim

from omegaconf import DictConfig

# Conditionally import ROSDataCollector
from .utils.ros_data_collector import *
# Useful utils
from .utils.habitat_utils import *

# ROS 2 Utils (only if ROS 2 is available)
ros_enabled = ROSDataCollector is not None

@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent / "config"),
    config_name="habitat_data_collector.yaml"
)
def main(cfg: DictConfig) -> None:

    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    print(cfg.output_path)
    os.makedirs(cfg.output_path, exist_ok=True)
    dataset_dir = Path(cfg.output_path) / cfg.dataset_name

    # Initialize ROS 2 node if enabled and ROS 2 is available
    if ros_enabled and cfg.get("use_ros", True):  # Only initialize ROS if it's enabled
        import rclpy
        rclpy.init()
        data_collector = ROSDataCollector(ros_enabled=True)
        data_listener = ROSDataListener(ros_enabled=True)
        # Run ROS 2 spin in a separate thread
        def spin_ros():
            rclpy.spin(data_listener)

        ros_thread = threading.Thread(target=spin_ros, daemon=True)
        ros_thread.start()
        print("ROS is enabled and initialized.")
    else:
        data_collector = None
        data_listener = None
        print("ROS is not enabled or unavailable.")

    id = 1
    while True:
        scene_dir = dataset_dir / f"{cfg.scene_name}_{id}"
        if not scene_dir.exists():
            break
        id += 1

    print(f"Collecting data for scene {cfg.scene_name} in dataset {cfg.dataset_name}...")
    print(f"Data will be saved at {scene_dir}")
    scene_dir.mkdir(parents=True, exist_ok=True)

    config = make_cfg(cfg)

    # create a simulator instance
    sim = habitat_sim.Simulator(config)

    print("Scene has been loaded")
    scene = sim.semantic_scene

    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
    )

    object_id_to_class_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
    # convert values to uint32
    object_id_to_class_id = {k: np.uint32(v) for k, v in object_id_to_class_id.items()}

    # save the scene information into the output folder
    save_scene_object_info(scene, scene_dir)

    # initialize the agent
    agent = sim.initialize_agent(cfg.default_agent)

    # Get all the objects
    obj_attr_mgr = sim.get_object_template_manager()

    obj_attr_mgr.load_configs(cfg.objects_path)

    id_handle_dict = register_templates_from_handles(obj_attr_mgr, cfg)

    # save this in cfg
    cfg.id_handle_dict = id_handle_dict

    handle_list = list(cfg.id_handle_dict.values())

    # the objects from predefined scene configs
    scene_objects = []

    if cfg.load_from_config:
        scene_objects = load_objects_from_config(sim, cfg)

    # Get all the placeable objects' bboxes
    all_bboxes_for_place = []

    for category in cfg.placable_categories:
        category_bbox_list = get_bounding_boxes_for_category(sim, category)
        all_bboxes_for_place.extend(category_bbox_list)

    
    if cfg.show_placable_categories:
        draw_bounding_boxes(sim, obj_attr_mgr, all_bboxes_for_place)

    # Set the initial state
    agent_state = habitat_sim.AgentState()

    sim.pathfinder.seed(cfg.data_cfg.seed)

    random_pt = sim.pathfinder.get_random_navigable_point()

    agent_state.position = random_pt

    agent.set_state(agent_state)

    # In actions LOOP!
    init_record_state = agent_state
    init_record_time = None
    actions_list = []

    # counter for help message display
    help_count = 0
    # counter for map display
    map_count = 0
    # counter for recording
    # NOTICE: Due to the init state of the agent, for one episode we only record one
    recording = False
    all_actions = []

    # navigation flag
    is_navigation = False
    nav_goal = None
    nav_path = None
    continuous_path_follower = None
    # global path from mapping sys
    previous_global_path = None

    # Initialize global list for all rigid objects
    all_rigid_objects = []
    all_rigid_objects.extend(scene_objects)

    # Get camera parameters and save
    fx, fy, cx, cy, width, height = get_camera_intrinsics(sim, "color_sensor")
    intrinsics_file_path = scene_dir / "camera_intrinsics.json"
    save_intrinsics(intrinsics_file_path, fx, fy, cx, cy, width, height)

    # Set frame rate
    target_fps = cfg.frame_rate
    frame_interval = 1 / target_fps

    # We ONLY calculate the topdown map once
    height = sim.pathfinder.get_bounds()[0][1]

    sim_topdown_map = sim.pathfinder.get_topdown_view(0.05, height)

    # Get SHrunk map of the topdown map
    shrunk_map = shrink_false_areas(sim_topdown_map, 2, 3)

    # Object sem id for grabbing
    grabbed_obj_sem_id = None


    last_time = time.time()

    while True:

        # control the frame rate
        elapsed_time = time.time() - last_time
        sleep_time = frame_interval - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Update last_time at the start of each loop iteration
        last_time = time.time()

        topdown_map = None
        sim.step_physics(frame_interval)

        # Filter invalid objects
        objects_to_remove = filter_rigid_objects(all_rigid_objects, height, 0.3)
        for obj in objects_to_remove:
            all_rigid_objects.remove(obj)
            sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)

        # Map handling
        if map_count % 2 == 1:
            last_point = agent.get_state().position
            agent_position = convert_points_to_topdown(sim.pathfinder, last_point, 0.05)
            all_rigid_objects_positions = [
                convert_points_to_topdown(sim.pathfinder, obj.translation, 0.05) for obj in all_rigid_objects
            ]

            # Visualize the map with the goal and path on the topdown map
            if nav_goal is not None:
                nav_goal_plot = convert_points_to_topdown(sim.pathfinder, nav_goal, 0.05)
                nav_path_plot = [
                    convert_points_to_topdown(sim.pathfinder, way_point, 0.05) for way_point in nav_path
                ]

                # Visualize the goal and path on the topdown map
                topdown_map = get_topdown_map_cv(
                    sim_topdown_map,
                    agent_position=agent_position,
                    all_rigid_objects_positions=all_rigid_objects_positions,
                    goal_position=nav_goal_plot,
                    nav_path=nav_path_plot  # Ensure path is passed correctly here
                )
            else:
                # Just display the top-down map without the goal or path
                topdown_map = get_topdown_map_cv(
                    sim_topdown_map,
                    agent_position=agent_position,
                    all_rigid_objects_positions=all_rigid_objects_positions,
                )

        # Capture the sensor state
        agent_state = sim.get_agent(0).get_state()
        sensor_pos = agent_state.sensor_states["color_sensor"].position
        sensor_quat = agent_state.sensor_states["color_sensor"].rotation

        # Navigation handling

        # FOR DEBUG, HERE commented for mannually control
        if continuous_path_follower and continuous_path_follower.progress < 1.0:

            # debug for local path developing
            # 显式的update
            continuous_path_follower.set_state(agent_state)

            pos, rot = continuous_path_follower.get_target_state()

            agent_state.position = pos
            agent_state.rotation = rot

            # Set the state of following
            sim.get_agent(0).set_state(agent_state)

        obs = sim.get_sensor_observations()
        display_obs(obs, help_count, topdown_map, recording)

        # Publish data via ROS if enabled
        if data_collector:
            rgb_img = np.array(obs["color_sensor"])
            depth_img = np.array(obs["depth_sensor"])
            semantic_img = np.array(obs["semantic_sensor"])
            semantic_img = np.vectorize(object_id_to_class_id.get)(semantic_img)
            fx, fy, cx, cy, width, height = get_camera_intrinsics(sim, "color_sensor")
            pose = get_agent_pose(agent)

            data_collector.publish_rgb(rgb_img)
            data_collector.publish_depth(depth_img)
            data_collector.publish_semantic(semantic_img)
            data_collector.publish_pose(pose)
            data_collector.publish_camera_info(fx, fy, cx, cy, width, height)

        # Actions based on ROS information
        if data_listener:
            # Check for the global path in the listener
            global_path = data_listener.get_latest_path()
            if global_path is not None and global_path != previous_global_path:
                print("Received updated global path: ", len(global_path), " points")
                is_navigation = True  # set navigation
                previous_global_path = global_path  # update the previous path

        if is_navigation:
            # Set Follower
            current_state = sim.get_agent(0).get_state()
            current_position = current_state.position
            current_rotation = current_state.rotation


            # 这里的逻辑需要进行改正，目前只要进入到online的path模式，
            # 就会一直在这个path中不会进入到按n手动的模式下的path follow
            if previous_global_path is not None:

                # switch to numpy
                # change Y to current Y
                pose_in_habitat = [
                    np.array([x, current_position[1], z])
                    for x, _, z in previous_global_path
                ]

                # Judge Path start point with current rotation
                # Ignore the points which are behind the current position, 

                print(f"current rotation: \t", current_rotation)
                print(f"current position: \t", current_position)
                print(f"first point in path: \t", pose_in_habitat[0])

                print(f"current  path length: ", len(pose_in_habitat))
                pose_in_habitat = filter_forward_path(current_position, current_rotation, pose_in_habitat)
                print(f"filtered path length: ", len(pose_in_habitat))

                # 若有上一次的 global path，就使用 online 路径
                nav_goal = pose_in_habitat[-1]
                nav_path = pose_in_habitat
            else:
                # Get The Random Path Here!!!
                # If not online, gen the path with the nav mesh
                if not sim.pathfinder.is_loaded:
                    print("Pathfinder not initialized, aborting.")

                goal, path_points = compute_random_nav_path(sim)

                nav_goal = goal
                nav_path = path_points

            # if we get the path, then initialize the follower, ready for navigation
            # waypoint_thresholdcontrol the distance from the final position to the goal
            # TODO: Magic number waypoint th
            continuous_path_follower = ContinuousPathFollower(
                sim, nav_path, sim.get_agent(0).get_state(), waypoint_threshold=0.1
            )

            # flip the navigation flag
            is_navigation = not is_navigation
            continue

        k, action = keyboard_control_fast()
        if k != -1:
            if action == "stop":
                cv2.destroyAllWindows()
                break

            if action == "record":
                recording = not recording
                if recording:
                    print("Recording started")
                    init_record_state = sim.get_agent(0).get_state()
                    init_record_time = time.time()
                    actions_list = []

                    # ROS2 bag
                    if cfg.record_rosbag:
                        rosbag_output_path = scene_dir / f"rosbag2_{int(time.time())}"
                        print(f"Start ROS bag recording: {rosbag_output_path}")
                        rosbag_process = start_rosbag_recording(rosbag_output_path)
                else:
                    print("Recording stopped")
                    all_actions.extend(actions_list)

                    # ROS2 bag 
                    if cfg.record_rosbag:
                        stop_rosbag_recording(rosbag_process)
                        rosbag_process = None
                continue

            if action == "map":
                map_count += 1
                continue

            if action == "add_object":
                print("Adding object to scene...")
                obj = sample_and_place(sim, handle_list, all_bboxes_for_place, shrunk_map)
                if obj:
                    all_rigid_objects.append(obj)
                    print("Object added and recorded.")
                    # logging the add_object action
                    manipulate_object_time = time.time()
                    action_info = get_object_action_info(obj, manipulate_object_time, action)
                    actions_list.append(action_info)

                else:
                    print("Failed to place object.")
                continue

            if action == "remove_object":
                if all_rigid_objects:
                    obj_to_remove = random.choice(all_rigid_objects)

                    # logging first
                    # logging the object action
                    manipulate_object_time = time.time()
                    action_info = get_object_action_info(obj_to_remove, manipulate_object_time, action)
                    actions_list.append(action_info)

                    # Then remove
                    obj_id = obj_to_remove.object_id
                    sim.get_rigid_object_manager().remove_object_by_id(obj_id)
                    all_rigid_objects.remove(obj_to_remove)
                    print(f"Removed object ID {obj_id} from scene, remaining {len(all_rigid_objects)} objects.")

                else:
                    print("No objects to remove.")

                continue

            if action == "place_in_view":
                # Filter bboxes in view
                bboxes_in_view = filter_bboxes_in_view(sim, all_bboxes_for_place)

                if not bboxes_in_view:
                    print("Error: No bounding boxes in view. Cannot place object.")
                    continue  # Skip placement if no bboxes are in view

                # Sample and place an object in view
                placed_object = sample_and_place(sim, handle_list, bboxes_in_view, shrunk_map)
                if placed_object:
                    all_rigid_objects.append(placed_object)
                    print("Object placed within camera view.")

                    # logging the object action
                    manipulate_object_time = time.time()
                    action_info = get_object_action_info(placed_object, manipulate_object_time, action)
                    actions_list.append(action_info)

                else:
                    print("Failed to place object in camera view.")

                continue

            if action == "grab":
                nearest_obj = find_nearest_obj(sim, all_rigid_objects)

                # remember this object semantic id
                obj_id = nearest_obj.object_id
                grabbed_obj_sem_id = nearest_obj.semantic_id
                obj_handle = id_handle_dict[grabbed_obj_sem_id]

                # remove the grabbed object
                sim.get_rigid_object_manager().remove_object_by_id(obj_id)
                all_rigid_objects.remove(nearest_obj)
                print(f"Object ID: {obj_id}, {obj_handle}  object is being grabbed, press 'r' to release to nearest bbox")

                continue

            if action == "release":
                
                if grabbed_obj_sem_id is None:
                    print("No grabbed object, Please grab object first")
                    continue

                nearest_bbox = find_nearest_bbox(sim, all_bboxes_for_place)

                # get grabbed object handle
                grabbed_handle = cfg.id_handle_dict[grabbed_obj_sem_id]

                placed_object = place_object_to_bbox(sim, grabbed_handle, nearest_bbox, shrunk_map)

                if placed_object:
                    all_rigid_objects.append(placed_object)
                    grabbed_obj_sem_id = None
                    print(f"Object {grabbed_handle} released.")
                else:
                    print("Failed to release the object in the nearest bbox, try again.")

                continue

            if action == "save_config":
                config_file_path = scene_dir / "scene_config.json"
                save_config(cfg, all_rigid_objects, config_file_path)

                continue

            if action == "navigation":
                is_navigation = not is_navigation
                continue

            # Process movement actions
            action_time = time.time()
            sim.step(action)
            print(f"Action timestamp: {action_time}, action: {action}")
            if recording:
                actions_list.append({
                    "action": action,
                    "timestamp": time.time()
                })
        else:
            # Record the "no action" state with a timestamp
            if recording:
                no_action_time = time.time()
                actions_list.append({
                    "action": None,
                    "timestamp": no_action_time
                })

    # Print all actions line by line, filtering out actions with 'None'
    for action in all_actions:
        if action["action"] is not None:
            print(action)

    if len(all_actions) > 0:
        # restart the sim
        sim = habitat_sim.Simulator(config)
        # Get all the objects
        obj_attr_mgr = sim.get_object_template_manager()

        obj_attr_mgr.load_configs(cfg.objects_path)

        register_templates_from_handles(obj_attr_mgr, cfg)

        replay_and_save(sim, cfg, scene_dir, all_actions, init_record_state, init_record_time)

    sim.close()

if __name__ == "__main__":

    main()