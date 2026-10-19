from typing import Optional

import gym
from gym.core import Env
import numpy as np
import os
import time
import copy

from tdw.replicant.arm import Arm
from tdw.tdw_utils import TDWUtils

from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from transport_challenge_multi_agent.paths import TDW_MAT_DIRECTORY
from collections import Counter
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.image_frequency import ImageFrequency
from tdw.output_data import OutputData, SegmentationColors, FieldOfView, Images
from tdw.scene_data.scene_bounds import SceneBounds
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.add_ons.object_manager import ObjectManager
from PIL import Image

import json
from functools import partial
import signal
from tenacity import retry, wait_fixed, retry_if_exception_type
import math

from target_resolver import resolve_forced_coop_targets
import action_protocol
from controller_lifecycle import shutdown_controller
from episode_limits import (
    optional_positive_int, positive_finite, resolve_episode_termination,
)



def _count_communication_chars(text):
    if text is None:
        return 0
    return len(str(text))

class TimeoutException(Exception):
    pass

class AnySpace(gym.Space):
    """Permissive space for variable symbolic feedback fields."""

    def __init__(self):
        super().__init__(shape=None, dtype=None)

    def sample(self):
        return None

    def contains(self, x):
        return True


class ActionEnvelopeSpace(gym.Space):
    """Gym declaration backed by the formal TDW Benchmark action schema."""

    def __init__(self, max_message_chars):
        super().__init__(shape=None, dtype=None)
        self.max_message_chars = int(max_message_chars)

    def sample(self):
        return {'physical_action': {'type': 7}, 'message': None}

    def contains(self, value):
        if not isinstance(value, dict) or set(value) != {
            'physical_action',
            'message',
        }:
            return False
        message = value.get('message')
        # Oversized strings are valid inputs and are truncated by the formal
        # communication protocol before delivery.
        if message is not None and not isinstance(message, str):
            return False
        physical_action = value.get('physical_action')
        if physical_action is None:
            return True
        _, reason = action_protocol.validate_physical_action_payload(
            physical_action
        )
        return reason is None

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution exceeded the timeout limit")

@retry(wait=wait_fixed(5), retry=retry_if_exception_type(TimeoutException))  # wait 5 seconds between retries
def might_fail_launch(launch, port = None):
    if port is not None:
        print("kill failure launch ...", f"ps ux | grep TDW.x86_64\ -port\ {port} | awk {{'print $2'}} | xargs kill")
        os.system(f"ps ux | grep TDW.x86_64\ -port\ {port} | awk {{'print $2'}} | xargs kill")
    # A just-closed ZMQ listener can need a short scheduling interval before a
    # replacement Controller binds. Retry only address-in-use failures; preserve
    # all other launch errors unchanged.
    for bind_attempt in range(1, 4):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)
        try:
            print("Trying to launch tdw ...")
            return launch()
        except Exception as error:
            address_in_use = (
                "Address already in use" in str(error)
                or "EADDRINUSE" in str(error)
            )
            if not address_in_use or bind_attempt >= 3:
                raise
            print(
                "TDW port {} is still in use; retrying bind ({}/3) ...".format(
                    port, bind_attempt + 1
                )
            )
            time.sleep(1)
        finally:
            signal.alarm(0)

class TDW(Env):
    def __init__(self, port = 1071, number_of_agents = 1, demo=False, rank=0, num_scenes = 0, train=False, \
                        screen_size = 512, exp = False, launch_build=True, gt_occupancy = False, gt_mask = True, enable_collision_detection = False, save_dir = TDW_MAT_DIRECTORY / "results", max_frames = None, data_prefix = TDW_MAT_DIRECTORY / "dataset" / "dataset_test",
                        communication_mode='exclusive_action', coop_action_interface='implicit_pairing',
                        communication_max_message_chars=12000,
                        communication_send_chars_per_second=125000.0,
                        minimum_step_frames=1, max_embodied_execution_time=20.0,
                        max_steps=None,
                        save_top_down_images=False,
                        check_version=True):
        self.messages = None
        self.data_prefix = data_prefix
        self.replicant_colors = None
        self.replicant_ids = None
        self.names_mapping = None
        self.rooms_name = None
        self.action_buffer = None
        self.scene_bounds = None
        self.goal_description = None
        self.object_manager = None
        self.occupancy_map = None
        self.gt_mask = gt_mask
        self.satisfied = None
        self.count = 0
        self.reach_threshold = 2
        self.number_of_agents = number_of_agents
        self.seed = None
        self.num_step = 0
        self.reward = 0
        self.done = False
        self.scene_info = None
        self.exp = exp
        self.success = False
        self.num_frames = 0
        self.data_id = rank     
        self.train = train
        self.port = port
        self.gt_occupancy = gt_occupancy
        self.screen_size = screen_size
        self.launch_build = launch_build
        self.check_version = bool(check_version)
        self.enable_collision_detection = enable_collision_detection
        self.save_top_down_images = bool(save_top_down_images)
        self.controller = None
        # ==================== Benchmark Protocol: Communication and Time ====================
        # Keep TDW's asynchronous physical frames while exposing the same paper-level
        # communication/action interface and embodied-time semantics as C-WAH.
        if communication_mode != "exclusive_action":
            raise ValueError('Unsupported communication_mode: {}'.format(communication_mode))
        if coop_action_interface != "implicit_pairing":
            raise ValueError('Unsupported coop_action_interface: {}'.format(coop_action_interface))
        if int(communication_max_message_chars) <= 0:
            raise ValueError('communication_max_message_chars must be positive')
        if not math.isfinite(float(communication_send_chars_per_second)) or float(communication_send_chars_per_second) <= 0:
            raise ValueError('communication_send_chars_per_second must be positive')

        if int(minimum_step_frames) <= 0:
            raise ValueError('minimum_step_frames must be positive')
        # Episode time is primary; cumulative frame/step caps are opt-in only.
        if max_embodied_execution_time is not None:
            positive_finite(max_embodied_execution_time)
        optional_positive_int(max_steps)
        optional_positive_int(max_frames)
        self.communication_mode = communication_mode
        self.coop_action_interface = coop_action_interface
        self.communication_max_message_chars = int(communication_max_message_chars)
        # One 500-character reference payload per 250-Hz simulator frame.
        self.communication_send_chars_per_second = float(communication_send_chars_per_second)
        self.communication_count_unit = 'unicode_codepoint'
        self.minimum_step_frames = int(minimum_step_frames)
        self.max_embodied_execution_time = (
            None if max_embodied_execution_time is None
            else float(max_embodied_execution_time)
        )
        self.max_steps = None if max_steps is None else int(max_steps)
        self.elapsed_embodied_execution_time = 0.0
        self.termination_reason = None
        self.physical_action_frame_rate = 250.0
        # ==================== End Benchmark Protocol: Communication and Time ====================
        self.last_action_status = None
        self.last_invalid_reasons = None
        self.last_invalid_events = []
        self.active_coop_carrying = None
        self.coop_action_events = []
        self.last_coop_action_events = []
        self.last_coop_feedback_events = []
        self.coop_distance_threshold = self.reach_threshold
        self.coop_hard_distance_threshold = 4.0
        self.forced_coop_info = None
        self.active_target_object_ids = None
        self.coop_target_object_ids = None
        self.ordinary_target_object_ids = None
        self.object_coop_properties = {
            i: {} for i in range(self.number_of_agents)
        }
        self.class_coop_properties = {
            i: {} for i in range(self.number_of_agents)
        }
        self.last_grasp_property_feedback = {
            i: [] for i in range(self.number_of_agents)
        }
        self.last_action_feedback = {
            i: self.default_action_feedback(i) for i in range(self.number_of_agents)
        }
        self.last_message_feedback = {
            i: self.default_message_feedback(i) for i in range(self.number_of_agents)
        }
        self.pending_action_transactions = {
            i: None for i in range(self.number_of_agents)
        }
        self.action_space_single = ActionEnvelopeSpace(
            self.communication_max_message_chars
        )
        # Observation values contain dynamic runtime ids, variable-length lists,
        # optional None fields, images, and symbolic feedback. The strict public
        # information boundary is enforced when observations are constructed and
        # handed to methods, not by an inaccurate fixed-size Gym schema.
        self.observation_space_single = AnySpace()

        self.observation_space = gym.spaces.Dict({
            str(i): self.observation_space_single for i in range(self.number_of_agents)
        })

        self.action_space = gym.spaces.Dict({
            str(i): self.action_space_single for i in range(self.number_of_agents)
        })
        self.max_frame = optional_positive_int(max_frames)
        os.makedirs(save_dir, exist_ok=True)
        self.f = open(os.path.join(save_dir, f'action{port}.log'), 'w')
        self.action_list = []
                    
        self.segmentation_colors = {}
        self.object_names = {}
        self.object_ids = {}
        self.object_categories = {}
        self.target_object_ids = []
        self.container_ids = []
        self.goal_position_id = None # The place to put the object
        self.fov = 0
        self.save_dir = save_dir
    
    def obs_filter(self, obs):
        if self.gt_mask:
            return obs
        else:
            new_obs = copy.deepcopy(obs)
            for agent in obs:
                new_obs[agent]['seg_mask'] = np.zeros_like(new_obs[agent]['seg_mask'])
                new_obs[agent]['visible_objects'] = []
                while len(new_obs[agent]['visible_objects']) < 50:
                    new_obs[agent]['visible_objects'].append({
                        'id': None,
                        'type': None,
                        'seg_color': None,
                        'name': None,
                    })
            return new_obs

    def get_object_type(self, id):
        # ==================== Coop: Public Active-Target Semantics ====================
        # Once the runtime resolver is available, only Benchmark-active targets
        # are exposed as target objects. This keeps filtered/excluded targets from
        # entering ordinary agents' task memory without exposing coop labels.
        public_target_ids = (
            self.active_target_object_ids
            if self.active_target_object_ids is not None
            else self.target_object_ids
        )
        if id in public_target_ids:
            return 0 # target object
        # ==================== End Coop: Public Active-Target Semantics ====================
        if id in self.container_ids:
            return 1 # container
        if self.object_categories[id] == 'bed':
            return 2 # goal position
        # 3: agent
        # 4: obstacle
        return 5 # unrelated object

    def get_with_character_mask(self, agent_id, character_object_ids):
        color_set = [self.segmentation_colors[id] for id in character_object_ids if id in self.segmentation_colors] + [self.replicant_colors[id] for id in character_object_ids if id in self.replicant_colors]
        curr_with_seg = np.zeros_like(self.obs[str(agent_id)]['seg_mask'])
        curr_seg_flag = np.zeros((self.screen_size, self.screen_size), dtype = bool)
        for i in range(len(color_set)):
            color_pos = (self.obs[str(agent_id)]['seg_mask'] == np.array(color_set[i])).all(axis=2)
            curr_seg_flag = np.logical_or(curr_seg_flag, color_pos)
            curr_with_seg[color_pos] = color_set[i]
        return curr_with_seg, curr_seg_flag
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        output_dir: Optional[str] = None
    ):
        """
        reset the environment
        input:
            data_id: reset based on the data_id
        """
        # Changes it to always, since in each step, we need to get the image
        self._shutdown_controller()
        # download_asset_bundles()
        launch = partial(
            TransportChallenge,
            port=self.port,
            check_version=self.check_version,
            launch_build=self.launch_build,
            screen_width=self.screen_size,
            screen_height=self.screen_size,
            image_frequency=ImageFrequency.always,
            png=True,
            image_passes=None,
            enable_collision_detection=self.enable_collision_detection,
            logger_dir=output_dir,
            enable_top_down_camera=self.save_top_down_images,
        )
        self.controller = might_fail_launch(launch, port=self.port)
        print("Controller connected")
        self.success = False
        self.messages = [None for _ in range(self.number_of_agents)]
        self.last_action_status = ['not_started' for _ in range(self.number_of_agents)]
        self.last_invalid_reasons = [None for _ in range(self.number_of_agents)]
        self.last_invalid_events = []
        self.active_coop_carrying = None
        self.coop_action_events = []
        self.last_coop_action_events = []
        self.last_coop_feedback_events = []
        self.object_coop_properties = {
            i: {} for i in range(self.number_of_agents)
        }
        self.class_coop_properties = {
            i: {} for i in range(self.number_of_agents)
        }
        self.last_grasp_property_feedback = {
            i: [] for i in range(self.number_of_agents)
        }
        self.last_action_feedback = {
            i: self.default_action_feedback(i) for i in range(self.number_of_agents)
        }
        self.last_message_feedback = {
            i: self.default_message_feedback(i) for i in range(self.number_of_agents)
        }
        self.pending_action_transactions = {
            i: None for i in range(self.number_of_agents)
        }
        self.coop_hard_distance_threshold = 4.0
        self.elapsed_embodied_execution_time = 0.0
        self.termination_reason = None
        self.reward = 0
        scene_info = options
        print(scene_info)
        self.satisfied = {}
        if output_dir is not None: self.save_dir = output_dir
        if scene_info is not None:
            scene = scene_info['scene']
            layout = scene_info['layout']
            if 'task' in scene_info:
                task = scene_info['task']
            else:
                task = None
        else: raise ValueError("No scene info assigned!")
        super().reset(seed=seed)
        self.seed = np.random.RandomState(seed)
        self.scene_info = scene_info
        
        # Now the scene is fixed, so num_containers and num_target_objects are not used anymore in new settings
        self.controller.start_floorplan_trial(scene=scene, layout=layout, replicants=self.number_of_agents, num_containers=4, num_target_objects=10,
                                   random_seed=seed, task = task, data_prefix = self.data_prefix)

        # Add a gt occupancy map. In the standard setting, we don't need this
        if self.gt_occupancy:
            self.occupancy_map = OccupancyMap()
            self.controller.add_ons.append(self.occupancy_map)
            self.occupancy_map.generate(cell_size=0.125, once = False)
        self.controller.communicate({"$type": "set_floorplan_roof",
                          "show": False})

        # Bright case   
        self.controller.communicate({"$type": "add_hdri_skybox", "name": "sky_white", "url": "https://tdw-public.s3.amazonaws.com/hdri_skyboxes/linux/2019.1/sky_white", "exposure": 2, "initial_skybox_rotation": 0, "sun_elevation": 90, "sun_initial_angle": 0, "sun_intensity": 1.25})
            
        # Set the field of view of the agent.
        for replicant_id in self.controller.replicants:
            self.controller.communicate({"$type": "set_field_of_view",
                            "avatar_id" : str(replicant_id), "field_of_view" : 90})
        self.fov = 90
        
        # Add a object manager for object position
        self.object_manager = ObjectManager()
        self.controller.add_ons.append(self.object_manager)

        data = self.controller.communicate({"$type": "send_segmentation_colors",
                          "show": False,
                          "frequency": "once"})
        
        # Show the occupancy map. In the standard setting, we don't need this
        if self.gt_occupancy:            
            self.occupancy_map.show()
            print(self.occupancy_map.occupancy_map)
            h, w = self.occupancy_map.occupancy_map.shape
            print(self.occupancy_map.occupancy_map.shape)

        # Make name easier to read
        names_mapping_path = TDW_MAT_DIRECTORY / "dataset" / "name_map.json"
        with open(names_mapping_path, 'r') as f: self.names_mapping = json.load(f)

        self.segmentation_colors = {}
        self.object_names = {}
        self.object_ids = {}
        self.object_categories = {}
        self.target_object_ids = self.controller.state.target_object_ids
        self.container_ids = self.controller.state.container_ids
        self.replicant_ids = [self.controller.replicants[i].static.replicant_id for i in range(self.number_of_agents)]
        
        for i in range(len(data) - 1):
            r_id = OutputData.get_data_type_id(data[i])
            if r_id == "segm":
                segm = SegmentationColors(data[i])
                for j in range(segm.get_num()):
                    object_id = segm.get_object_id(j)
                    self.segmentation_colors[object_id] = segm.get_object_color(j)
                    self.object_names[object_id] = segm.get_object_name(j).lower()
                    if self.object_names[object_id] in self.names_mapping:
                        self.object_names[object_id] = self.names_mapping[self.object_names[object_id]]
                    self.object_categories[object_id] = segm.get_object_category(j)
                    if self.object_categories[object_id] == 'bed':
                        self.goal_position_id = object_id
        
        self.replicant_colors ={i: self.controller.replicants[i].static.segmentation_color for i in range(self.number_of_agents)}

        self.containment_all = {}
        
        # check colors are different:
        for x in self.segmentation_colors.keys():
            for y in self.segmentation_colors.keys():
                if x != y: assert (self.segmentation_colors[x] != self.segmentation_colors[y]).any()

        self.num_step = 0
        self.num_frames = 0

        # ==================== Coop: Runtime Target Resolver ====================
        # Dataset constraints are class-level; concrete TDW object ids are known
        # only after reset and segmentation/name mapping are available.
        contained_target_ids = set()
        for contained_objects in self.controller.state.containment.values():
            contained_target_ids.update(contained_objects)
        for contained_objects in self.containment_all.values():
            contained_target_ids.update(contained_objects)
        self.forced_coop_info = resolve_forced_coop_targets(
            self.scene_info, self.target_object_ids, self.object_names,
            excluded_target_ids=contained_target_ids)
        self.active_target_object_ids = self.forced_coop_info['active_target_ids']
        self.coop_target_object_ids = self.forced_coop_info['coop_target_ids']
        self.ordinary_target_object_ids = self.forced_coop_info['ordinary_target_ids']
        self.goal_description = self.forced_coop_info['active_goal_description']
        # ==================== End Coop: Runtime Target Resolver ====================

        room_type_path = TDW_MAT_DIRECTORY / "dataset" / "room_types.json"
        with open(room_type_path, 'r') as f: room_types = json.load(f)
        
        self.rooms_name = {}
        #now return <room_type> (id) for each room.        
        if type(layout) == str: now_layout = int(layout[0])
        else: now_layout = int(layout)
        for i, rooms_name in enumerate(room_types[scene[0]][now_layout]):
            if rooms_name not in ['Kitchen', 'Livingroom', 'Bedroom', 'Office']:
                the_name = None
            else:
                the_name = f'<{rooms_name}> ({1000 * (i + 1)})'
            self.rooms_name[i] = the_name

        self.done = False
        self.action_buffer = [[] for _ in range(self.number_of_agents)]

        resp = self.controller.communicate([{"$type": "send_scene_regions"}])
        self.scene_bounds = SceneBounds(resp=resp)
        self.all_rooms = [self.rooms_name[i] for i in range(len(self.rooms_name)) if self.rooms_name[i] is not None]
        info = {
            'goal_description': self.goal_description,
            'rooms_name': self.all_rooms,
            'agent_colors': self.replicant_colors,
            'forced_coop': self.forced_coop_info,
        }
        env_api = [{
            'belongs_to_which_room': self.belongs_to_which_room,
            'center_of_room': self.center_of_room,
            'check_pos_in_room': self.check_pos_in_room,
            'get_room_distance': self.get_room_distance,
            'get_id_from_mask': partial(self.get_id_from_mask, agent_id=i),
            'get_with_character_mask': partial(self.get_with_character_mask, agent_id=i),
        } for i in range(self.number_of_agents)]
        self.obs = self.get_obs()
        self.initial_goal_progress = self.check_goal()
        info['initial_goal_progress'] = list(self.initial_goal_progress)
        info['max_embodied_execution_time'] = self.max_embodied_execution_time
        return self.obs_filter(self.obs), info, env_api

    def pos_to_2d_box_distance(self, px, py, rx1, ry1, rx2, ry2):
        if px < rx1:
            if py < ry1:
                return ((px - rx1) ** 2 + (py - ry1) ** 2) ** 0.5
            elif py > ry2:
                return ((px - rx1) ** 2 + (py - ry2) ** 2) ** 0.5
            else:
                return rx1 - px
        elif px > rx2:
            if py < ry1:
                return ((px - rx2) ** 2 + (py - ry1) ** 2) ** 0.5
            elif py > ry2:
                return ((px - rx2) ** 2 + (py - ry2) ** 2) ** 0.5
            else:
                return px - rx2
        else:
            if py < ry1:
                return ry1 - py
            elif py > ry2:
                return py - ry2
            else:
                return 0
    
    def belongs_to_which_room(self, pos):
        min_dis = 100000
        room = None
        for i, region in enumerate(self.scene_bounds.regions):
            distance = self.pos_to_2d_box_distance(pos[0], pos[2], region.x_min, region.z_min, region.x_max, region.z_max)
            if distance < min_dis and self.rooms_name[i] is not None:
                min_dis = distance
                room = self.rooms_name[i]
        return room
    
    def get_room_distance(self, pos):
        min_dis = 100000
        room = None
        for i, region in enumerate(self.scene_bounds.regions):
            distance = self.pos_to_2d_box_distance(pos[0], pos[2], region.x_min, region.z_min, region.x_max, region.z_max)
            if distance < min_dis and self.rooms_name[i] is not None:
                min_dis = distance
                room = self.rooms_name[i]
        return min_dis
    
    def center_of_room(self, room):
        assert type(room) == str
        for index, name in self.rooms_name.items():
            if name == room:
                room = index
        return self.scene_bounds.regions[room].center
    
    def check_pos_in_room(self, pos):
        if len(pos) == 3:
            for region in self.scene_bounds.regions:
                if region.is_inside(pos[0], pos[2]):
                    return True
        elif len(pos) == 2:
            for region in self.scene_bounds.regions:
                if region.is_inside(pos[0], pos[1]):
                    return True
        return False

    def map_status(self, status, buffer_len = 0):
        if status == ActionStatus.ongoing or buffer_len > 0:
            return 0
        elif status == ActionStatus.success or status == ActionStatus.still_dropping:
            return 1
        else: return 2

    def get_2d_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1[[0, 2]]) - np.array(pos2[[0, 2]]))

    def check_goal(self):
        r'''
        Check if the goal is achieved
        return: count, total, done
        '''
        place_pos = self.object_manager.transforms[self.goal_position_id].position
        target_ids = self.active_target_object_ids if self.active_target_object_ids is not None else self.target_object_ids
        # Forced-coop transport treats a target as completed once it has been
        # released in the goal area. Keep ids cumulative to avoid physical
        # settling/jitter after drop removing an already completed subgoal.
        cumulative_satisfied = dict(self.satisfied or {})
        for object_id in target_ids:
            if object_id in cumulative_satisfied:
                continue
            pos = self.object_manager.transforms[object_id].position
            room = self.belongs_to_which_room(pos)
            in_goal_area = self.get_2d_distance(pos, place_pos) < 3 and room is not None and 'Bedroom' in room
            if in_goal_area and not self.object_held_by_any_agent(object_id):
                cumulative_satisfied[object_id] = True
        self.satisfied = cumulative_satisfied
        count = sum(1 for object_id in target_ids if object_id in self.satisfied)
        return count, len(target_ids), count == len(target_ids)

    def get_id_from_mask(self, agent_id, mask, name = None):
        r'''
        Get the object id from the mask
        '''
        seg_with_mask = (self.obs[str(agent_id)]['seg_mask'] * np.expand_dims(mask, axis = -1)).reshape(-1, 3)
        seg_with_mask = [tuple(x) for x in seg_with_mask]
        seg_counter = Counter(seg_with_mask)
        
        for seg in seg_counter:
            if seg == (0, 0, 0): continue
            if seg_counter[seg] / np.sum(mask) > 0.5:
                for i in range(len(self.obs[str(agent_id)]['visible_objects'])):
                    if self.obs[str(agent_id)]['visible_objects'][i]['seg_color'] == seg:
                        return self.obs[str(agent_id)]['visible_objects'][i]
        return {
                    'id': None,
                    'type': None,
                    'seg_color': None,
                    'name': None,
                }

    def get_obs(self):
        for x in self.controller.state.containment.keys():
            if x not in self.containment_all.keys():
                self.containment_all[x] = []
            for y in self.controller.state.containment[x]:
                if y not in self.containment_all[x]:
                    self.containment_all[x].append(y)
        obs = {str(i): {} for i in range(self.number_of_agents)}
        containment_info_get = {str(i): [str(i)] for i in range(self.number_of_agents)}
        for replicant_id in self.controller.replicants:
            id = str(replicant_id)
            obs[id]['visible_objects'] = []
            if 'img' in self.controller.replicants[replicant_id].dynamic.images.keys():
                obs[id]['rgb'] = np.array(self.controller.replicants[replicant_id].dynamic.get_pil_image('img')).transpose(2, 0, 1)
                obs[id]['seg_mask'] = np.array(self.controller.replicants[replicant_id].dynamic.get_pil_image('id'))
                colors = Counter(self.controller.replicants[replicant_id].dynamic.get_pil_image('id').getdata())
                for object_id in self.segmentation_colors:
                    segmentation_color = tuple(self.segmentation_colors[object_id])
                    object_name = self.object_names[object_id]
                    if segmentation_color in colors:
                        obs[id]['visible_objects'].append({
                            'id': object_id,
                            'type': self.get_object_type(object_id),
                            'seg_color': segmentation_color,
                            'name': object_name,
                        })
                for agent_id in self.replicant_colors:
                    segmentation_color = tuple(self.replicant_colors[agent_id])
                    if segmentation_color in colors:
                        obs[id]['visible_objects'].append({
                            'id': agent_id,
                            'type': 3,
                            'seg_color': segmentation_color,
                            'name': 'agent',
                        })
                        if str(agent_id) not in containment_info_get[id]: containment_info_get[id].append(str(agent_id))
                        
                obs[id]['depth'] = np.flip(np.array(TDWUtils.get_depth_values(self.controller.replicants[replicant_id].dynamic.get_pil_image('depth'),
                        width = self.screen_size,
                        height = self.screen_size)), 0)
                obs[id]['camera_matrix'] = np.array(self.controller.replicants[replicant_id].dynamic.camera_matrix).reshape((4, 4))
            else:
                assert -1, "No image received"
            while len(obs[id]['visible_objects']) < 50:
                obs[id]['visible_objects'].append({
                    'id': None,
                    'type': None,
                    'seg_color': None,
                    'name': None,
                })
            x, y, z = self.controller.replicants[replicant_id].dynamic.transform.position
            fx, fy, fz = self.controller.replicants[replicant_id].dynamic.transform.forward
            obs[id]['agent'] = [x, y, z, fx, fy, fz]
            held_objects = list(self.controller.state.replicants[replicant_id].values())
            obs[id]['held_objects'] = []
            for hand in range(2):
                if held_objects[hand] is None:
                    obs[id]['held_objects'].append({
                        'id': None,
                        'type': None,
                        'name': None,
                        'contained': [None, None, None],
                        'contained_name': [None, None, None],
                    })
                elif self.get_object_type(held_objects[hand]) == 0:
                    obs[id]['held_objects'].append({
                        'id': held_objects[hand],
                        'type': 0,
                        'name': self.object_names[held_objects[hand]],
                        'contained': [None, None, None],
                        'contained_name': [None, None, None],
                    })
                else:
                    if held_objects[hand] in self.containment_all.keys():
                        contained_obj = [x for x in self.containment_all[held_objects[hand]] if x not in held_objects and x in self.target_object_ids]
                        obs[id]['held_objects'].append({
                            'id': held_objects[hand],
                            'type': 1,
                            'name': self.object_names[held_objects[hand]],
                            'contained': contained_obj + [None] * (3 - len(contained_obj)),
                            'contained_name': [self.object_names[object_id] for object_id in contained_obj] + [None] * (3 - len(contained_obj)),
                        })
                    else:
                        obs[id]['held_objects'].append({
                            'id': held_objects[hand],
                            'type': 1,
                            'name': self.object_names[held_objects[hand]],
                            'contained': [None] * 3,
                            'contained_name': [None] * 3,
                        })
                        
            if len(containment_info_get[id]) == 2:
                oppo_held_objects = list(self.controller.state.replicants[1 - replicant_id].values())
            else:
                oppo_held_objects = [None, None]
            obs[id]['oppo_held_objects'] = []
            for hand in range(2):
                if oppo_held_objects[hand] is None:
                    obs[id]['oppo_held_objects'].append({
                        'id': None,
                        'type': None,
                        'name': None,
                        'contained': [None, None, None],
                        'contained_name': [None, None, None]
                    })
                elif self.get_object_type(oppo_held_objects[hand]) == 0:
                    obs[id]['oppo_held_objects'].append({
                        'id': oppo_held_objects[hand],
                        'type': 0,
                        'name': self.object_names[oppo_held_objects[hand]],
                        'contained': [None, None, None],
                        'contained_name': [None, None, None],
                    })
                else:
                    if oppo_held_objects[hand] in self.containment_all.keys(): 
                        contained_obj = [x for x in self.containment_all[oppo_held_objects[hand]] if x not in oppo_held_objects and x in self.target_object_ids]
                        obs[id]['oppo_held_objects'].append({
                            'id': oppo_held_objects[hand],
                            'type': 1,
                            'name': self.object_names[oppo_held_objects[hand]],
                            'contained': contained_obj + [None] * (3 - len(contained_obj)),
                            'contained_name': [self.object_names[object_id] for object_id in contained_obj] + [None] * (3 - len(contained_obj)),
                        })
                    else:
                       obs[id]['oppo_held_objects'].append({
                            'id': oppo_held_objects[hand],
                            'type': 1,
                            'name': self.object_names[oppo_held_objects[hand]],
                            'contained': [None] * 3,
                            'contained_name': [None] * 3,
                        })
            obs[id]['FOV'] = self.fov
            if (
                self.last_action_status is not None
                and self.last_action_status[replicant_id] in ['wait', 'coop_waiting', 'conflict_waiting']
            ):
                obs[id]['status'] = 1
            else:
                obs[id]['status'] = self.map_status(self.controller.replicants[replicant_id].action.status, len(self.action_buffer[replicant_id]))
            obs[id]['messages'] = [None, None]
            obs[id]['valid'] = self.last_invalid_reasons[replicant_id] is None if self.last_invalid_reasons is not None else True
            obs[id]['action_status'] = self.last_action_status[replicant_id] if self.last_action_status is not None else 'not_started'
            obs[id]['action_feedback'] = copy.deepcopy(
                self.last_action_feedback.get(replicant_id, self.default_action_feedback(replicant_id))
            )
            obs[id]['message_feedback'] = copy.deepcopy(
                self.last_message_feedback.get(replicant_id, self.default_message_feedback(replicant_id))
            )
            obs[id]['coop_action_feedback'] = copy.deepcopy(self.last_coop_action_events)
            obs[id]['coop_info'] = self.get_coop_info(replicant_id, self.last_coop_feedback_events)
            object_properties, class_properties = self.get_public_coop_properties(
                replicant_id,
                obs[id]['visible_objects'],
            )
            obs[id]['object_coop_properties'] = object_properties
            obs[id]['class_coop_properties'] = class_properties
            obs[id]['grasp_property_feedback'] = copy.deepcopy(
                self.last_grasp_property_feedback.get(replicant_id, [])
            )
            obs[id]['current_frames'] = self.num_frames
        return obs

    def get_coop_info(self, replicant_id, feedback_events=None):
        if self.active_coop_carrying is None:
            return {'active': False}
        feedback_events = feedback_events or []
        holder_id = self.active_coop_carrying['holder_id']
        supporter_id = self.active_coop_carrying['supporter_id']
        holder_pos = np.array(self.controller.replicants[holder_id].dynamic.transform.position)
        supporter_pos = np.array(self.controller.replicants[supporter_id].dynamic.transform.position)
        distance = self.get_2d_distance(holder_pos, supporter_pos)
        if distance > self.coop_hard_distance_threshold:
            distance_level = 'hard'
        elif distance > self.coop_distance_threshold:
            distance_level = 'soft'
        else:
            distance_level = 'ok'
        feedback_reason = None
        feedback_event = None
        for event in feedback_events:
            if isinstance(event, dict) and event.get('reason') is not None:
                feedback_reason = event.get('reason')
                feedback_event = copy.deepcopy(event)
                break
        return {
            'active': True,
            'object_id': self.active_coop_carrying['object_id'],
            'holder_id': holder_id,
            'supporter_id': supporter_id,
            'self_role': 'holder' if replicant_id == holder_id else 'supporter',
            'holder_pos': holder_pos.tolist(),
            'supporter_pos': supporter_pos.tolist(),
            'distance': float(distance),
            'soft_threshold': float(self.coop_distance_threshold),
            'hard_threshold': float(self.coop_hard_distance_threshold),
            'distance_level': distance_level,
            'distance_measurement_phase': 'post_action',
            'distance_measurement_env_step': int(self.num_step),
            'distance_measurement_frame': int(self.num_frames),
            'pre_action_feedback': feedback_reason,
            'pre_action_feedback_event': feedback_event,
        }

    def get_info(self):
        #todo: add info needed
        return {}

    def add_name(self, inst):
        if type(inst) == int and inst in self.object_names:
            return f'{inst}_{self.object_names[inst]}'
        else:
            if type(inst) == dict:
                return {self.add_name(key): self.add_name(value) for key, value in inst.items()}
            elif type(inst) == list:
                return [self.add_name(item) for item in inst]
            else: raise NotImplementedError
    
    def add_name_and_empty(self, inst):
        for x in self.container_ids:
            if x not in inst:
                inst[x] = []
        return self.add_name(inst)

    # ==================== Coop: Ordinary Action Checks ====================
    # These checks keep the original TDW action dispatch readable while adding
    # benchmark-level feasibility reasons before Replicant execution.
    def agent_holding_anything(self, replicant_id):
        return any(obj is not None for obj in self.controller.state.replicants[replicant_id].values())

    def get_agent_held_objects(self, replicant_id):
        return [obj for obj in self.controller.state.replicants[replicant_id].values() if obj is not None]

    def get_holding_arm(self, replicant_id, object_id=None):
        for arm, held_object in self.controller.state.replicants[replicant_id].items():
            if held_object is not None and (object_id is None or held_object == object_id):
                return arm
        return None

    def object_held_by_any_agent(self, object_id):
        for replicant_id in self.controller.replicants:
            if object_id in self.controller.state.replicants[replicant_id].values():
                return True
        return False

    def object_inside_container(self, object_id, include_history=True):
        for contained_objects in self.controller.state.containment.values():
            if object_id in contained_objects:
                return True
        if not include_history:
            return False
        for contained_objects in self.containment_all.values():
            if object_id in contained_objects:
                return True
        return False

    def agent_distance_to_object(self, replicant_id, object_id):
        agent_pos = self.controller.replicants[replicant_id].dynamic.transform.position
        object_pos = self.object_manager.transforms[int(object_id)].position
        return self.get_2d_distance(agent_pos, object_pos)

    def agent_distance_to_agent(self, agent_id_a, agent_id_b):
        pos_a = self.controller.replicants[agent_id_a].dynamic.transform.position
        pos_b = self.controller.replicants[agent_id_b].dynamic.transform.position
        return self.get_2d_distance(pos_a, pos_b)

    def agent_action_ongoing(self, replicant_id):
        return self.controller.replicants[replicant_id].action.status == ActionStatus.ongoing or len(self.action_buffer[replicant_id]) > 0

    def begin_action_transaction(self, replicant_id, action):
        """Track one already-dispatched high-level action across TDW steps."""
        self.pending_action_transactions[replicant_id] = {
            'action': copy.deepcopy(action),
            'action_type': action.get('type'),
            'object_id': action.get('object'),
            'start_step': int(self.num_step),
            'start_env_step': int(self.num_step) + 1,
            'start_env_step_index': int(self.num_step),
            'start_frame': int(self.num_frames),
        }

    def action_postcondition_failure(self, replicant_id, transaction):
        """Return a stable experiment-layer postcondition failure, if any."""
        if not isinstance(transaction, dict):
            return None
        action_type = transaction.get('action_type')
        object_id = transaction.get('object_id')
        if action_type in [3, 4]:
            if object_id not in self.controller.state.replicants[replicant_id].values():
                return 'postcondition_failed'
        elif action_type in [5, 6] and object_id is not None:
            if object_id in self.controller.state.replicants[replicant_id].values():
                return 'postcondition_failed'
        return None

    def get_object_coop_property(self, object_id):
        return (
            'cooperative'
            if int(object_id) in (self.coop_target_object_ids or [])
            else 'ordinary'
        )

    def discover_grasp_property(self, agent_ids, object_id):
        """Record a property only after all property-independent guards pass."""
        object_id = int(object_id)
        property_value = self.get_object_coop_property(object_id)
        class_name = self.object_names.get(object_id)
        for agent_id in {int(value) for value in agent_ids}:
            self.object_coop_properties.setdefault(agent_id, {})[
                object_id
            ] = property_value
            if class_name is not None:
                self.class_coop_properties.setdefault(agent_id, {})[
                    class_name
                ] = property_value
        return property_value

    def get_public_coop_properties(self, replicant_id, visible_objects):
        """Expose visible targets as unknown and retain legitimately learned ids."""
        replicant_id = int(replicant_id)
        known_objects = copy.deepcopy(
            self.object_coop_properties.get(replicant_id, {})
        )
        class_properties = copy.deepcopy(
            self.class_coop_properties.get(replicant_id, {})
        )
        object_properties = {}
        for item in visible_objects or []:
            if not isinstance(item, dict) or item.get('id') is None:
                continue
            object_id = int(item['id'])
            if object_id not in (self.active_target_object_ids or []):
                continue
            class_name = self.object_names.get(object_id, item.get('name'))
            object_properties[object_id] = known_objects.get(
                object_id,
                class_properties.get(class_name, 'unknown'),
            )
            if class_name is not None:
                class_properties.setdefault(class_name, 'unknown')
        for object_id, property_value in known_objects.items():
            object_properties[int(object_id)] = property_value
        return object_properties, class_properties

    def validate_grasp_common(self, replicant_id, action):
        """Validate grasp conditions that don't depend on hidden coop property."""
        if "object" not in action:
            return "missing_object"
        try:
            object_id = int(action["object"])
        except (TypeError, ValueError):
            return "invalid_object_id"
        if object_id not in self.object_names:
            return "object_not_found"
        if self.agent_distance_to_object(replicant_id, object_id) > self.reach_threshold:
            return "object_out_of_reach"
        if self.agent_holding_anything(replicant_id):
            return "already_holding_object"
        if object_id in self.container_ids:
            return "grasp_container_disabled"
        if self.object_held_by_any_agent(object_id):
            return "grasp_target_already_held"
        if object_id not in (self.active_target_object_ids or []):
            return "grasp_non_active_target"
        return None

    def validate_grasp_action(self, replicant_id, action):
        reason = self.validate_grasp_common(replicant_id, action)
        if reason is not None:
            return reason
        if int(action["object"]) in (self.coop_target_object_ids or []):
            return "requires_two_agent_cooperation"
        return None
    # ==================== End Coop: Ordinary Action Checks ====================

    # ==================== Communication: Envelope and Timing Helpers ====================
    def default_action_feedback(self, agent_id, requested_envelope=None):
        requested_physical = (
            requested_envelope.get('physical_action')
            if isinstance(requested_envelope, dict)
            else None
        )
        return {
            'agent_id': int(agent_id),
            'valid': True,
            'action_status': 'not_started',
            'failure_kind': None,
            'reason_code': None,
            'requested_envelope': copy.deepcopy(requested_envelope),
            'requested_physical_action': copy.deepcopy(requested_physical),
            'effective_physical_action': None,
            'action_transaction': None,
            'underlying_action_continues': False,
        }

    def default_message_feedback(self, agent_id):
        return {
            'agent_id': int(agent_id),
            'attempted': False,
            'sent': False,
            'truncated': False,
            'char_count': 0,
            
            'invalid_reason': None,
        }

    def normalize_action_envelope(self, actions):
        physical_actions = {}
        messages = [None for _ in range(self.number_of_agents)]
        truncated = [False for _ in range(self.number_of_agents)]
        raw_actions = copy.deepcopy(actions)
        envelope_results = action_protocol.normalize_action_envelopes(
            actions,
            self.controller.replicants,
        )
        protocol_reasons = {}
        for replicant_id in self.controller.replicants:
            result = envelope_results[int(replicant_id)]
            physical_action, message, protocol_reason = action_protocol.apply_communication_mode(
                result,
                self.communication_mode,
            )
            physical_actions[str(replicant_id)] = physical_action
            messages[replicant_id], truncated[replicant_id] = self.prepare_outgoing_message(message)
            protocol_reasons[int(replicant_id)] = protocol_reason
        return (
            physical_actions,
            messages,
            truncated,
            raw_actions,
            envelope_results,
            protocol_reasons,
        )

    def prepare_outgoing_message(self, message):
        if message is None or message == '':
            return None, False
        message = str(message)
        truncated = False
        max_chars = int(self.communication_max_message_chars or 12000)
        if len(message) > max_chars:
            message = message[:max_chars]
            truncated = True
        return message, truncated

    def message_char_count(self, message):
        return _count_communication_chars(message)

    def build_step_execution_time_info(self, previous_messages, current_messages, message_truncated, num_frames, active_step_agents):
        # Previous messages remain observable, but their delivery was charged only once to the sender.
        frame_rate = float(self.physical_action_frame_rate or 0.0)
        step_physical_time = float(num_frames) / frame_rate if frame_rate else 0.0
        physical_action_frame_info = {
            'env_step': int(self.num_step) + 1,
            'env_step_index': int(self.num_step),
            'step_physical_action_frames': int(num_frames),
            'step_physical_action_time': step_physical_time,
            'frame_rate': frame_rate,
            'per_agent': {},
        }
        for agent_id in range(self.number_of_agents):
            agent_frames = int(num_frames) if agent_id in active_step_agents else 0
            physical_action_frame_info['per_agent'][agent_id] = {
                'physical_action_active': agent_id in active_step_agents,
                'frames': agent_frames,
                'time': float(agent_frames) / frame_rate if frame_rate else 0.0,
            }

        per_agent = {}
        messages = []
        write_step_time = 0.0
        communication_chars = 0

        for sender_id, message in enumerate(current_messages or []):
            if message is None:
                continue
            chars = self.message_char_count(message)
            communication_chars += chars
            messages.append({
                'sender_agent': sender_id,
                'content': message,
                'message_chars': chars,
                'truncated': bool(message_truncated[sender_id]) if sender_id < len(message_truncated) else False,
            })

        for agent_id in range(self.number_of_agents):
            write_message = current_messages[agent_id] if agent_id < len(current_messages) else None
            send_chars = self.message_char_count(write_message) if write_message is not None else 0
            write_time = float(send_chars) / float(self.communication_send_chars_per_second) if self.communication_send_chars_per_second else 0.0
            physical_time = physical_action_frame_info['per_agent'][agent_id]['time']
            agent_total = max(physical_time, write_time)
            write_step_time += write_time
            per_agent[agent_id] = {
                'write_message': write_message,
                'send_chars': send_chars,
                
                'message_send_time': write_time,
                'physical_action_time': physical_time,
                'total_execution_time': agent_total,
                'embodied_execution_time': agent_total,
            }

        minimum_step_time = (
            float(self.minimum_step_frames) / frame_rate if frame_rate else 0.0
        )
        # Native ongoing actions remain active even without a newly submitted command.
        has_physical_action = bool(active_step_agents) or num_frames > 0
        is_idle_step = not has_physical_action and not messages
        idle_time = minimum_step_time if is_idle_step else 0.0
        communication_step_time = max(agent['message_send_time'] for agent in per_agent.values())
        step_total_time = max(step_physical_time, communication_step_time, idle_time)
        communication_overhead = max(
            0.0,
            step_total_time - step_physical_time - idle_time,
        )
        execution_time = {
            'has_physical_action': has_physical_action,
            'is_idle_step': is_idle_step,
            'idle_step_count': int(is_idle_step),
            'idle_execution_time': idle_time,
            'communication_time_step_sum': communication_step_time,
            'communication_extra_time': communication_overhead,
            'nonidle_execution_time': step_total_time - idle_time,
            'message_count': len(messages),
            'env_step': int(self.num_step) + 1,
            'env_step_index': int(self.num_step),
            'communication_chars': communication_chars,
            'communication_count_unit': self.communication_count_unit,
            'communication_send_chars_per_second': self.communication_send_chars_per_second,
            
            'message_send_time_agent_sum': write_step_time,
            'physical_action_time': step_physical_time,
            'physical_execution_time': step_physical_time,
            'total_execution_time': step_total_time,
            'step_embodied_execution_time': step_total_time,
            'communication_critical_path_overhead': communication_overhead,
            'minimum_step_frames': self.minimum_step_frames,
            'minimum_step_time': minimum_step_time,
            'communication_mode': self.communication_mode,
            'max_embodied_execution_time': self.max_embodied_execution_time,
            'messages': messages,
            'per_agent': per_agent,
        }
        return physical_action_frame_info, execution_time
    # ==================== End Communication: Envelope and Timing Helpers ====================

    def step(self, actions):
        '''
        Run one timestep of the environment's dynamics
        '''
        start = time.time()
        valid = [True for _ in range(self.number_of_agents)]
        invalid_reasons = [None for _ in range(self.number_of_agents)]
        action_status = ['accepted' for _ in range(self.number_of_agents)]
        invalid_events = []
        physical_failure_events = []
        coop_events = []
        handled_agents = set()
        active_step_agents = set()
        pending_coop_grasp = None
        pending_coop_drop = None
        transaction_feedback = {
            i: None for i in range(self.number_of_agents)
        }
        grasp_property_feedback = {
            i: [] for i in range(self.number_of_agents)
        }
        previous_messages = copy.deepcopy(self.messages) if self.messages is not None else [None for _ in range(self.number_of_agents)]
        (
            actions,
            current_messages,
            message_truncated,
            raw_actions,
            envelope_results,
            protocol_reasons,
        ) = self.normalize_action_envelope(actions)
        action_feedback = {
            replicant_id: self.default_action_feedback(
                replicant_id,
                envelope_results[replicant_id].raw,
            )
            for replicant_id in self.controller.replicants
        }
        message_feedback = {}
        for replicant_id in self.controller.replicants:
            raw_envelope = envelope_results[replicant_id].raw
            requested_message = (
                raw_envelope.get('message')
                if isinstance(raw_envelope, dict) and 'message' in raw_envelope
                else envelope_results[replicant_id].message
            )
            message_reason = protocol_reasons[replicant_id]
            if message_reason not in {
                'invalid_message_type',
                'message_blocked_by_ongoing_action',
            }:
                message_reason = None
            message_feedback[replicant_id] = {
                'agent_id': int(replicant_id),
                'attempted': requested_message not in (None, ''),
                'sent': current_messages[replicant_id] is not None,
                'truncated': bool(message_truncated[replicant_id]),
                'char_count': len(current_messages[replicant_id] or ''),

                'invalid_reason': message_reason,
            }

        def stamped_event(event, phase='pre_action'):
            event = copy.deepcopy(event)
            event.setdefault('env_step', int(self.num_step) + 1)
            event.setdefault('env_step_index', int(self.num_step))
            event.setdefault('frame', int(self.num_frames))
            event.setdefault('phase', phase)
            return event

        def mark_invalid(replicant_id, action, reason):
            if invalid_reasons[replicant_id] is not None:
                return
            reason = action_protocol.canonical_action_status_code(reason)
            valid[replicant_id] = False
            invalid_reasons[replicant_id] = reason
            action_status[replicant_id] = 'invalid'
            handled_agents.add(replicant_id)
            invalid_events.append(stamped_event({
                'agent_id': replicant_id,
                'action': copy.deepcopy(action),
                'reason': reason,
                'reason_kind': 'invalid_action',
            }))

        def mark_execution_failure(
            replicant_id,
            action,
            reason,
            raw_terminal_status=None,
        ):
            reason = action_protocol.canonical_action_status_code(reason)
            if invalid_reasons[replicant_id] is None:
                invalid_reasons[replicant_id] = reason
            valid[replicant_id] = False
            if action_status[replicant_id] != 'invalid':
                action_status[replicant_id] = 'failed'
            physical_failure_events.append(stamped_event({
                'agent_id': replicant_id,
                'action': copy.deepcopy(action),
                'reason': reason,
                'reason_kind': 'physical_execution_failure',
                'raw_terminal_status': (
                    None
                    if raw_terminal_status is None
                    else str(raw_terminal_status)
                ),
            }, phase='post_action'))

        def normalize_arm(action):
            if action["type"] == 3:
                action["arm"] = 'right'
            if "arm" in action:
                if action['arm'] == 'left':
                    action['arm'] = Arm.left
                elif action['arm'] == 'right':
                    action['arm'] = Arm.right

        def append_buffer(replicant_id, action, buffer_type):
            self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': buffer_type})
            active_step_agents.add(replicant_id)

        def add_coop_event(event, phase='post_action'):
            event = stamped_event(event, phase=phase)
            coop_events.append(event)
            self.coop_action_events.append(event)

        def add_coop_feedback(event):
            event = stamped_event(event, phase='pre_action')
            coop_feedback_events.append(event)

        def record_grasp_property(
            agent_ids,
            object_id,
            interaction,
            outcome,
            executing_agent=None,
        ):
            property_value = self.discover_grasp_property(
                agent_ids,
                object_id,
            )
            for agent_id in agent_ids:
                grasp_property_feedback[int(agent_id)].append({
                    'type': 'grasp_property_discovered',
                    'env_step': int(self.num_step) + 1,
                    'env_step_index': int(self.num_step),
                    'frame': self.num_frames,
                    'phase': 'pre_action',
                    'agent_id': int(agent_id),
                    'object_id': int(object_id),
                    'object_class': self.object_names.get(int(object_id)),
                    'coop_property': property_value,
                    'interaction': interaction,
                    'outcome': outcome,
                    'executing_agent': executing_agent,
                    'self_executed': (
                        None
                        if executing_agent is None
                        else int(agent_id) == int(executing_agent)
                    ),
                })
            return property_value

        coop_feedback_events = []

        # TDW actions can span multiple calls to step(). Preserve the in-flight
        # action and its buffer: callers may only submit the ongoing sentinel
        # until that transaction reaches a terminal state.
        busy_at_step_start = {
            int(replicant_id): self.agent_action_ongoing(replicant_id)
            for replicant_id in self.controller.replicants
        }
        for replicant_id in self.controller.replicants:
            if not busy_at_step_start[replicant_id]:
                continue
            requested = envelope_results[replicant_id].requested_physical_action
            requested_type = (
                requested.get('type') if isinstance(requested, dict) else None
            )
            actions[str(replicant_id)] = {'type': 'ongoing'}
            active_step_agents.add(replicant_id)

            if message_feedback[replicant_id]['attempted']:
                current_messages[replicant_id] = None
                message_truncated[replicant_id] = False
                message_feedback[replicant_id].update({
                    'sent': False,
                    'truncated': False,
                    'char_count': 0,
                    
                    'invalid_reason': 'message_blocked_by_ongoing_action',
                })
                protocol_reasons[
                    replicant_id
                ] = 'message_blocked_by_ongoing_action'

            if envelope_results[replicant_id].error is not None:
                mark_invalid(
                    replicant_id,
                    requested,
                    envelope_results[replicant_id].error,
                )
            elif requested_type != 'ongoing':
                mark_invalid(
                    replicant_id,
                    requested,
                    'new_action_while_ongoing',
                )

        # Strict envelope failures are physical-action failures. In exclusive
        # mode a new physical command is suppressed while its message is retained;
        # an already-running TDW action is preserved and the new message is blocked.
        for replicant_id in self.controller.replicants:
            if busy_at_step_start[replicant_id]:
                continue
            protocol_reason = protocol_reasons[replicant_id]
            if protocol_reason is None or protocol_reason == 'message_blocked_by_ongoing_action':
                continue
            mark_invalid(replicant_id, actions[str(replicant_id)], protocol_reason)

        implicit_conversions = {}
        if not handled_agents:
            implicit_conversions = action_protocol.apply_implicit_coop_conversion(
                actions,
                self.controller.replicants,
                coop_action_interface=self.coop_action_interface,
                active_coop_carrying=self.active_coop_carrying,
            )
        effective_actions_for_feedback = copy.deepcopy(actions)

        # ==================== Coop: Paired Action Precheck ====================
        # Cooperative actions must be checked using both agents' actions before
        # ordinary per-agent dispatch writes any action buffer.
        if self.active_coop_carrying is not None:
            holder_id = self.active_coop_carrying['holder_id']
            supporter_id = self.active_coop_carrying['supporter_id']
            coop_distance = self.agent_distance_to_agent(holder_id, supporter_id)
            if coop_distance > self.coop_distance_threshold:
                hard_limit = coop_distance > self.coop_hard_distance_threshold
                feedback = {
                    'type': 'coop_carrying_distance_hard' if hard_limit else 'coop_carrying_distance_soft',
                    'frame': self.num_frames,
                    'object_id': self.active_coop_carrying['object_id'],
                    'holder_id': holder_id,
                    'supporter_id': supporter_id,
                    'level': 'hard' if hard_limit else 'soft',
                    'reason': 'coop_carrying_agents_too_far_hard' if hard_limit else 'coop_carrying_agents_too_far_soft',
                    'distance': float(coop_distance),
                    'soft_threshold': float(self.coop_distance_threshold),
                    'hard_threshold': float(self.coop_hard_distance_threshold),
                }
                add_coop_feedback(feedback)
                if hard_limit:
                    holder_action = actions[str(holder_id)]
                    if holder_action['type'] not in [7, 'ongoing']:
                        mark_invalid(holder_id, holder_action, 'coop_carrying_agents_too_far_hard')
                        if actions[str(supporter_id)]['type'] == 6:
                            action_status[supporter_id] = 'coop_waiting'
                            handled_agents.add(supporter_id)

        for replicant_id in self.controller.replicants:
            if replicant_id in handled_agents:
                continue
            action = actions[str(replicant_id)]
            if self.active_coop_carrying is not None and action['type'] not in [0, 1, 2, 6, 7, 'ongoing']:
                mark_invalid(replicant_id, action, 'in_coop_carrying')

        coop_ids = [rid for rid in self.controller.replicants
                    if rid not in handled_agents and actions[str(rid)]['type'] in [4, 6]]
        if len(coop_ids) == 1:
            rid = coop_ids[0]
            mark_invalid(rid, actions[str(rid)], 'coop_action_requires_pair')
        elif len(coop_ids) == 2:
            a0, a1 = actions[str(coop_ids[0])], actions[str(coop_ids[1])]
            if a0['type'] != a1['type']:
                for rid in coop_ids:
                    mark_invalid(rid, actions[str(rid)], 'mismatched_coop_action')
            elif a0['type'] == 4:
                reason = None
                object_id = None
                if self.active_coop_carrying is not None:
                    reason = 'in_coop_carrying'
                elif 'object' not in a0 or 'object' not in a1:
                    reason = 'missing_object'
                else:
                    try:
                        object_ids = (int(a0['object']), int(a1['object']))
                    except (TypeError, ValueError):
                        reason = 'invalid_object_id'
                    if reason is None and object_ids[0] != object_ids[1]:
                        reason = 'coop_grasp_target_mismatch'
                    if reason is None:
                        object_id = object_ids[0]
                if reason is None:
                    if object_id not in self.object_names:
                        reason = 'object_not_found'
                    elif any(
                        self.agent_distance_to_object(rid, object_id)
                        > self.coop_distance_threshold
                        for rid in coop_ids
                    ):
                        reason = 'coop_grasp_too_far'
                    elif (
                        self.agent_distance_to_agent(
                            coop_ids[0], coop_ids[1]
                        ) > self.coop_distance_threshold
                    ):
                        reason = 'coop_grasp_agents_too_far'
                    elif object_id in self.container_ids:
                        reason = 'grasp_container_disabled'
                    elif self.object_held_by_any_agent(object_id):
                        reason = 'coop_grasp_target_already_held'
                    elif any(
                        self.agent_holding_anything(rid)
                        for rid in coop_ids
                    ):
                        reason = 'already_holding_object'
                    elif object_id not in (self.active_target_object_ids or []):
                        reason = 'coop_grasp_non_active_target'
                if reason is not None:
                    for rid in coop_ids:
                        mark_invalid(rid, actions[str(rid)], reason)
                else:
                    distances = {
                        rid: self.agent_distance_to_object(rid, object_id)
                        for rid in coop_ids
                    }
                    executor_id = min(
                        coop_ids,
                        key=lambda rid: (distances[rid], rid),
                    )
                    supporter_id = [
                        rid for rid in coop_ids if rid != executor_id
                    ][0]
                    coop_property = self.get_object_coop_property(object_id)
                    interaction_mode = (
                        'cooperative_carry'
                        if coop_property == 'cooperative'
                        else 'ordinary_fallback'
                    )
                    supporter_status = (
                        'coop_waiting'
                        if coop_property == 'cooperative'
                        else 'conflict_waiting'
                    )
                    outcome = (
                        'coop_grasp_dispatched'
                        if coop_property == 'cooperative'
                        else 'ordinary_grasp_fallback_dispatched'
                    )
                    record_grasp_property(
                        coop_ids,
                        object_id,
                        interaction='coop_grasp',
                        outcome=outcome,
                        executing_agent=executor_id,
                    )
                    exec_action = copy.deepcopy(actions[str(executor_id)])
                    exec_action['arm'] = Arm.right
                    self.begin_action_transaction(executor_id, exec_action)
                    append_buffer(executor_id, exec_action, 'reach_for')
                    append_buffer(executor_id, exec_action, 'grasp')
                    action_status[supporter_id] = supporter_status
                    handled_agents.update(coop_ids)
                    pending_coop_grasp = {
                        'object_id': object_id,
                        'object_class': self.object_names.get(object_id),
                        'holder_id': executor_id,
                        'supporter_id': supporter_id,
                        'coop_property': coop_property,
                        'interaction_mode': interaction_mode,
                        'supporter_status': supporter_status,
                    }
                    add_coop_event({
                        'type': (
                            'coop_grasp_started'
                            if coop_property == 'cooperative'
                            else 'ordinary_grasp_fallback_started'
                        ),
                        'frame': self.num_frames,
                        **copy.deepcopy(pending_coop_grasp),
                    }, phase='action_dispatch')
            elif a0['type'] == 6:
                reason = None
                if self.active_coop_carrying is None:
                    reason = 'no_active_coop_carrying'
                elif any(
                    action.get('object') is not None
                    and int(action['object'])
                    != int(self.active_coop_carrying['object_id'])
                    for action in (a0, a1)
                ):
                    reason = 'coop_drop_object_mismatch'
                elif self.agent_distance_to_agent(coop_ids[0], coop_ids[1]) > self.coop_distance_threshold:
                    reason = 'coop_drop_agents_too_far'
                else:
                    object_id = self.active_coop_carrying['object_id']
                    holders = [rid for rid in coop_ids if object_id in self.controller.state.replicants[rid].values()]
                    if len(holders) != 1:
                        reason = 'coop_object_not_held'
                if reason is not None:
                    for rid in coop_ids:
                        mark_invalid(rid, actions[str(rid)], reason)
                else:
                    holder_id = holders[0]
                    supporter_id = [rid for rid in coop_ids if rid != holder_id][0]
                    drop_arm = self.get_holding_arm(holder_id, object_id)
                    drop_action = copy.deepcopy(actions[str(holder_id)])
                    drop_action['object'] = object_id
                    drop_action['arm'] = drop_arm
                    self.begin_action_transaction(holder_id, drop_action)
                    append_buffer(holder_id, drop_action, 'drop')
                    action_status[supporter_id] = 'coop_waiting'
                    handled_agents.update(coop_ids)
                    pending_coop_drop = {'object_id': object_id, 'holder_id': holder_id, 'supporter_id': supporter_id}
                    add_coop_event({
                        'type': 'coop_drop_started',
                        'frame': self.num_frames,
                        'object_id': object_id,
                        'holder_id': holder_id,
                        'supporter_id': supporter_id,
                    }, phase='action_dispatch')
        # ==================== End Coop: Paired Action Precheck ====================

        # In explicit-only mode, simultaneous legal grasps of the same ordinary
        # target are resolved deterministically instead of depending on TDW's
        # controller update order. Cooperative targets remain blocked below.
        ordinary_grasp_ids = [
            rid for rid in self.controller.replicants
            if rid not in handled_agents and actions[str(rid)]['type'] == 3
        ]
        if len(ordinary_grasp_ids) == 2:
            first_action = actions[str(ordinary_grasp_ids[0])]
            second_action = actions[str(ordinary_grasp_ids[1])]
            if first_action.get('object') == second_action.get('object'):
                object_id = int(first_action['object'])
                if (
                    all(
                        self.validate_grasp_common(rid, actions[str(rid)]) is None
                        for rid in ordinary_grasp_ids
                    )
                    and self.get_object_coop_property(object_id) == 'ordinary'
                ):
                    distances = {
                        rid: self.agent_distance_to_object(rid, object_id)
                        for rid in ordinary_grasp_ids
                    }
                    executor_id = min(
                        ordinary_grasp_ids,
                        key=lambda rid: (distances[rid], rid),
                    )
                    waiting_id = next(
                        rid for rid in ordinary_grasp_ids if rid != executor_id
                    )
                    action_status[waiting_id] = 'conflict_waiting'
                    handled_agents.add(waiting_id)
                    record_grasp_property(
                        [waiting_id],
                        object_id,
                        interaction='ordinary_grasp',
                        outcome='ordinary_grasp_conflict_waiting',
                        executing_agent=executor_id,
                    )

        # ==================== Coop: Action Execution Feedback ====================
        # Keep the original action dispatch structure but record per-agent reasons
        # whenever an action is rejected before reaching the TDW Replicant layer.
        for replicant_id in self.controller.replicants:
            if replicant_id in handled_agents:
                continue
            action = actions[str(replicant_id)]
            if action['type'] == 'ongoing':
                if self.agent_action_ongoing(replicant_id):
                    action_status[replicant_id] = 'ongoing'
                    active_step_agents.add(replicant_id)
                else:
                    mark_invalid(replicant_id, action, 'no_ongoing_action')
                continue
            # otherwise we start an action directly
            self.action_buffer[replicant_id] = []
            normalize_arm(action)
            if action["type"] == 0:       # move forward 0.5m
                self.begin_action_transaction(replicant_id, action)
                append_buffer(replicant_id, action, 'move_forward')
            elif action["type"] == 1:     # turn left by 15 degree
                self.begin_action_transaction(replicant_id, action)
                append_buffer(replicant_id, action, 'turn_left')
            elif action["type"] == 2:     # turn right by 15 degree
                self.begin_action_transaction(replicant_id, action)
                append_buffer(replicant_id, action, 'turn_right')
            elif action["type"] == 3:     # go to and grasp object with arm
                reason = self.validate_grasp_action(replicant_id, action)
                if reason == 'requires_two_agent_cooperation':
                    record_grasp_property(
                        [replicant_id],
                        int(action['object']),
                        interaction='ordinary_grasp',
                        outcome='blocked_requires_cooperation',
                    )
                    mark_invalid(replicant_id, action, reason)
                elif reason is not None:
                    mark_invalid(replicant_id, action, reason)
                else:
                    record_grasp_property(
                        [replicant_id],
                        int(action['object']),
                        interaction='ordinary_grasp',
                        outcome='ordinary_grasp_dispatched',
                        executing_agent=replicant_id,
                    )
                    self.begin_action_transaction(replicant_id, action)
                    append_buffer(replicant_id, action, 'reach_for')
                    append_buffer(replicant_id, action, 'grasp')
            elif action["type"] == 5:      # drop held object in arm
                held_objects = self.get_agent_held_objects(replicant_id)
                requested_object = action.get('object')
                if not held_objects:
                    mark_invalid(replicant_id, action, 'not_holding_object')
                elif (
                    requested_object is not None
                    and int(requested_object) not in held_objects
                ):
                    mark_invalid(replicant_id, action, 'drop_object_not_held')
                else:
                    object_id = (
                        int(requested_object)
                        if requested_object is not None
                        else int(held_objects[0])
                    )
                    action['object'] = object_id
                    action['arm'] = self.get_holding_arm(
                        replicant_id,
                        object_id,
                    )
                    self.begin_action_transaction(replicant_id, action)
                    append_buffer(replicant_id, action, 'drop')
            elif action["type"] == 7:      # wait / no-op
                action_status[replicant_id] = 'wait'
            else:
                mark_invalid(
                    replicant_id,
                    action,
                    'unsupported_action_type',
                )
        # ==================== End Coop: Action Execution Feedback ====================

        # Do action here
        delay_frame_count = [0 for _ in range(self.number_of_agents)]
        finish = False
        num_frames = 0
        while not finish: # continue until any active agent's action finishes
            if len(active_step_agents) == 0:
                break
            for replicant_id in self.controller.replicants:
                if replicant_id not in active_step_agents:
                    continue
                if delay_frame_count[replicant_id] > 0:
                    delay_frame_count[replicant_id] -= 1
                    continue
                if self.controller.replicants[replicant_id].action.status != ActionStatus.ongoing and len(self.action_buffer[replicant_id]) == 0:
                    finish = True
                elif self.controller.replicants[replicant_id].action.status != ActionStatus.ongoing:
                    curr_action = self.action_buffer[replicant_id].pop(0)
                    if curr_action['type'] == 'move_forward':       # move forward 0.5m
                        self.controller.replicants[replicant_id].move_forward()
                    elif curr_action['type'] == 'turn_left':     # turn left by 15 degree
                        self.controller.replicants[replicant_id].turn_by(angle = -15)
                    elif curr_action['type'] == 'turn_right':     # turn right by 15 degree
                        self.controller.replicants[replicant_id].turn_by(angle = 15)
                    elif curr_action['type'] == 'reach_for':     # go to and grasp object with arm
                        distance = self.get_2d_distance(self.controller.replicants[replicant_id].dynamic.transform.position, self.object_manager.transforms[int(curr_action["object"])].position)
                        if distance > self.reach_threshold:
                            mark_invalid(replicant_id, curr_action, 'object_out_of_reach')
                            self.action_buffer[replicant_id] = [] # the action is invaild
                        else: self.controller.replicants[replicant_id].move_to_position(self.object_manager.transforms[int(curr_action["object"])].position)
                    elif curr_action['type'] == 'grasp':
                        self.controller.replicants[replicant_id].grasp(int(curr_action["object"]), curr_action["arm"], relative_to_hand = False, axis = "yaw")
                    elif curr_action["type"] == 'put_in':      # deprecated high-level route
                        mark_invalid(replicant_id, curr_action, 'put_in_deprecated')
                    elif curr_action["type"] == 'drop':      # drop held object in arm
                        self.controller.replicants[replicant_id].drop(curr_action['arm'], max_num_frames = 30)
            if finish: break
            data = self.controller.communicate([])
            for i in range(len(data) - 1):
                r_id = OutputData.get_data_type_id(data[i])
                if r_id == 'imag':
                    images = Images(data[i])
                    if self.save_top_down_images and images.get_avatar_id() == "a":
                        TDWUtils.save_images(images=images, filename= f"{self.num_frames + num_frames:05d}", output_directory = os.path.join(self.save_dir, 'top_down_image'))
            num_frames += 1

        self.num_frames += num_frames
        physical_action_frame_info, execution_time_info = self.build_step_execution_time_info(
            previous_messages, current_messages, message_truncated, num_frames, active_step_agents)
        previous_elapsed_time = float(self.elapsed_embodied_execution_time)
        self.elapsed_embodied_execution_time += float(
            execution_time_info.get('step_embodied_execution_time') or 0.0
        )
        execution_time_info['previous_elapsed_embodied_execution_time'] = previous_elapsed_time
        execution_time_info['elapsed_embodied_execution_time'] = self.elapsed_embodied_execution_time
        self.action_list.append({'raw_actions': raw_actions, 'physical_actions': copy.deepcopy(actions), 'messages': copy.deepcopy(current_messages)})
        reward = 0
        for replicant_id in self.controller.replicants:
            action = actions[str(replicant_id)]
            task_status = self.controller.replicants[replicant_id].action.status
            passive_status = action_status[replicant_id] in ['wait', 'coop_waiting', 'conflict_waiting']
            self.f.write('step: {}, action: {}, time: {}, status: {}\n'
                    .format(self.num_step, action["type"],
                    time.time() - start,
                    action_status[replicant_id] if passive_status else task_status))
            container_info = self.add_name_and_empty(copy.deepcopy(self.controller.state.containment))
            self.f.write('position: {}, forward: {}, containment: {}, goal: {}, container: {}\n'.format(
                    self.controller.replicants[replicant_id].dynamic.transform.position,
                    self.controller.replicants[replicant_id].dynamic.transform.forward,
                    container_info, self.add_name(self.target_object_ids), self.add_name(self.container_ids)))
            self.f.flush()
            if passive_status:
                continue
            if task_status not in [ActionStatus.success, ActionStatus.ongoing, ActionStatus.still_dropping]:
                reward -= 0.1
                mark_execution_failure(
                    replicant_id,
                    action,
                    task_status,
                    raw_terminal_status=task_status,
                )
            elif action_status[replicant_id] in ['accepted', 'ongoing']:
                action_status[replicant_id] = str(task_status)
        
        if pending_coop_grasp is not None:
            object_id = pending_coop_grasp['object_id']
            holder_id = pending_coop_grasp['holder_id']
            coop_property = pending_coop_grasp['coop_property']
            holder_has_object = (
                object_id
                in self.controller.state.replicants[holder_id].values()
            )
            if holder_has_object and coop_property == 'cooperative':
                self.active_coop_carrying = copy.deepcopy(pending_coop_grasp)
                add_coop_event({
                    'type': 'coop_carrying_started',
                    'frame': self.num_frames,
                    **copy.deepcopy(self.active_coop_carrying),
                })
            elif holder_has_object:
                add_coop_event({
                    'type': 'ordinary_grasp_fallback_confirmed',
                    'frame': self.num_frames,
                    **copy.deepcopy(pending_coop_grasp),
                })
            else:
                add_coop_event({
                    'type': (
                        'coop_grasp_failed'
                        if coop_property == 'cooperative'
                        else 'ordinary_grasp_fallback_failed'
                    ),
                    'frame': self.num_frames,
                    **copy.deepcopy(pending_coop_grasp),
                })
        if pending_coop_drop is not None:
            object_id = pending_coop_drop['object_id']
            holder_id = pending_coop_drop['holder_id']
            if object_id not in self.controller.state.replicants[holder_id].values():
                add_coop_event({
                    'type': 'coop_carrying_released',
                    'frame': self.num_frames,
                    **copy.deepcopy(pending_coop_drop),
                })
                self.active_coop_carrying = None

        # Finalize experiment-layer action transactions only after the complete
        # TDW buffer has drained, then verify grasp/drop effects from state.
        for replicant_id in self.controller.replicants:
            transaction = self.pending_action_transactions.get(replicant_id)
            if transaction is None:
                continue
            transaction_ongoing = self.agent_action_ongoing(replicant_id)
            underlying_status = self.controller.replicants[
                replicant_id
            ].action.status
            failure_reason = None
            postcondition_failure = None
            if not transaction_ongoing:
                if underlying_status not in [
                    ActionStatus.success,
                    ActionStatus.still_dropping,
                ]:
                    failure_reason = action_protocol.canonical_action_status_code(
                        underlying_status
                    )
                else:
                    postcondition_failure = self.action_postcondition_failure(
                        replicant_id,
                        transaction,
                    )
                    failure_reason = postcondition_failure
                if (
                    failure_reason is not None
                    and not any(
                        event.get('agent_id') == replicant_id
                        for event in physical_failure_events
                    )
                ):
                    mark_execution_failure(
                        replicant_id,
                        transaction.get('action'),
                        failure_reason,
                        raw_terminal_status=underlying_status,
                    )
            transaction_feedback[replicant_id] = {
                **copy.deepcopy(transaction),
                'state': (
                    'ongoing'
                    if transaction_ongoing
                    else ('failed' if failure_reason is not None else 'completed')
                ),
                'terminal_status': (
                    None if transaction_ongoing else str(underlying_status)
                ),
                'terminal_status_code': (
                    None
                    if transaction_ongoing
                    else action_protocol.canonical_action_status_code(
                        underlying_status
                    )
                ),
                'failure_reason': failure_reason,
                'postcondition_failure': postcondition_failure,
                'end_step': None if transaction_ongoing else int(self.num_step),
                'end_env_step': (
                    None if transaction_ongoing else int(self.num_step) + 1
                ),
                'end_env_step_index': (
                    None if transaction_ongoing else int(self.num_step)
                ),
                'end_frame': None if transaction_ongoing else int(self.num_frames),
            }
            if not transaction_ongoing:
                self.pending_action_transactions[replicant_id] = None

        self.last_action_status = action_status
        self.last_invalid_reasons = invalid_reasons
        self.last_invalid_events = invalid_events
        self.last_coop_action_events = coop_events
        self.last_coop_feedback_events = coop_feedback_events
        self.last_grasp_property_feedback = copy.deepcopy(
            grasp_property_feedback
        )
        for replicant_id in self.controller.replicants:
            entry = action_feedback[replicant_id]
            entry['valid'] = bool(valid[replicant_id])
            entry['action_status'] = action_status[replicant_id]
            # Preserve the distinction between rejected and physically failed actions.
            entry['failure_kind'] = (
                'invalid_action'
                if action_status[replicant_id] == 'invalid'
                else (
                    'physical_execution_failure'
                    if action_status[replicant_id] == 'failed'
                    else None
                )
            )
            entry['reason_code'] = invalid_reasons[replicant_id]
            entry['effective_physical_action'] = copy.deepcopy(
                effective_actions_for_feedback[str(replicant_id)]
            )
            entry['action_transaction'] = copy.deepcopy(
                transaction_feedback[replicant_id]
            )
            entry['underlying_action_continues'] = bool(
                transaction_feedback[replicant_id] is not None
                and transaction_feedback[replicant_id].get('state') == 'ongoing'
            )
            if replicant_id in implicit_conversions:
                entry['implicit_conversion'] = True
                entry['conversion_type'] = implicit_conversions[replicant_id]
            protocol_reason = protocol_reasons[replicant_id]
            if protocol_reason == 'physical_message_mutually_exclusive':
                entry['physical_action_suppressed'] = True
                entry['message_retained'] = True
            elif protocol_reason == 'message_blocked_by_ongoing_action':
                entry['message_blocked'] = True
        self.last_action_feedback = copy.deepcopy(action_feedback)
        self.last_message_feedback = copy.deepcopy(message_feedback)
        self.num_step += 1        
        self.reward += reward
        # Sample terminal progress after coop release/transaction state updates,
        # so progress, latched IDs and budget termination share one snapshot.
        goal_put, goal_total, raw_goal_success = self.check_goal()
        max_embodied_time = self.max_embodied_execution_time
        done, task_success_within_time, self.termination_reason = resolve_episode_termination(
            raw_goal_success, self.elapsed_embodied_execution_time,
            max_embodied_time, self.num_step, self.max_steps,
            self.num_frames, self.max_frame,
        )
        self.success = task_success_within_time
        self.done = done
        obs = self.get_obs()
        # Current-step messages are returned in the next observation. They are
        # independent of physical action success or failure.
        if self.number_of_agents == 2:
            self.messages = copy.deepcopy(current_messages)
            for replicant_id in self.controller.replicants:
                obs[str(replicant_id)]['messages'] = copy.deepcopy(self.messages)

        for replicant_id in self.controller.replicants:
            obs[str(replicant_id)]['valid'] = valid[replicant_id]
            obs[str(replicant_id)]['action_status'] = action_status[replicant_id]
            obs[str(replicant_id)]['coop_action_feedback'] = copy.deepcopy(coop_events)
            obs[str(replicant_id)]['coop_info'] = self.get_coop_info(replicant_id, coop_feedback_events)
            obs[str(replicant_id)]['current_frames'] = self.num_frames

        info = self.get_info()
        info['action_status'] = action_status
        info['invalid_action_events'] = invalid_events
        info['physical_execution_failure_events'] = physical_failure_events
        info['action_failure_events'] = (
            copy.deepcopy(invalid_events)
            + copy.deepcopy(physical_failure_events)
        )
        info['coop_action_events'] = coop_events
        info['coop_feedback_events'] = coop_feedback_events
        info['active_coop_carrying'] = self.active_coop_carrying
        info['action_feedback'] = copy.deepcopy(self.last_action_feedback)
        info['message_feedback'] = copy.deepcopy(self.last_message_feedback)
        info['communication'] = {
            'previous_messages': copy.deepcopy(previous_messages),
            'current_messages': copy.deepcopy(current_messages),
            'messages': copy.deepcopy(execution_time_info.get('messages', [])),
            'message_truncated': copy.deepcopy(message_truncated),
            'communication_mode': self.communication_mode,
            'communication_chars': execution_time_info.get('communication_chars'),
            'message_send_time_agent_sum': execution_time_info.get('message_send_time_agent_sum'),
            'communication_critical_path_overhead': execution_time_info.get('communication_critical_path_overhead'),
        }
        info['physical_action_frame_info'] = physical_action_frame_info
        info['execution_time_step'] = execution_time_info
        info['done'] = done
        info['raw_goal_success'] = bool(raw_goal_success)
        info['goal_progress'] = [goal_put, goal_total, bool(raw_goal_success)]
        info['satisfied_target_ids'] = sorted(int(value) for value in self.satisfied)
        info['task_success_within_time'] = task_success_within_time
        info['termination_reason'] = self.termination_reason
        info['elapsed_embodied_execution_time'] = self.elapsed_embodied_execution_time
        info['max_embodied_execution_time'] = max_embodied_time
        info['num_frames_for_step'] = num_frames
        info['num_step'] = self.num_step
        info['env_step'] = self.num_step
        info['env_step_index'] = self.num_step - 1
        if done:
            info['reward'] = self.reward

        self.obs = obs
        return self.obs_filter(self.obs), reward, done, info
     
    def render(self):
        return None
        
    def save_images(self, save_dir='./Images'):
        '''
        save images of current step, including rgb, depth and segmentation image
        '''
        os.makedirs(save_dir, exist_ok=True)
        for replicant_id in self.controller.replicants:
            save_path = os.path.join(save_dir, str(replicant_id))
            os.makedirs(save_path, exist_ok=True)
            img = self.controller.replicants[replicant_id].dynamic.get_pil_image('img')
            depth = np.flip(np.array(TDWUtils.get_depth_values(self.controller.replicants[replicant_id].dynamic.get_pil_image('depth'), width = self.screen_size, height = self.screen_size), dtype = np.float32), 0)
            depth_img = Image.fromarray(100 / depth).convert('RGB')
            seg = self.controller.replicants[replicant_id].dynamic.get_pil_image('id')
            img.save(os.path.join(save_path, f'{self.num_step:04}_{self.num_frames:04}.png'))
            seg.save(os.path.join(save_path, f'{self.num_step:04}_{self.num_frames:04}_seg.png'))
            depth_img.save(os.path.join(save_path, f'{self.num_step:04}_{self.num_frames:04}_depth.png'))

    def _shutdown_controller(self):
        controller = self.controller
        self.controller = None
        for stage, error in shutdown_controller(controller):
            print('TDW controller cleanup {} failed: {}'.format(stage, error))

    def close(self):
        print('close environment ...')
        self._shutdown_controller()
        if getattr(self, 'f', None) is not None and not self.f.closed:
            self.f.close()
        super().close()
