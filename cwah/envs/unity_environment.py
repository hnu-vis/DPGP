from utils import utils_environment as utils
from envs import action_protocol
from envs.episode_limits import resolve_episode_termination
from envs.physical_time import (
	PHYSICAL_TIME_SETTINGS,
	build_step_physical_time_info,
)
import sys
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{curr_dir}/../../virtualhome/simulation/')
print(f'{curr_dir}/../../virtualhome/simulation/')

from environment.unity_environment import UnityEnvironment as BaseUnityEnvironment
from evolving_graph import utils as utils_env
import pdb
import numpy as np
import copy
import ipdb
from functools import partial
import re

import math


def _count_communication_chars(text):
	if text is None:
		return 0
	return len(str(text))


class UnityEnvironment(BaseUnityEnvironment):

	def __init__(self,
				 num_agents=2,
				 max_episode_length=None,
				 env_task_set=None,
				 observation_types=None,
				 agent_goals=None,
				 use_editor=False,
				 base_port=np.random.randint(11000, 13000),  # sjmchangeit from 8080 to 12345
				 port_id=0,
				 executable_args={},
				 recording_options={'recording': False,
									'output_folder': None,
									'file_name_prefix': None,
									'cameras': 'PERSON_FROM_BACK',
									'modality': 'normal'},
				 data_collection = False,
				 data_collection_dir = None,
				 save_image = False,
				 gt_seg = True,
				 enable_physical_action_time=True,
				 physical_action_frame_rate=5,
				 physical_action_render_mode='joint',
				 communication_mode='exclusive_action',
				 coop_action_interface='implicit_pairing',
				 communication_max_message_chars=12000,
				 communication_send_chars_per_second=2500.0,
				 minimum_step_frames=1,
				 max_embodied_execution_time=600.0,
				 seed=123):

		if agent_goals is not None:
			self.agent_goals = agent_goals
		else:
			self.agent_goals = ['full' for _ in range(num_agents)]

		self.task_goal, self.goal_spec = {0: {}, 1: {}}, {0: {}, 1: {}}
		self.env_task_set = env_task_set
		if max_episode_length is not None and int(max_episode_length) <= 0:
			raise ValueError('max_episode_length must be positive when supplied')
		super(UnityEnvironment, self).__init__(
			num_agents=num_agents,
			max_episode_length=max_episode_length,
			observation_types=observation_types,
			use_editor=use_editor,
			base_port=base_port,
			port_id=port_id,
			executable_args=executable_args,
			recording_options=recording_options,
			seed=seed
		)
		self.full_graph = None
		self.num_camera_per_agent = 9
		self.CAMERA_NUM = 8 
		self.default_image_width = 512
		if self.observation_types[0] == 'full_image':
			self.default_image_width = 1024		
		self.default_image_height = 256
		self.message_said = [None for _ in range(num_agents)]
		self.last_message_feedback = {i: {'sent': False, 'truncated': False, 'char_count': 0} for i in range(num_agents)}
		# ==================== Communication: Execution-Time State ====================
		# Purpose: Configure method-independent message transport and execution-time semantics.
		if communication_mode not in {'exclusive_action', 'parallel_side_channel'}:
			raise ValueError('Unsupported communication_mode: {}'.format(communication_mode))
		if coop_action_interface not in {'explicit_only', 'implicit_pairing'}:
			raise ValueError('Unsupported coop_action_interface: {}'.format(coop_action_interface))
		if int(communication_max_message_chars) <= 0:
			raise ValueError('communication_max_message_chars must be positive')
		if not math.isfinite(float(communication_send_chars_per_second)) or float(communication_send_chars_per_second) <= 0:
			raise ValueError('communication_send_chars_per_second must be positive')

		if int(minimum_step_frames) <= 0:
			raise ValueError('minimum_step_frames must be positive')
		if max_embodied_execution_time is not None and float(max_embodied_execution_time) <= 0:
			raise ValueError('max_embodied_execution_time must be positive or None')
		if physical_action_render_mode != 'joint':
			raise ValueError('Formal C-WAH requires joint physical-action rendering')
		self.communication_mode = communication_mode
		self.coop_action_interface = coop_action_interface
		self.communication_max_message_chars = int(communication_max_message_chars)
		self.communication_send_chars_per_second = float(communication_send_chars_per_second)
		self.communication_count_unit = 'unicode_codepoint'
		self.minimum_step_frames = int(minimum_step_frames)
		self.max_embodied_execution_time = (
			None if max_embodied_execution_time is None
			else float(max_embodied_execution_time)
		)
		self.elapsed_embodied_execution_time = 0.0
		self.termination_reason = None
		# ==================== End Communication: Execution-Time State ====================
		self.gt_seg = gt_seg
		self.save_image = save_image

		#For collecting data
		self.data_collection = data_collection
		self.data_collection_dir = data_collection_dir
		self.global_episode_id = 0
		self.container_classes = [
			'bathroomcabinet',
			'kitchencabinet',
			'cabinet',
			'fridge',
			'stove',
			# 'kitchencounterdrawer',
			# 'coffeepot',
			'dishwasher',
			'microwave']
		self.detection_all_object = ['plate', 'coffeetable', 'poundcake', 'wine', 'juice', 'pudding', 'apple', 'pancake', 'cutleryfork', 'kitchentable', 'cupcake', 'coffeepot'] + self.container_classes + ['character']
		self.detection_name_id_map = {name:  (i + 1) * 10 for i, name in enumerate(self.detection_all_object)}
		self.location = None
		self.navigation_history = None

		# ==================== PAET: Physical Action Execution Time State ====================
		# Estimate benchmark physical time from the fixed calibration settings.
		self.enable_physical_action_time = enable_physical_action_time
		self.physical_time_settings = PHYSICAL_TIME_SETTINGS
		if int(physical_action_frame_rate) != self.physical_time_settings.frame_rate:
			raise ValueError('physical_action_frame_rate must match the benchmark calibration')
		self.physical_action_frame_rate = self.physical_time_settings.frame_rate
		self.physical_action_render_mode = 'joint'
		# ==================== End PAET: Physical Action Execution Time State ====================

		# ==================== Coop: Constraint State ====================
		# Purpose: Keep task-level forced-cooperation metadata without changing action semantics.
		self.enable_coop_carry = True
		self.coop_distance_threshold = 2.0
		self.coop_constraints = {}
		self.known_coop_objects = {i: set() for i in range(num_agents)}
		self.known_coop_classes = {i: set() for i in range(num_agents)}
		self.object_coop_properties = {i: {} for i in range(num_agents)}
		self.class_coop_properties = {i: {} for i in range(num_agents)}
		self.coop_carrying = None
		self.coop_events = []
		self.last_coop_event = None
		self.coop_step_events = []
		self.last_coop_events_by_agent = {i: [] for i in range(num_agents)}
		self.current_action_feedback = {}
		self.last_action_feedback = {
			i: self._default_action_feedback(i, None, status='reset')
			for i in range(num_agents)
		}
		# ==================== End Coop: Constraint State ====================
 
	def reward(self):
		reward = 0.
		done = True
		satisfied, unsatisfied = utils.check_progress(self.get_graph(), self.goal_spec[0])
		for key, value in satisfied.items():
			preds_needed, mandatory, reward_per_pred = self.goal_spec[0][key]
			# How many predicates achieved
			value_pred = min(len(value), preds_needed)
			reward += value_pred * reward_per_pred

			if mandatory and unsatisfied[key] > 0:
				done = False

		self.prev_reward = reward
		return reward, done, {'satisfied_goals': satisfied}

	def get_action_space(self):
		dict_action_space = {}
		for agent_id in range(self.num_agents):
			if 'image' in self.observation_types[agent_id]:
				return None
			#In vision, no RL is applied
			if self.observation_types[agent_id] not in ['partial', 'full']:
				raise NotImplementedError
			# TODO:  Now the action space is base on graph
			else:
				# Even if you can see all the graph, you can only interact with visible objects
				obs_type = 'partial'
			visible_graph = self.get_observation(agent_id, obs_type)
			dict_action_space[agent_id] = [node['id'] for node in visible_graph['nodes']]
		return dict_action_space

	def check_subgoal(self, subgoal):
		# Check if the subgoal is satisfied
		# If the subgoal is satisfied, return True
		# If the subgoal is not satisfied, return False
		if subgoal is None: return False
		if type(subgoal) == list: 
			if len(subgoal) == 0: return False
			else: subgoal = subgoal[0]
		id_to_grab = int(subgoal.split('_')[1])
		class_to_grab = self.id_to_name[id_to_grab]
		satisfied, unsatisfied = utils.check_progress(self.get_graph(), self.goal_spec[0])
		for x in satisfied.keys():
			for t in satisfied[x]:
				if t is None: continue
				if str(id_to_grab) in t:
					print(subgoal, 'is satisfied')
					return True
		for x in unsatisfied.keys():
			if class_to_grab in x and unsatisfied[x] == 0:
				print(subgoal, 'is satisfied')
				return True
		return False
	
	def get_goal(self, task_spec, agent_goal):
		if agent_goal == 'full':
			pred = [x for x, y in task_spec.items() if y > 0 and x.split('_')[0] in ['on', 'inside']]
			# object_grab = [pr.split('_')[1] for pr in pred]
			# predicates_grab = {'holds_{}_1'.format(obj_gr): [1, False, 2] for obj_gr in object_grab}
			res_dict = {goal_k: [goal_c, True, 2] for goal_k, goal_c in task_spec.items()}
			# res_dict.update(predicates_grab)
			return res_dict
		elif agent_goal == 'grab':
			candidates = [x.split('_')[1] for x, y in task_spec.items() if
						  y > 0 and x.split('_')[0] in ['on', 'inside']]
			object_grab = self.rnd.choice(candidates)
			# print('GOAL', candidates, object_grab)
			return {'holds_' + object_grab + '_' + '1': [1, True, 10],
					'close_' + object_grab + '_' + '1': [1, False, 0.1]}
		elif agent_goal == 'put':
			pred = self.rnd.choice([x for x, y in task_spec.items() if y > 0 and x.split('_')[0] in ['on', 'inside']])
			object_grab = pred.split('_')[1]
			return {
				pred: [1, True, 60],
				'holds_' + object_grab + '_' + '1': [1, False, 2],
				'close_' + object_grab + '_' + '1': [1, False, 0.05]

			}
		elif agent_goal == 'LLM':  # todo: Hongxin added
			'''
			Hongxin changed task_goal from "on_cupcake_268" to "on_cupcake_<coffeetable> (268)"
			'''
			goal_class = {}
			for predicate in self.goal_class.keys():
				rel, obj1, obj2 = predicate.split('_')
				goal_class[f"{rel}_{obj1}"] = obj2
			new_task_goal = {}
			for predicate, count in task_spec.items():
				if count == 0:
					continue
				rel, obj1, obj2 = predicate.split('_')
				obj2_name = goal_class[f"{rel}_{obj1}"]
				new_predicate = predicate.replace(obj2, f"<{obj2_name}> ({obj2})")
				new_task_goal[new_predicate] = count
			# print(new_task_goal)
			res_dict = {goal_k: [goal_c, True, 2] for goal_k, goal_c in new_task_goal.items()}
			# res_dict.update(predicates_grab)
			return res_dict
		else:
			raise NotImplementedError

	@property
	def all_relative_name(self) -> list:
		return self.all_containers_name + self.all_goal_objects_name + ['character']

	@property
	def all_relative_id(self) -> list:
		return [node['id'] for node in self.full_graph['nodes'] if node['class_name'] in self.all_relative_name]

	@property
	def all_detection_id(self) -> list:
		return [node['id'] for node in self.full_graph['nodes'] if node['class_name'] in self.detection_all_object]

	@property
	def all_containers_name(self) -> list:
		r'''
		 get all containers in the scene, exclude rooms and containers with no objects inside.
		'''
		'''
		id2node = {node['id']: node for node in self.full_graph['nodes']}
		room_name = [node['class_name'] for node in self.full_graph['nodes'] if node['category'] == 'Rooms']
		all_container = list(set([id2node[link['to_id']]['class_name'] for link in self.full_graph['edges'] if
								  link['relation_type'] == 'INSIDE']))
		all_container = [x for x in all_container if x not in room_name]
		'''
		container_classes = [
			'bathroomcabinet',
			'kitchencabinet',
			'cabinet',
			'fridge',
			'stove',
			# 'coffeepot',
			'dishwasher',
			'microwave']
		return container_classes

	@property
	def all_goal_objects_name(self) -> list:
		r'''
		 get all objects that related to goal.
		ZHX: update to adapt to new goal_spec of LLM
		'''
		goal_objects = []
		id2node = {node['id']: node for node in self.full_graph['nodes']}
		for predicate in self.goal_spec[0]:
			elements = predicate.split('_')
			for x in elements[1:]:
				if x.isdigit():
					goal_objects += [id2node[int(x)]['class_name']]
				elif '(' in x:
					y = x.split('(')[1].split(')')[0]
					if y.isdigit():
						goal_objects += [id2node[int(y)]['class_name']]
				else:
					goal_objects += [x]
		goal_obj = list(set(goal_objects))
		# if ('character' not in goal_obj):
		# 	goal_obj += ['character']
		return goal_obj

	@property
	def room_info(self):
		r'''
		 get room info in the scene.
		'''
		return [node for node in self.full_graph['nodes'] if node['id'] in self.all_room_and_character_id]

	@property
	def all_room_name(self) -> list:
		r'''
		 get all rooms in the scene.
		'''
		# room_name = [node['class_name'] for node in self.full_graph['nodes'] if node['category'] == 'Rooms']
		room_name = ["livingroom", "kitchen", "bedroom", "bathroom"]
		return room_name


	@property
	def all_room_and_character_id(self) -> list:
		r'''
		get all room_and_character_ids in the scene.
		'''
		return [node['id'] for node in self.full_graph['nodes'] if
						  node['class_name'] == 'character' or node['category'] in ['Rooms']]
	
	@property
	def all_room_id(self) -> list:
		r'''
		get all room_and_character_ids in the scene.
		'''
		return [node['id'] for node in self.full_graph['nodes'] if node['category'] in ['Rooms']]

	def filter_graph(self, obs):
		relative_id = self.all_relative_id + self.all_room_id
		new_graph = {
			"edges": [edge for edge in obs['edges'] if
					edge['from_id'] in relative_id and edge['to_id'] in relative_id],
			"nodes": [node for node in obs['nodes'] if node['id'] in relative_id]
		}
		return new_graph

	def reset(self, environment_graph=None, task_id=None):
		# Make sure that characters are out of graph, and ids are ok
		# ipdb.set_trace()
		self.global_episode_id += 1 # For collecting data
		if task_id is None:
			task_id = self.rnd.choice(list(range(len(self.env_task_set))))
		env_task = self.env_task_set[task_id]

		self.task_id = env_task['task_id']
		self.init_graph = copy.deepcopy(env_task['init_graph'])
		self.init_rooms = env_task['init_rooms']
		self.init_agent_positions = env_task.get('init_agent_positions', None)
		self.task_goal = env_task['task_goal']
		print('task_goal: ', self.task_goal)
		if 'total_goal' in env_task.keys():
			print('total_goal: ', env_task['total_goal'])
		self.goal_class = env_task['goal_class']
		self.task_name = env_task['task_name']
		self.cache_id_map = dict()
		self.cache_data_map = dict() # For collecting data

		self._init_coop_state(env_task)

		old_env_id = self.env_id
		self.env_id = env_task['env_id']
		print("Resetting... Envid: {}. Taskid: {}. Index: {}".format(self.env_id, self.task_id, task_id))

		# TODO: in the future we may want different goals
		self.goal_spec = {agent_id: self.get_goal(self.task_goal[agent_id], self.agent_goals[agent_id])
						  for agent_id in range(self.num_agents)}

		if False:  # old_env_id == self.env_id:
			print("Fast reset")
			self.comm.fast_reset()
		else:
			self.comm.reset(self.env_id)

		s, g = self.comm.environment_graph()
		edge_ids = set([edge['to_id'] for edge in g['edges']] + [edge['from_id'] for edge in g['edges']])
		node_ids = set([node['id'] for node in g['nodes']])
		if len(edge_ids - node_ids) > 0:
			pdb.set_trace()

		if self.env_id not in self.max_ids.keys():
			max_id = max([node['id'] for node in g['nodes']])
			self.max_ids[self.env_id] = max_id

		max_id = self.max_ids[self.env_id]

		if environment_graph is not None:
			updated_graph = environment_graph
			s, g = self.comm.environment_graph()
			updated_graph = utils.separate_new_ids_graph(updated_graph, max_id)
			success, m = self.comm.expand_scene(updated_graph)
		else:
			updated_graph = self.init_graph
			s, g = self.comm.environment_graph()
			updated_graph = utils.separate_new_ids_graph(updated_graph, max_id)
			success, m = self.comm.expand_scene(updated_graph)

		if not success:
			print("Error expanding scene")
			return None

		self.offset_cameras = self.comm.camera_count()[1]
		if self.init_rooms[0] not in ['kitchen', 'bedroom', 'livingroom', 'bathroom']:
			rooms = self.rnd.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)
		else:
			rooms = list(self.init_rooms)

		self.num_static_cameras = self.comm.camera_count()[1]
		s, g = self.comm.add_character_camera(position=[0, 1.8, 0.15], rotation=[30,0,90], name="up_camera")
		#field_view=90, does not work, since the unity env do not support it
		for i in range(self.num_agents):
			character_position = self._init_agent_position_for_agent(i)
			if character_position is not None:
				print('[InitPositions] add character {} at {}'.format(i, character_position))
				if i in self.agent_info:
					self.comm.add_character(self.agent_info[i], position=character_position)
				else:
					self.comm.add_character(position=character_position)
			elif i in self.agent_info:
				self.comm.add_character(self.agent_info[i], initial_room=rooms[i])
			else:
				self.comm.add_character()
		_, self.init_unity_graph = self.comm.environment_graph()

		self.changed_graph = True
		graph = self.get_graph()
		self.rooms = [(node['class_name'], node['id']) for node in graph['nodes'] if node['category'] == 'Rooms']
		self.id2node = {node['id']: node for node in graph['nodes']}

		curr_graph = self.get_graph()
		curr_graph = utils.inside_not_trans(curr_graph)
		self.full_graph = curr_graph
		self.id_to_name = {node['id']: node['class_name'] for node in curr_graph['nodes']}
		self.message_said = [None for _ in range(self.num_agents)]
		self.last_message_feedback = {i: {'sent': False, 'truncated': False, 'char_count': 0} for i in range(self.num_agents)}

		self.location = [[] for _ in range(self.num_agents)]
		self.last_script = [None for _ in range(self.num_agents)]
		self.navigation_history = [[] for _ in range(self.num_agents)]
		self.elapsed_embodied_execution_time = 0.0
		self.termination_reason = None
		initial_satisfied, initial_unsatisfied = utils.check_progress(
			self.get_graph(), self.goal_spec[0]
		)
		self.initial_progress = {
			'satisfied': copy.deepcopy(initial_satisfied),
			'unsatisfied': copy.deepcopy(initial_unsatisfied),
		}
		obs = self.get_observations()
		self.steps = 0
		self.prev_reward = 0.
		return obs

	def handle_script(script_list):
		pass

	# ==================== Coop: Fixed Initial Agent Positions ====================
	# Purpose: Use dataset-provided agent coordinates while preserving room fallback.
	def _init_agent_position_for_agent(self, agent_id):
		init_positions = getattr(self, 'init_agent_positions', None)
		if not isinstance(init_positions, list) or agent_id >= len(init_positions):
			return None

		position = init_positions[agent_id]
		if position is None:
			return None
		if not isinstance(position, (list, tuple)) or len(position) < 3:
			print('[InitPositions][warning] ignore invalid position for agent {}: {}'.format(
				agent_id,
				position,
			))
			return None
		return list(position)
	# ==================== End Coop: Fixed Initial Agent Positions ====================
	
	def range_distance(self, pos_list):
		max_dis = 0
		for i in range(len(pos_list)):
			for j in range(i+1, len(pos_list)):
				dis = np.linalg.norm(np.array(pos_list[i]) - np.array(pos_list[j]))
				if dis > max_dis:
					max_dis = dis
		return max_dis

	def _apply_navigation_stall_recovery(self, agent_id, parsed):
		"""Redirect five stalled attempts at one target; messages preserve evidence."""
		history = self.navigation_history[agent_id]
		if parsed.kind == action_protocol.WAIT:
			return parsed
		if parsed.kind != 'walktowards':
			history.clear()
			return parsed

		target_id = int(parsed.object_id)
		if history and history[-1][0] != target_id:
			history.clear()
		locations = self.location[agent_id] if self.location is not None else []
		current = tuple(locations[-1]) if locations else None
		stalled = current is not None and len(history) >= 5 and self.range_distance(
			[position for _, position in history[-5:]] + [current]) < 0.5
		if stalled:
			rooms = getattr(self, 'rooms', [])
			history.clear()
			if not rooms:
				self._mark_action_invalid(agent_id, 'navigation_stalled')
				return None
			# Preserve upstream randint's exclusive upper bound (including its
			# omission of the last room); guard the degenerate one-room case.
			random_place = rooms[np.random.randint(0, max(1, len(rooms) - 1))]
			redirect_action = '[walktowards] <{}> ({})'.format(*random_place)
			candidate, reason = action_protocol.parse_physical_action(redirect_action)
			details = None
			if reason is None:
				reason, details = action_protocol.validate_class_ids(self, candidate)
			if reason is None:
				reason = action_protocol.validate_active_carrying_action(self, candidate)
			if reason is not None:
				self._mark_action_invalid(agent_id, reason, details=details)
				return None
			self.current_action_feedback[agent_id]['navigation_recovery'] = {
				'original_physical_action': action_protocol.format_physical_action(parsed),
				'redirected_physical_action': redirect_action,
			}
			print('[navigation_recovery] agent{} randomly changing walktowards to {}'.format(
				agent_id, redirect_action))
			return candidate
		if current is not None:
			history.append((target_id, current))
			del history[:-5]
		return parsed


	# ==================== PAET: Physical Action Execution Time Helpers ====================
	def _build_calibrated_physical_action_info(
		self,
		submitted_actions,
		charged_actions,
		charge_reasons,
		before_graph,
		after_graph,
		render_success,
	):
		return build_step_physical_time_info(
			step=self.steps,
			num_agents=self.num_agents,
			submitted_actions=submitted_actions,
			charged_actions=charged_actions,
			charge_reasons=charge_reasons,
			before_graph=before_graph,
			after_graph=after_graph,
			render_success=render_success,
			enabled=self.enable_physical_action_time,
			timing=self.physical_time_settings,
		)
	# ==================== End PAET: Physical Action Execution Time Helpers ====================

	# ==================== Communication: Execution-Time Helpers ====================
	# Purpose: Estimate per-step read/write message time without treating communication as a physical action.
	def _prepare_outgoing_message(self, message):
		if message is None or message == '':
			return None, False
		message = str(message)
		truncated = False
		max_chars = int(getattr(self, 'communication_max_message_chars', 12000) or 12000)
		if len(message) > max_chars:
			message = message[:max_chars]
			truncated = True
		return message, truncated

	def _message_char_count(self, message):
		return _count_communication_chars(message)

	def _agent_physical_action_time(self, physical_action_frame_info, agent_id):
		per_agent = (physical_action_frame_info or {}).get('per_agent', {}) or {}
		agent_info = per_agent.get(agent_id, per_agent.get(str(agent_id), {}))
		if not isinstance(agent_info, dict):
			return 0.0
		time_value = agent_info.get('time')
		if time_value is not None:
			return float(time_value or 0.0)
		frames = int(agent_info.get('frames') or 0)
		frame_rate = (physical_action_frame_info or {}).get('frame_rate') or self.physical_action_frame_rate
		return float(frames) / float(frame_rate) if frame_rate else 0.0

	def _build_step_execution_time_info(self, previous_messages, current_messages, message_truncated, physical_action_frame_info):
		# Previous messages remain observable, but their delivery was charged only once to the sender.
		per_agent = {}
		messages = []
		send_step_time = 0.0
		step_total_time = 0.0
		communication_chars = 0

		for sender_id, message in enumerate(current_messages or []):
			if message is None:
				continue
			chars = self._message_char_count(message)
			communication_chars += chars
			messages.append({
				'sender_agent': sender_id,
				'content': message,
				'message_chars': chars,
				'truncated': bool(message_truncated[sender_id]) if sender_id < len(message_truncated) else False,
			})

		for agent_id in range(self.num_agents):
			write_message = current_messages[agent_id] if agent_id < len(current_messages) else None
			send_chars = self._message_char_count(write_message) if write_message is not None else 0
			send_rate = float(self.communication_send_chars_per_second)
			write_time = float(send_chars) / send_rate if send_rate else 0.0
			physical_time = self._agent_physical_action_time(physical_action_frame_info, agent_id)
			agent_total = max(physical_time, write_time)
			send_step_time += write_time
			step_total_time = max(step_total_time, agent_total)
			per_agent[agent_id] = {
				'write_message': write_message,
				'send_chars': send_chars,
				
				'message_send_time': write_time,
				'physical_action_time': physical_time,
				'total_execution_time': agent_total,
				'embodied_execution_time': agent_total,
			}

		frame_rate = float((physical_action_frame_info or {}).get('frame_rate') or self.physical_action_frame_rate or 0.0)
		minimum_step_time = (
			float(getattr(self, 'minimum_step_frames', 1)) / frame_rate
			if frame_rate > 0 else 0.0
		)
		physical_step_time = float((physical_action_frame_info or {}).get('step_physical_action_time') or 0.0)
		physical_agents = (physical_action_frame_info or {}).get('per_agent') or {}
		has_physical_action = physical_step_time > 0 or any(
			agent.get('submitted_physical_action') or agent.get('charged_physical_attempt')
			for agent in physical_agents.values()
		)
		is_idle_step = not has_physical_action and not messages
		idle_time = minimum_step_time if is_idle_step else 0.0
		communication_step_time = max(
			(agent['message_send_time'] for agent in per_agent.values()), default=0.0)
		step_total_time = max(physical_step_time, communication_step_time, idle_time)
		communication_overhead = max(
			0.0,
			step_total_time - physical_step_time - idle_time,
		)
		return {
			'step': self.steps - 1,
			'communication_chars': communication_chars,
			'communication_count_unit': self.communication_count_unit,
			'communication_send_chars_per_second': self.communication_send_chars_per_second,
			
			'message_send_time_agent_sum': send_step_time,
			'communication_time_step_sum': communication_step_time,
			'message_count': len(messages),
			'is_idle_step': is_idle_step,
			'has_physical_action': has_physical_action,
			'idle_step_count': int(is_idle_step),
			'idle_execution_time': idle_time,
			'nonidle_execution_time': step_total_time - idle_time,
			'communication_extra_time': communication_overhead,
			'physical_action_time': physical_step_time,
			'physical_execution_time': physical_step_time,
			'total_execution_time': step_total_time,
			'step_embodied_execution_time': step_total_time,
			'communication_critical_path_overhead': communication_overhead,
			'minimum_step_frames': int(getattr(self, 'minimum_step_frames', 1)),
			'minimum_step_time': minimum_step_time,
			'communication_mode': getattr(self, 'communication_mode', 'exclusive_action'),
			'max_embodied_execution_time': getattr(self, 'max_embodied_execution_time', None),
			'messages': messages,
			'per_agent': per_agent,
		}

	def _print_communication_step_debug(self, execution_time_info):
		for agent_id, agent in (execution_time_info.get('per_agent') or {}).items():
			print('[CommTime] step={} agent={} send_chars={} end_to_end_seconds={}'.format(
				execution_time_info.get('step'), agent_id,
				agent.get('send_chars', 0), agent.get('message_send_time', 0.0)))
	# ==================== End Communication: Execution-Time Helpers ====================

	# ==================== Execution Failure Classification Helpers ====================
	# Purpose: Normalize Unity render_script transport failures and semantic executor failures.
	def _flatten_render_message_text(self, message):
		if message is None:
			return ''
		if isinstance(message, str):
			return message
		if isinstance(message, (list, tuple)):
			return ' '.join(self._flatten_render_message_text(item) for item in message)
		if isinstance(message, dict):
			parts = []
			for key, value in message.items():
				parts.append(str(key))
				parts.append(self._flatten_render_message_text(value))
			return ' '.join(parts)
		return str(message)

	def _classify_render_execution_failure(self, success, message):
		reasons = []
		if not success:
			reasons.append('render_script_failed')

		message_text = self._flatten_render_message_text(message).lower()
		message_patterns = [
			('script_impossible', 'script is impossible'),
			('execution_general', 'execution_general'),
			('cannot_reach', 'cannot reach'),
			('max_frame_exceeded', 'max frame number exceeded'),
		]
		for reason, pattern in message_patterns:
			if pattern in message_text:
				reasons.append(reason)

		return len(reasons) > 0, reasons
	# ==================== End Execution Failure Classification Helpers ====================

	# ==================== Coop: Constraint Helpers ====================
	# Purpose: Read forced-coop constraints and expose env-level coop state.
	def _init_coop_state(self, env_task):
		self.coop_constraints = copy.deepcopy(env_task.get('coop_constraints', {}) or {})
		self.known_coop_objects = {i: set() for i in range(self.num_agents)}
		self.known_coop_classes = {i: set() for i in range(self.num_agents)}
		self.object_coop_properties = {i: {} for i in range(self.num_agents)}
		self.class_coop_properties = {i: {} for i in range(self.num_agents)}
		self.coop_carrying = None
		self.coop_events = []
		self.last_coop_event = None
		self.coop_step_events = []
		self.last_coop_events_by_agent = {i: [] for i in range(self.num_agents)}
		self.current_action_feedback = {}
		self.last_action_feedback = {
			i: self._default_action_feedback(i, None, status='reset')
			for i in range(self.num_agents)
		}

	def _coop_enabled(self):
		return bool(getattr(self, 'enable_coop_carry', True))

	# ==================== Coop: Public Action Feedback ====================
	# Purpose: Expose only the current agent's previous-step action result and coop events.
	def _default_action_feedback(self, agent_id, requested_action, status=None, requested_physical_action=None):
		if requested_physical_action is None and not isinstance(requested_action, dict):
			requested_physical_action = requested_action
		if status is None:
			status = 'wait' if requested_physical_action is None else 'accepted'
		return {
			'agent_id': int(agent_id),
			'valid': True,
			'action_status': status,
			'invalid_reason': None,
			'reason_code': None,
			'requested_action': copy.deepcopy(requested_action),
			'requested_envelope': copy.deepcopy(requested_action) if isinstance(requested_action, dict) else None,
			'requested_physical_action': copy.deepcopy(requested_physical_action),
			'effective_action': copy.deepcopy(requested_physical_action),
			'effective_physical_action': copy.deepcopy(requested_physical_action),
		}

	def _start_action_feedback_step(self, requested_actions, requested_physical_actions=None):
		self.coop_step_events = []
		requested_physical_actions = requested_physical_actions or requested_actions
		self.current_action_feedback = {
			agent_id: self._default_action_feedback(
				agent_id,
				requested_actions.get(agent_id),
				requested_physical_action=requested_physical_actions.get(agent_id),
			)
			for agent_id in range(self.num_agents)
		}

	def _set_action_feedback(self, agent_id, valid=None, status=None, reason=None, details=None):
		agent_id = int(agent_id)
		if agent_id not in self.current_action_feedback:
			self.current_action_feedback[agent_id] = self._default_action_feedback(agent_id, None)
		entry = self.current_action_feedback[agent_id]
		if valid is not None:
			entry['valid'] = bool(valid)
		if status is not None:
			entry['action_status'] = status
		if reason is not None:
			entry['invalid_reason'] = reason
			entry['reason_code'] = reason
		if details is not None:
			entry['details'] = copy.deepcopy(details)

	def _mark_action_invalid(self, agent_id, reason, details=None):
		self._set_action_feedback(
			agent_id,
			valid=False,
			status='invalid',
			reason=reason,
			details=details,
		)

	def _finalize_step_public_feedback(self, effective_actions, render_messages):
		messages_by_agent = {
			item.get('agent_id'): item
			for item in render_messages or []
			if isinstance(item, dict)
		}
		for agent_id in range(self.num_agents):
			entry = self.current_action_feedback.setdefault(
				agent_id,
				self._default_action_feedback(agent_id, None),
			)
			effective_action = effective_actions.get(agent_id)
			entry['effective_action'] = copy.deepcopy(effective_action)
			entry['effective_physical_action'] = copy.deepcopy(effective_action)
			if entry.get('action_status') in ['invalid', 'failed', 'success', 'coop_waiting', 'conflict_waiting', 'message']:
				continue
			render_result = messages_by_agent.get(agent_id)
			if render_result is not None:
				if render_result.get('failed_exec', False):
					reasons = render_result.get('failed_exec_reasons') or ['execution_failed']
					self._set_action_feedback(
						agent_id,
						valid=False,
						status='failed',
						reason=reasons[0],
						details={'reasons': reasons, 'message': render_result.get('message')},
					)
				else:
					self._set_action_feedback(agent_id, valid=True, status='success')
			elif effective_action is None:
				self._set_action_feedback(agent_id, valid=True, status='wait')

		self.last_action_feedback = copy.deepcopy(self.current_action_feedback)
		self.last_coop_events_by_agent = {}
		for agent_id in range(self.num_agents):
			public_events = []
			for event in getattr(self, 'coop_step_events', []):
				public_event = self._public_coop_event_for_agent(event, agent_id)
				if public_event is not None:
					public_events.append(public_event)
			self.last_coop_events_by_agent[agent_id] = public_events

	# ==================== End Coop: Public Action Feedback ====================


	# ==================== Coop: Proximity Helpers ====================
	# Purpose: Query graph nodes, CLOSE edges, distances, and discovery state.
	def _get_node_from_current_graph(self, node_id):
		if node_id is None:
			return None

		node_id = int(node_id)
		graph = self.get_graph()
		for node in graph.get('nodes', []):
			if node.get('id') == node_id:
				return node
		return None

	def _get_class_name_by_id(self, obj_id):
		if obj_id is None:
			return None

		node = self._get_node_from_current_graph(obj_id)
		if node is None:
			return None
		return node.get('class_name')

	def _get_coop_constraint_for_object(self, obj_id):
		if obj_id is None:
			return None

		obj_id = int(obj_id)
		class_name = self._get_class_name_by_id(obj_id)

		# Forced-cooperation constraints are keyed by object class.
		if class_name is not None and class_name in self.coop_constraints:
			return self.coop_constraints[class_name]
		return None

	def _is_coop_required(self, obj_id):
		if not self._coop_enabled():
			return False

		constraint = self._get_coop_constraint_for_object(obj_id)
		if constraint is None:
			return False
		return bool(constraint.get('requires_two_agents', False))

	def _get_node_position_from_current_graph(self, node_id):
		node = self._get_node_from_current_graph(node_id)
		if node is None:
			return None

		transform = node.get('obj_transform', {})
		position = transform.get('position', None)
		if position is None:
			return None
		return np.array(position, dtype=float)

	def _edge_close(self, id1, id2):
		if id1 is None or id2 is None:
			return False

		id1 = int(id1)
		id2 = int(id2)
		graph = self.get_graph()
		for edge in graph.get('edges', []):
			if edge.get('relation_type') != 'CLOSE':
				continue
			from_id = edge.get('from_id')
			to_id = edge.get('to_id')
			if (from_id == id1 and to_id == id2) or (from_id == id2 and to_id == id1):
				return True
		return False

	def _distance_between_nodes(self, id1, id2):
		p1 = self._get_node_position_from_current_graph(id1)
		p2 = self._get_node_position_from_current_graph(id2)
		if p1 is None or p2 is None:
			return None
		return float(np.linalg.norm(p1 - p2))

	def _coop_carry_distance_threshold(self):
		# Keep the active-carrying hard limit looser than the 2.0 soft
		# threshold so short-lived separation does not unnecessarily pause
		# the physical holder.
		return self.coop_distance_threshold * 1.5

	def _agent_close_to_object(self, agent_id, obj_id):
		char_id = int(agent_id) + 1
		return self._edge_close(char_id, int(obj_id))

	def _agents_close_to_each_other(self, agent_a=0, agent_b=1):
		char_a = int(agent_a) + 1
		char_b = int(agent_b) + 1
		return self._edge_close(char_a, char_b)

	def _agents_close_for_coop_interaction(self, holder_agent=0, partner_agent=1):
		char_a = int(holder_agent) + 1
		char_b = int(partner_agent) + 1
		distance = self._distance_between_nodes(char_a, char_b)
		threshold = self.coop_distance_threshold
		edge_close = self._edge_close(char_a, char_b)
		return edge_close, {
			'distance': distance,
			'threshold': threshold,
			'edge_close': edge_close,
			'distance_fallback_disabled': True,
		}

	def _agents_close_for_active_coop_carrying(self):
		soft_threshold = self.coop_distance_threshold
		hard_threshold = self._coop_carry_distance_threshold()
		if not self.coop_carrying:
			return True, {
				'distance': None,
				'soft_threshold': soft_threshold,
				'hard_threshold': hard_threshold,
				'edge_close': False,
				'distance_level': 'inactive',
				'active_carrying': False,
			}

		holder_agent = self.coop_carrying['holder_agent']
		partner_agent = self.coop_carrying['partner_agent']
		char_a = int(holder_agent) + 1
		char_b = int(partner_agent) + 1
		distance = self._distance_between_nodes(char_a, char_b)
		edge_close = self._edge_close(char_a, char_b)
		if distance is None:
			distance_level = 'ok' if edge_close else 'unknown'
			return edge_close, {
				'distance': None,
				'soft_threshold': soft_threshold,
				'hard_threshold': hard_threshold,
				'edge_close': edge_close,
				'distance_level': distance_level,
			}
		if distance > hard_threshold:
			distance_level = 'hard'
		elif distance > soft_threshold:
			distance_level = 'soft'
		else:
			distance_level = 'ok'
		return distance_level != 'hard', {
			'distance': distance,
			'soft_threshold': soft_threshold,
			'hard_threshold': hard_threshold,
			'edge_close': edge_close,
			'distance_level': distance_level,
		}

	def _actual_coop_property(self, obj_id):
		return 'cooperative' if self._is_coop_required(obj_id) else 'ordinary'

	def _discover_grab_property(self, agent_ids, obj_id):
		"""Publish ordinary/cooperative only after a legal grab interaction."""
		obj_id = int(obj_id)
		class_name = self._get_class_name_by_id(obj_id)
		property_value = self._actual_coop_property(obj_id)
		for agent_id in {int(value) for value in agent_ids}:
			self.object_coop_properties.setdefault(agent_id, {})[obj_id] = property_value
			if class_name is not None:
				self.class_coop_properties.setdefault(agent_id, {})[class_name] = property_value
			if property_value == 'cooperative':
				self.known_coop_objects.setdefault(agent_id, set()).add(obj_id)
				if class_name is not None:
					self.known_coop_classes.setdefault(agent_id, set()).add(class_name)
		return property_value

	# ==================== Coop: Action Helpers ====================
	# Purpose: Validate and rewrite coop_grab / coop_put actions before script conversion.
	def _replace_action_name(self, action_string, new_action_name):
		if action_string is None:
			return None
		return re.sub(r'^\[[^\]]+\]', '[{}]'.format(new_action_name), action_string)

	def _block_action(self, action_dict, agent_id, reason=None, details=None):
		action_dict[agent_id] = None
		if reason is not None:
			self._mark_action_invalid(agent_id, reason, details=details)

	def _get_action_object_id(self, action_string):
		if action_string is None:
			return None
		try:
			return utils.get_id(action_string)
		except Exception:
			return None

	def _get_action_target_id(self, action_string):
		if action_string is None:
			return None
		object_ids = re.findall(r'\((-?\d+)\)', action_string)
		if len(object_ids) < 2:
			return None
		return int(object_ids[-1])

	def _mark_implicit_action_conversion(self, agent_ids, conversion_type):
		for agent_id in agent_ids:
			entry = self.current_action_feedback.get(agent_id)
			if entry is None:
				continue
			entry['implicit_conversion'] = True
			entry['conversion_type'] = conversion_type

	def _apply_implicit_coop_action_conversion(self, action_dict, actions):
		"""Rewrite matching ordinary interaction intents to internal coop actions."""
		if getattr(self, 'coop_action_interface', 'implicit_pairing') != 'implicit_pairing':
			return False
		if self.num_agents != 2 or len(actions) != 2:
			return False
		if any(action_dict.get(agent_id) is None for agent_id in range(self.num_agents)):
			return False

		parsed = []
		for agent_id in range(self.num_agents):
			item, reason = action_protocol.parse_physical_action(action_dict.get(agent_id))
			if reason is not None:
				return False
			parsed.append(item)

		if all(item.kind == 'grab' for item in parsed):
			if len({int(item.object_id) for item in parsed}) != 1:
				return False
			for agent_id in range(self.num_agents):
				action_dict[agent_id] = self._replace_action_name(action_dict[agent_id], 'coop_grab')
				actions[agent_id] = 'coop_grab'
			self._mark_implicit_action_conversion(range(self.num_agents), 'paired_grab_to_coop_grab')
			return True

		carrying = getattr(self, 'coop_carrying', None)
		if not isinstance(carrying, dict) or not carrying.get('active', False):
			return False
		if len({item.kind for item in parsed}) != 1 or parsed[0].kind not in {'putin', 'putback'}:
			return False
		if len({int(item.object_id) for item in parsed}) != 1:
			return False
		if len({int(item.target_id) for item in parsed}) != 1:
			return False
		if int(parsed[0].object_id) != int(carrying.get('object_id', -1)):
			return False

		coop_action = 'coop_{}'.format(parsed[0].kind)
		for agent_id in range(self.num_agents):
			action_dict[agent_id] = self._replace_action_name(action_dict[agent_id], coop_action)
			actions[agent_id] = coop_action
		self._mark_implicit_action_conversion(
			range(self.num_agents),
			'paired_{}_to_{}'.format(parsed[0].kind, coop_action),
		)
		return True

	def _paired_coop_event_type(self, action_names):
		if action_names and all(name == 'coop_grab' for name in action_names):
			return 'coop_grab_blocked'
		if action_names and all(name in ['coop_putin', 'coop_putback'] for name in action_names):
			return 'coop_put_blocked'
		return 'coop_action_blocked'

	def _apply_coop_pair_precheck(self, action_dict, actions):
		if not self._coop_enabled():
			return False
		coop_names = {'coop_grab', 'coop_putin', 'coop_putback'}
		coop_agents = [
			agent_id
			for agent_id, action_name in enumerate(actions)
			if action_name in coop_names
		]
		if not coop_agents:
			return False

		action_names = [actions[agent_id] for agent_id in coop_agents]
		event_type = self._paired_coop_event_type(action_names)
		reason = None
		if len(coop_agents) != self.num_agents:
			reason = 'coop_action_requires_pair'
		elif len(set(action_names)) != 1:
			reason = 'mismatched_coop_action'
		else:
			action_name = action_names[0]
			object_ids = [
				self._get_action_object_id(action_dict.get(agent_id))
				for agent_id in coop_agents
			]
			if any(object_id is None for object_id in object_ids):
				reason = 'missing_object'
			elif len(set(object_ids)) != 1:
				if action_name == 'coop_grab':
					reason = 'coop_grab_target_mismatch'
				else:
					reason = 'coop_put_object_mismatch'
			elif action_name in ['coop_putin', 'coop_putback']:
				target_ids = [
					self._get_action_target_id(action_dict.get(agent_id))
					for agent_id in coop_agents
				]
				if any(target_id is None for target_id in target_ids):
					reason = 'missing_target'
				elif len(set(target_ids)) != 1:
					reason = 'coop_put_target_mismatch'

		if reason is None:
			return False

		for agent_id in coop_agents:
			self._block_action(action_dict, agent_id, reason=reason)
			actions[agent_id] = None
		self._add_coop_event({
			'type': event_type,
			'agents': coop_agents,
			'action_names': action_names,
			'reason': reason,
		})
		return True

	def _apply_open_rules(self, action_dict, actions):
		changed = False
		valid_by_target = {}
		for agent_id, action_name in enumerate(actions):
			if action_name != 'open' or action_dict.get(agent_id) is None:
				continue
			parsed, parse_reason = action_protocol.parse_physical_action(action_dict.get(agent_id))
			if parse_reason is not None:
				self._block_action(action_dict, agent_id, reason=parse_reason)
				actions[agent_id] = None
				changed = True
				continue
			reason, details = action_protocol.validate_open(self, agent_id, parsed)
			if reason is not None:
				self._block_action(action_dict, agent_id, reason=reason, details=details)
				actions[agent_id] = None
				changed = True
				continue
			valid_by_target.setdefault(parsed.object_id, []).append(agent_id)

		# Unity executes scripts serially. Resolve a same-target open conflict before
		# rendering so the second agent receives a stable, explicit result.
		for target_id, agent_ids in valid_by_target.items():
			if len(agent_ids) < 2:
				continue
			winner = min(
				agent_ids,
				key=lambda aid: (
					self._distance_between_nodes(aid + 1, target_id)
					if self._distance_between_nodes(aid + 1, target_id) is not None
					else float('inf'),
					aid,
				),
			)
			for agent_id in agent_ids:
				if agent_id == winner:
					continue
				self._block_action(action_dict, agent_id)
				actions[agent_id] = None
				self._set_action_feedback(
					agent_id,
					valid=True,
					status='conflict_waiting',
					details={'executing_agent': winner, 'target_id': int(target_id)},
				)
				changed = True
		return changed

	def _agent_held_objects(self, agent_id):
		char_id = int(agent_id) + 1
		held_objects = []
		for edge in self.get_graph().get('edges', []):
			if edge.get('from_id') != char_id:
				continue
			if edge.get('relation_type') not in ['HOLDS_RH', 'HOLDS_LH']:
				continue
			object_id = edge.get('to_id')
			held_objects.append({
				'object_id': object_id,
				'object_class': self._get_class_name_by_id(object_id),
				'relation': edge.get('relation_type'),
			})
		return held_objects

	def _agent_holds_object(self, agent_id, object_id):
		return any(
			int(item.get('object_id')) == int(object_id)
			for item in self._agent_held_objects(agent_id)
		)

	def _find_agent_holding_object(self, object_id):
		for agent_id in range(self.num_agents):
			if self._agent_holds_object(agent_id, object_id):
				return agent_id
		return None

	def _agent_active_coop_carrying(self, agent_id):
		if not self._coop_enabled() or self.coop_carrying is None:
			return False
		if not self.coop_carrying.get('active', False):
			return False
		agent_id = int(agent_id)
		coop_agents = {
			self.coop_carrying.get('holder_agent'),
			self.coop_carrying.get('partner_agent'),
		}
		return agent_id in {int(aid) for aid in coop_agents if aid is not None}

	def _agent_grab_occupancy(self, agent_id):
		held_objects = self._agent_held_objects(agent_id)
		if len(held_objects) > 0:
			return {
				'blocked': True,
				'reason': 'agent_already_holding_object',
				'held_objects': held_objects,
			}
		if self._agent_active_coop_carrying(agent_id):
			return {
				'blocked': True,
				'reason': 'agent_active_coop_carrying',
				'held_objects': [],
			}
		return {
			'blocked': False,
			'reason': 'free',
			'held_objects': [],
		}

	def _enforce_active_coop_carrying_constraint(self, action_dict_tobe_changed, stage):
		if not self._coop_enabled() or self.coop_carrying is None:
			return True
		if not self.coop_carrying.get('active', False):
			return True

		holder_agent = self.coop_carrying['holder_agent']
		partner_agent = self.coop_carrying['partner_agent']
		is_close, details = self._agents_close_for_active_coop_carrying()
		if is_close:
			previous_status = self.coop_carrying.get('status', 'carrying')
			self.coop_carrying['status'] = 'carrying'
			self.coop_carrying['last_valid_step'] = self.steps
			self.coop_carrying['last_distance_check'] = details
			if details.get('distance_level') == 'soft':
				self._add_coop_event({
					'type': 'coop_carrying_distance_soft',
					'stage': stage,
					'object_id': self.coop_carrying.get('object_id'),
					'object_class': self.coop_carrying.get('object_class'),
					'holder_agent': holder_agent,
					'partner_agent': partner_agent,
					'reason': 'coop_carrying_agents_too_far_soft',
					'status': 'soft',
					'distance_check': details,
				})
			if previous_status == 'paused_distance':
				self._add_coop_event({
					'type': 'coop_carrying_resumed',
					'stage': stage,
					'object_id': self.coop_carrying.get('object_id'),
					'object_class': self.coop_carrying.get('object_class'),
					'holder_agent': holder_agent,
					'partner_agent': partner_agent,
					'distance_check': details,
				})
			return True

		blocked_action = action_dict_tobe_changed.get(holder_agent)
		if blocked_action is not None:
			self._block_action(
				action_dict_tobe_changed,
				holder_agent,
				reason='coop_carrying_agents_too_far_hard',
				details=details,
			)
		self.coop_carrying['status'] = 'paused_distance'
		self.coop_carrying['last_violation_step'] = self.steps
		self.coop_carrying['last_violation'] = details
		self.coop_carrying['distance_violations'] = self.coop_carrying.get('distance_violations', 0) + 1
		self._add_coop_event({
			'type': 'coop_carrying_distance_violation',
			'stage': stage,
			'object_id': self.coop_carrying.get('object_id'),
			'object_class': self.coop_carrying.get('object_class'),
			'holder_agent': holder_agent,
			'partner_agent': partner_agent,
			'blocked_agent': holder_agent,
			'blocked_action': blocked_action,
			'reason': 'coop_carrying_agents_too_far_hard',
			'status': 'hard',
			'distance_check': details,
			'distance_violations': self.coop_carrying['distance_violations'],
		})
		return False

	def _coop_grab_assignment(self, obj_id):
		if self._get_node_from_current_graph(obj_id) is None:
			return None, 'object_not_found', None

		close_agents = []
		for agent_id in range(self.num_agents):
			char_id = agent_id + 1
			distance = self._distance_between_nodes(char_id, int(obj_id))
			if self._agent_close_to_object(agent_id, obj_id):
				close_agents.append((agent_id, distance if distance is not None else float('inf')))
		if not close_agents:
			return None, 'coop_grab_too_far', {
				'object_id': int(obj_id),
				'close_to_object_agents': [],
			}

		holder_agent = sorted(close_agents, key=lambda item: item[1])[0][0]
		partner_agent = 1 - holder_agent
		agents_close, distance_details = self._agents_close_for_coop_interaction(
			holder_agent,
			partner_agent,
		)
		partner_close_to_object = self._agent_close_to_object(partner_agent, obj_id)
		if not (agents_close or partner_close_to_object):
			details = copy.deepcopy(distance_details)
			details.update({
				'object_id': int(obj_id),
				'holder_agent': holder_agent,
				'partner_agent': partner_agent,
				'holder_close_to_object': True,
				'partner_close_to_object': False,
				'partner_close_to_holder': False,
			})
			return None, 'coop_grab_agents_too_far', details
		return {
			'holder_agent': holder_agent,
			'partner_agent': partner_agent,
			'distance_check': distance_details,
			'object_distance_checks': {
				agent_id: distance
				for agent_id, distance in close_agents
			},
			'partner_close_to_object': partner_close_to_object,
			'partner_close_to_holder': agents_close,
		}, 'ok', None

	def _build_coop_carrying_state(self, pending_grab):
		return {
			'object_id': int(pending_grab.get('object_id')),
			'object_class': pending_grab.get('object_class'),
			'holder_agent': int(pending_grab.get('holder_agent')),
			'partner_agent': int(pending_grab.get('partner_agent')),
			'active': True,
			'status': 'carrying',
			'started_step': pending_grab.get('started_step', self.steps),
			'last_valid_step': self.steps,
			'distance_violations': 0,
			'last_violation': None,
			'carry_distance_threshold': self._coop_carry_distance_threshold(),
			'constraint': copy.deepcopy(pending_grab.get('constraint')),
		}

	def _confirm_coop_pending_grab(self, pending_grab, render_success, render_failure_reasons):
		if not self._coop_enabled() or pending_grab is None:
			return

		object_id = int(pending_grab.get('object_id'))
		pending_holder_agent = int(pending_grab.get('holder_agent'))
		pending_partner_agent = int(pending_grab.get('partner_agent'))
		actual_holder_agent = self._find_agent_holding_object(object_id)
		property_value = pending_grab.get('coop_property', 'cooperative')
		if actual_holder_agent is None:
			failure_reason = 'coop_grab_failed' if property_value == 'cooperative' else 'grab_conflict_failed'
			failure_details = {
				'render_success': bool(render_success),
				'render_failure_reasons': copy.deepcopy(render_failure_reasons),
			}
			for agent_id in [pending_holder_agent, pending_partner_agent]:
				self._set_action_feedback(agent_id, valid=False, status='failed', reason=failure_reason, details=failure_details)
			self._add_coop_event({
				'type': 'coop_grab_cancelled' if property_value == 'cooperative' else 'ordinary_grab_conflict_failed',
				'object_id': object_id,
				'object_class': pending_grab.get('object_class'),
				'holder_agent': pending_holder_agent,
				'partner_agent': pending_partner_agent,
				'reason': 'no_agent_holding_object_after_grab',
				'coop_property': property_value,
			})
			return

		confirmed = copy.deepcopy(pending_grab)
		confirmed['holder_agent'] = int(actual_holder_agent)
		confirmed['partner_agent'] = 1 - int(actual_holder_agent) if self.num_agents == 2 else pending_partner_agent
		if property_value == 'cooperative':
			self.coop_carrying = self._build_coop_carrying_state(confirmed)
		for agent_id in [pending_holder_agent, pending_partner_agent]:
			self._set_action_feedback(
				agent_id,
				valid=True,
				status='success' if agent_id == actual_holder_agent else pending_grab.get('partner_status', 'coop_waiting'),
				details={
					'executing_agent': int(actual_holder_agent),
					'self_executed': agent_id == actual_holder_agent,
					'object_id': object_id,
					'object_class': pending_grab.get('object_class'),
					'coop_property': property_value,
				},
			)
		self._add_coop_event({
			'type': 'coop_grab_confirmed' if property_value == 'cooperative' else 'ordinary_grab_fallback_confirmed',
			'object_id': object_id,
			'object_class': pending_grab.get('object_class'),
			'holder_agent': int(actual_holder_agent),
			'partner_agent': confirmed.get('partner_agent'),
			'coop_property': property_value,
			'holder_holds_object': True,
		})

	def _apply_coop_grab_rules(self, action_dict_tobe_changed, actions):
		changed = False
		pending_grab = None
		processed = set()

		# Paired coop_grab is validated jointly. One participant must be CLOSE to
		# the object; the other must be CLOSE to that participant or the object,
		# regardless of the object's hidden property.
		coop_agents = [aid for aid, name in enumerate(actions) if name == 'coop_grab']
		if coop_agents:
			if len(coop_agents) == self.num_agents and all(action_dict_tobe_changed.get(aid) is not None for aid in coop_agents):
				parsed, _ = action_protocol.parse_physical_action(action_dict_tobe_changed[coop_agents[0]])
				reason, details = action_protocol.validate_coop_grab(self, coop_agents, parsed)
				if reason is not None:
					for aid in coop_agents:
						self._block_action(action_dict_tobe_changed, aid, reason=reason, details=details)
						actions[aid] = None
					changed = True
					self._add_coop_event({'type': 'coop_grab_blocked', 'agents': coop_agents, 'object_id': parsed.object_id, 'object_class': parsed.object_class, 'reason': reason})
				else:
					assignment, assignment_reason, assignment_details = self._coop_grab_assignment(parsed.object_id)
					if assignment is None:
						for aid in coop_agents:
							self._block_action(
								action_dict_tobe_changed,
								aid,
								reason=assignment_reason,
								details=assignment_details,
							)
							actions[aid] = None
						changed = True
					else:
						property_value = self._discover_grab_property(coop_agents, parsed.object_id)
						holder_agent = assignment['holder_agent']
						partner_agent = assignment['partner_agent']
						action_dict_tobe_changed[holder_agent] = self._replace_action_name(action_dict_tobe_changed[holder_agent], 'grab')
						action_dict_tobe_changed[partner_agent] = None
						partner_status = 'coop_waiting' if property_value == 'cooperative' else 'conflict_waiting'
						self._set_action_feedback(partner_agent, valid=True, status=partner_status, details={'executing_agent': holder_agent, 'self_executed': False, 'object_id': parsed.object_id, 'object_class': parsed.object_class, 'coop_property': property_value})
						self._set_action_feedback(holder_agent, valid=True, status='accepted', details={'executing_agent': holder_agent, 'self_executed': True, 'object_id': parsed.object_id, 'object_class': parsed.object_class, 'coop_property': property_value})
						changed = True
						pending_grab = dict(assignment, object_id=int(parsed.object_id), object_class=parsed.object_class, started_step=self.steps, constraint=copy.deepcopy(self._get_coop_constraint_for_object(parsed.object_id)), coop_property=property_value, partner_status=partner_status)
						self._add_coop_event({'type': 'coop_grab_pending' if property_value == 'cooperative' else 'ordinary_grab_fallback_pending', 'agents': coop_agents, 'object_id': parsed.object_id, 'object_class': parsed.object_class, 'holder_agent': holder_agent, 'partner_agent': partner_agent, 'coop_property': property_value})
			processed.update(coop_agents)

		# Validate all ordinary grabs before resolving a same-object conflict.
		valid_by_object = {}
		for agent_id, action_name in enumerate(actions):
			if action_name != 'grab' or agent_id in processed or action_dict_tobe_changed.get(agent_id) is None:
				continue
			parsed, parse_reason = action_protocol.parse_physical_action(action_dict_tobe_changed[agent_id])
			if parse_reason is not None:
				reason, details = parse_reason, None
			else:
				reason, details = action_protocol.validate_grab(self, agent_id, parsed)
			if reason is not None:
				self._block_action(action_dict_tobe_changed, agent_id, reason=reason, details=details)
				actions[agent_id] = None
				changed = True
				continue
			property_value = self._discover_grab_property([agent_id], parsed.object_id)
			if property_value == 'cooperative':
				self._block_action(action_dict_tobe_changed, agent_id, reason='requires_two_agent_cooperation', details={'object_id': parsed.object_id, 'object_class': parsed.object_class, 'coop_property': property_value})
				actions[agent_id] = None
				changed = True
				self._add_coop_event({'type': 'solo_grab_blocked', 'agent': agent_id, 'object_id': parsed.object_id, 'object_class': parsed.object_class, 'reason': 'requires_two_agent_cooperation', 'coop_property': property_value})
				continue
			valid_by_object.setdefault(parsed.object_id, []).append((agent_id, parsed))

		for object_id, entries in valid_by_object.items():
			if len(entries) < 2:
				continue
			winner, parsed = min(entries, key=lambda item: (self._distance_between_nodes(item[0] + 1, object_id) if self._distance_between_nodes(item[0] + 1, object_id) is not None else float('inf'), item[0]))
			losers = [aid for aid, _ in entries if aid != winner]
			for aid in losers:
				self._block_action(action_dict_tobe_changed, aid)
				actions[aid] = None
				self._set_action_feedback(aid, valid=True, status='conflict_waiting', details={'executing_agent': winner, 'self_executed': False, 'object_id': int(object_id), 'object_class': parsed.object_class, 'coop_property': 'ordinary'})
			changed = True
			pending_grab = {'object_id': int(object_id), 'object_class': parsed.object_class, 'holder_agent': winner, 'partner_agent': losers[0], 'started_step': self.steps, 'coop_property': 'ordinary', 'partner_status': 'conflict_waiting'}
			self._add_coop_event({'type': 'ordinary_grab_conflict_resolved', 'object_id': int(object_id), 'object_class': parsed.object_class, 'holder_agent': winner, 'partner_agent': losers[0], 'coop_property': 'ordinary'})

		return changed, pending_grab

	def _apply_coop_put_rules(self, action_dict_tobe_changed, actions):
		changed = False
		pending_finish = None
		coop_put_actions = {'coop_putin': 'putin', 'coop_putback': 'putback'}
		processed_coop_agents = set()

		coop_agents = [aid for aid, name in enumerate(actions) if name in coop_put_actions]
		if coop_agents and len(coop_agents) == self.num_agents and all(action_dict_tobe_changed.get(aid) is not None for aid in coop_agents):
			processed_coop_agents.update(coop_agents)
			parsed, parse_reason = action_protocol.parse_physical_action(action_dict_tobe_changed[coop_agents[0]])
			if parse_reason is None:
				reason, details = action_protocol.validate_coop_put(self, parsed)
			else:
				reason, details = parse_reason, None
			if reason is not None:
				for aid in coop_agents:
					self._block_action(action_dict_tobe_changed, aid, reason=reason, details=details)
					actions[aid] = None
				changed = True
				self._add_coop_event({'type': 'coop_put_blocked', 'agents': coop_agents, 'object_id': parsed.object_id if parsed else None, 'object_class': parsed.object_class if parsed else None, 'target_id': parsed.target_id if parsed else None, 'reason': reason})
			else:
				holder_agent = self.coop_carrying['holder_agent']
				partner_agent = self.coop_carrying['partner_agent']
				real_action_name = coop_put_actions[parsed.kind]
				action_dict_tobe_changed[holder_agent] = self._replace_action_name(
					action_dict_tobe_changed[holder_agent],
					real_action_name,
				)
				actions[holder_agent] = real_action_name
				self._block_action(action_dict_tobe_changed, partner_agent)
				actions[partner_agent] = None
				feedback_details = {
					'object_id': parsed.object_id,
					'object_class': parsed.object_class,
					'target_id': parsed.target_id,
					'target_class': parsed.target_class,
					'commit_mode': 'physical_holder_execution',
					'executing_agent': holder_agent,
				}
				self._set_action_feedback(
					holder_agent,
					valid=True,
					status='accepted',
					details=dict(feedback_details, self_executed=True),
				)
				self._set_action_feedback(
					partner_agent,
					valid=True,
					status='coop_waiting',
					details=dict(feedback_details, self_executed=False),
				)
				changed = True
				pending_finish = {
					'object_id': int(parsed.object_id),
					'object_class': parsed.object_class,
					'target_id': int(parsed.target_id),
					'target_class': parsed.target_class,
					'real_action_name': real_action_name,
					'commit_mode': 'physical_holder_execution',
					'holder_agent': holder_agent,
					'partner_agent': partner_agent,
				}
				self._add_coop_event(dict({'type': 'coop_put_accepted'}, **pending_finish))

		for agent_id, action_name in enumerate(actions):
			if (
				agent_id in processed_coop_agents
				or action_name not in ['putin', 'putback']
				or action_dict_tobe_changed.get(agent_id) is None
			):
				continue
			parsed, parse_reason = action_protocol.parse_physical_action(action_dict_tobe_changed[agent_id])
			if parse_reason is None:
				reason, details = action_protocol.validate_put(self, agent_id, parsed)
			else:
				reason, details = parse_reason, None
			if reason is None and self._is_coop_required(parsed.object_id):
				reason = 'requires_active_coop_put'
				details = {'object_id': parsed.object_id, 'object_class': parsed.object_class, 'coop_property': 'cooperative'}
			if reason is not None:
				self._block_action(action_dict_tobe_changed, agent_id, reason=reason, details=details)
				actions[agent_id] = None
				changed = True
		return changed, pending_finish

	# ==================== Coop: Physical Put Confirmation ====================
	# Purpose: Finish active carrying only after the holder's real put succeeds.
	def _commit_coop_pending_update(self, pending_finish, execution_success):
		if not self._coop_enabled() or pending_finish is None:
			return
		if not execution_success:
			failure_details = {
				'object_id': int(pending_finish.get('object_id')),
				'target_id': int(pending_finish.get('target_id')),
				'holder_agent': pending_finish.get('holder_agent'),
				'partner_agent': pending_finish.get('partner_agent'),
				'commit_mode': 'physical_holder_execution',
			}
			for agent_id in [
				pending_finish.get('holder_agent'),
				pending_finish.get('partner_agent'),
			]:
				if agent_id is not None:
					self._set_action_feedback(
						agent_id,
						valid=False,
						status='failed',
						reason='coop_put_failed',
						details=failure_details,
					)
			self._add_coop_event({
				'type': 'coop_pending_update_cancelled',
				'reason': 'physical_put_failed',
				'pending_finish': copy.deepcopy(pending_finish),
			})
			return

		self._add_coop_event({
			'type': 'coop_put_relation_committed',
			'object_id': int(pending_finish.get('object_id')),
			'target_id': int(pending_finish.get('target_id')),
			'relation': (
				'INSIDE'
				if pending_finish.get('real_action_name') == 'putin'
				else 'ON'
			),
			'commit_mode': 'physical_holder_execution',
			'holder_agent': pending_finish.get('holder_agent'),
			'partner_agent': pending_finish.get('partner_agent'),
		})
		self._add_coop_event({
			'type': 'coop_carrying_finished',
			'finish': copy.deepcopy(pending_finish),
			'previous_coop_carrying': copy.deepcopy(self.coop_carrying),
		})
		self.coop_carrying = None


	# ==================== Coop: Info Helpers ====================
	# Purpose: Record structured env-level coop events and expose step info.
	def _add_coop_event(self, event):
		event = copy.deepcopy(event)
		event['step'] = self.steps
		self.coop_events.append(event)
		self.coop_step_events.append(event)
		self.last_coop_event = event

	def _build_coop_info(self):
		return {
			'enable_coop_carry': self._coop_enabled(),
			'coop_constraints': copy.deepcopy(self.coop_constraints),
			'known_coop_objects': {
				i: sorted(self.known_coop_objects.get(i, set()))
				for i in range(self.num_agents)
			},
			'known_coop_classes': {
				i: sorted(self.known_coop_classes.get(i, set()))
				for i in range(self.num_agents)
			},
			'coop_carrying': copy.deepcopy(self.coop_carrying),
			'coop_events': copy.deepcopy(self.coop_events),
			'coop_step_events': copy.deepcopy(self.coop_step_events),
			'action_feedback': copy.deepcopy(self.last_action_feedback),
			'last_coop_event': copy.deepcopy(self.last_coop_event),
		}


	def _coop_event_relevant_to_agent(self, event, agent_id):
		if not isinstance(event, dict):
			return False
		agent_id = int(agent_id)
		if agent_id in {int(aid) for aid in event.get('agents', []) if aid is not None}:
			return True
		for key in ['agent', 'holder_agent', 'partner_agent', 'blocked_agent']:
			value = event.get(key)
			if value is not None and int(value) == agent_id:
				return True
		for key in ['finish', 'previous_coop_carrying', 'pending_finish', 'pending_grab']:
			value = event.get(key)
			if self._coop_event_relevant_to_agent(value, agent_id):
				return True
		return False

	def _public_coop_event_for_agent(self, event, agent_id):
		if not self._coop_event_relevant_to_agent(event, agent_id):
			return None
		result = {}
		for key in [
			'type',
			'step',
				'object_id',
				'object_class',
				'target_id',
				'relation',
				'reason',
			'status',
			'holder_agent',
			'partner_agent',
			'blocked_agent',
			'executing_agent',
			'coop_property',
		]:
			if key in event:
				result[key] = copy.deepcopy(event[key])
		distance_check = event.get('distance_check')
		if isinstance(distance_check, dict):
			for key in ['distance', 'soft_threshold', 'hard_threshold', 'distance_level']:
				if key in distance_check:
					result[key] = copy.deepcopy(distance_check[key])
		if 'target_id' in result:
			result['target_class'] = self._get_class_name_by_id(result['target_id'])
		return result

	def _build_public_active_coop_info(self, agent_id):
		if not isinstance(self.coop_carrying, dict) or not self.coop_carrying.get('active', False):
			return {'active': False}

		holder_agent = self.coop_carrying['holder_agent']
		partner_agent = self.coop_carrying['partner_agent']
		if int(agent_id) not in {holder_agent, partner_agent}:
			return {'active': False}

		_, distance_details = self._agents_close_for_active_coop_carrying()
		holder_pos = self._get_node_position_from_current_graph(int(holder_agent) + 1)
		partner_pos = self._get_node_position_from_current_graph(int(partner_agent) + 1)
		public_events = self.last_coop_events_by_agent.get(int(agent_id), [])
		distance_feedback = None
		for event in reversed(public_events):
			if 'distance' in str(event.get('type', '')) or event.get('type') == 'coop_carrying_resumed':
				distance_feedback = copy.deepcopy(event)
				break
		return {
			'active': True,
			'status': self.coop_carrying.get('status'),
			'object_id': self.coop_carrying.get('object_id'),
			'object_class': self.coop_carrying.get('object_class'),
			'holder_agent': holder_agent,
			'supporter_agent': partner_agent,
			'self_role': 'holder' if int(agent_id) == holder_agent else 'supporter',
			'role': 'holder' if int(agent_id) == holder_agent else 'support',
			'partner_agent': partner_agent if int(agent_id) == holder_agent else holder_agent,
			'holder_pos': holder_pos.tolist() if holder_pos is not None else None,
			'supporter_pos': partner_pos.tolist() if partner_pos is not None else None,
			'distance': distance_details.get('distance'),
			'soft_threshold': distance_details.get('soft_threshold'),
			'hard_threshold': distance_details.get('hard_threshold'),
			'distance_level': distance_details.get('distance_level'),
			'feedback': distance_feedback,
			'started_step': self.coop_carrying.get('started_step'),
		}

	def _public_coop_properties(self, agent_id):
		agent_id = int(agent_id)
		known_objects = copy.deepcopy(self.object_coop_properties.get(agent_id, {}))
		known_classes = copy.deepcopy(self.class_coop_properties.get(agent_id, {}))
		graph = self.full_graph if isinstance(self.full_graph, dict) else self.get_graph()
		try:
			visible = utils_env.get_visible_nodes(graph, agent_id=agent_id + 1)
		except Exception:
			visible = {'nodes': []}
		object_properties = {}
		class_properties = dict(known_classes)
		for node in visible.get('nodes', []):
			if 'GRABBABLE' not in {str(value).upper() for value in node.get('properties', [])}:
				continue
			object_id = int(node.get('id'))
			class_name = node.get('class_name')
			object_properties[object_id] = known_objects.get(object_id, known_classes.get(class_name, 'unknown'))
			if class_name is not None:
				class_properties.setdefault(class_name, 'unknown')
		for object_id, property_value in known_objects.items():
			object_properties[int(object_id)] = property_value
		return object_properties, class_properties

	def _build_agent_coop_observation(self, agent_id):
		agent_id = int(agent_id)
		public_events = copy.deepcopy(self.last_coop_events_by_agent.get(agent_id, []))
		last_event = public_events[-1] if public_events else None
		last_feedback = None
		for event in reversed(public_events):
			event_type = str(event.get('type', ''))
			if event_type.endswith(('blocked', 'cancelled', 'failed')) or 'distance' in event_type:
				last_feedback = copy.deepcopy(event)
				break
		action_feedback = copy.deepcopy(self.last_action_feedback.get(agent_id, self._default_action_feedback(agent_id, None, status='reset')))
		object_properties, class_properties = self._public_coop_properties(agent_id)
		return {
			'object_coop_properties': object_properties,
			'class_coop_properties': class_properties,
			'known_coop_objects': sorted(self.known_coop_objects.get(agent_id, set())),
			'known_coop_classes': sorted(self.known_coop_classes.get(agent_id, set())),
			'active_coop_carrying': self._build_public_active_coop_info(agent_id),
			'coop_action_feedback': public_events,
			'last_coop_event': last_event,
			'last_coop_feedback': last_feedback,
			'action_feedback': action_feedback,
			'message_feedback': copy.deepcopy(getattr(self, 'last_message_feedback', {}).get(agent_id, {})),
		}

	def _build_observation_feedback_fields(self, agent_id, coop_observation=None):
		if coop_observation is None:
			coop_observation = self._build_agent_coop_observation(agent_id)
		action_feedback = coop_observation.get('action_feedback') or {}
		return {
			'valid': action_feedback.get('valid', True),
			'action_status': action_feedback.get('action_status', 'reset'),
			'invalid_reason': action_feedback.get('invalid_reason'),
			'action_feedback': copy.deepcopy(action_feedback),
			'message_feedback': copy.deepcopy(coop_observation.get('message_feedback', {})),
			'object_coop_properties': copy.deepcopy(coop_observation.get('object_coop_properties', {})),
			'class_coop_properties': copy.deepcopy(coop_observation.get('class_coop_properties', {})),
			'coop_action_feedback': copy.deepcopy(coop_observation.get('coop_action_feedback', [])),
			'coop_info': copy.deepcopy(coop_observation.get('active_coop_carrying', {'active': False})),
		}

	def _print_env_step_debug(self, graph, physical_action_frame_info):
		if self.enable_physical_action_time:
			joint_info = physical_action_frame_info.get('joint')
			if isinstance(joint_info, dict):
				print(
					'[PAET][calibrated][joint] step={} frames={} seconds={}'.format(
						physical_action_frame_info.get('step'),
						joint_info.get('frames', 0),
						joint_info.get('time', 0.0),
					)
				)
			for agent_id in range(self.num_agents):
				agent_info = (physical_action_frame_info.get('per_agent') or {}).get(agent_id, {})
				if not isinstance(agent_info, dict):
					agent_info = {}
				print(
					'[PAET][calibrated][agent] step={} agent={} submitted={} charged={} '
					'action={} distance={} individual_seconds={} shared_seconds={}'.format(
						physical_action_frame_info.get('step'),
						agent_id,
						agent_info.get('submitted_physical_action', False),
						agent_info.get('charged_physical_attempt', False),
						agent_info.get('action_kind'),
						agent_info.get('distance', 0.0),
						agent_info.get('individual_time', 0.0),
						agent_info.get('time', 0.0),
					)
				)


	def _print_coop_carry_position_debug(self, graph):
		if not self._coop_enabled():
			return

		node_positions = {}
		for node in graph.get('nodes', []):
			if node.get('class_name') != 'character':
				continue
			transform = node.get('obj_transform', {})
			position = transform.get('position')
			if position is not None:
				node_positions[node.get('id')] = position

		for char_id in sorted(node_positions):
			agent_id = char_id - 1 if 1 <= char_id <= self.num_agents else None
			print(
				'[CoopCarry][pos] step={} agent={} char_id={} pos={}'.format(
					self.steps - 1,
					agent_id,
					char_id,
					node_positions[char_id],
				)
			)
	# ==================== End Coop: Helpers ====================

	def _check_action_postconditions(self, effective_actions, render_messages, pending_grab=None):
		"""Verify state-changing ordinary actions after Unity reports success."""
		render_by_agent = {
			item.get('agent_id'): item
			for item in render_messages or []
			if isinstance(item, dict)
		}
		pending_holder = pending_grab.get('holder_agent') if isinstance(pending_grab, dict) else None
		failures = []
		for agent_id, physical_action in effective_actions.items():
			if physical_action is None:
				continue
			render_result = render_by_agent.get(agent_id)
			if render_result is not None and render_result.get('failed_exec', False):
				continue
			parsed, reason = action_protocol.parse_physical_action(physical_action)
			if reason is not None or parsed.kind in ['walktowards', 'turnleft']:
				continue
			passed = True
			expected = None
			if parsed.kind == 'grab':
				if pending_holder is not None and int(agent_id) == int(pending_holder):
					continue
				passed = self._agent_holds_object(agent_id, parsed.object_id)
				expected = 'agent_holds_object'
			elif parsed.kind == 'open':
				node = self._get_node_from_current_graph(parsed.object_id)
				passed = node is not None and 'OPEN' in {str(value).upper() for value in node.get('states', [])}
				expected = 'target_open'
			elif parsed.kind in ['putin', 'putback']:
				relation = 'INSIDE' if parsed.kind == 'putin' else 'ON'
				relation_present = any(
					int(edge.get('from_id', -1)) == int(parsed.object_id)
					and int(edge.get('to_id', -1)) == int(parsed.target_id)
					and str(edge.get('relation_type', '')).upper() == relation
					for edge in self.get_graph().get('edges', [])
				)
				object_released = not self._agent_holds_object(
					agent_id,
					parsed.object_id,
				)
				passed = relation_present and object_released
				expected = '{}_relation_and_object_released'.format(
					relation.lower()
				)
			if not passed:
				details = {'expected_postcondition': expected, 'physical_action': physical_action}
				self._set_action_feedback(agent_id, valid=False, status='failed', reason='postcondition_failed', details=details)
				failures.append({'agent_id': agent_id, 'reason': 'postcondition_failed', 'details': details})
		return failures

	# ==================== Step Log Formatting Helpers ====================
	# Purpose: Print compact per-agent env execution and result logs.
	def _script_entries_by_agent(self, script_list):
		entries = {agent_id: None for agent_id in range(self.num_agents)}
		if not script_list or len(script_list) == 0:
			return entries
		script_text = script_list[0]
		if not script_text:
			return entries
		for index, script in enumerate(script_text.split('|')):
			char_match = re.match(r'<char(\d+)>', script)
			agent_id = int(char_match.group(1)) if char_match else index
			entries[agent_id] = script
		return entries

	def _print_env_exec_log(self, script_list_verbose):
		print('[EnvExec]')
		entries = self._script_entries_by_agent(script_list_verbose)
		for agent_id in range(self.num_agents):
			print('  agent{} script={}'.format(agent_id, entries.get(agent_id)))

	def _render_message_for_agent(self, message, agent_id):
		def compact_message(value):
			if isinstance(value, str):
				return value.strip()
			return value

		if not isinstance(message, dict):
			return compact_message(message)

		for key in (agent_id, str(agent_id)):
			if key in message:
				agent_message = message[key]
				if isinstance(agent_message, dict) and set(agent_message.keys()) == {'message'}:
					return compact_message(agent_message.get('message'))
				return compact_message(agent_message)

		if set(message.keys()) == {'message'}:
			return compact_message(message.get('message'))
		return message

	def _render_success_for_agent(self, success, message, agent_id):
		"""Prefer an explicit per-agent Success in a joint Unity response."""
		if isinstance(message, dict) and (agent_id in message or str(agent_id) in message):
			agent_message = self._render_message_for_agent(message, agent_id)
			status_value = agent_message.get('message') if isinstance(agent_message, dict) else agent_message
			if self._flatten_render_message_text(status_value).strip().lower() == 'success':
				return True
		return bool(success)

	def _print_env_result_log(self, script_list_verbose, render_messages):
		print('[EnvResult]')
		script_entries = self._script_entries_by_agent(script_list_verbose)
		messages_by_agent = {
			message.get('agent_id'): message
			for message in render_messages or []
			if isinstance(message, dict)
		}
		for agent_id in range(self.num_agents):
			message = messages_by_agent.get(agent_id)
			if message is None:
				print(
					'  agent{} executed=False failed_exec=False reasons=[] message=None'.format(
						agent_id
					)
				)
				continue
			print(
				'  agent{} executed={} failed_exec={} reasons={} message={}'.format(
					agent_id,
					script_entries.get(agent_id) is not None,
					message.get('failed_exec', False),
					message.get('failed_exec_reasons', []),
					message.get('message'),
				)
			)
	# ==================== End Step Log Formatting Helpers ====================

	def step(self, action_dict):
		if self.steps > 245:
			print("Warning: too many steps")
		previous_messages = copy.deepcopy(self.message_said)
		if not isinstance(getattr(self, 'navigation_history', None), list):
			self.navigation_history = [[] for _ in range(self.num_agents)]
		envelopes = action_protocol.normalize_action_envelopes(action_dict, self.num_agents)
		requested_envelopes = {
			agent_id: copy.deepcopy(envelopes[agent_id].raw)
			for agent_id in range(self.num_agents)
		}
		requested_physical_actions = {
			agent_id: envelopes[agent_id].physical_action
			for agent_id in range(self.num_agents)
		}
		self._start_action_feedback_step(requested_envelopes, requested_physical_actions)

		action_dict_tobe_changed = {agent_id: None for agent_id in range(self.num_agents)}
		actions = [None for _ in range(self.num_agents)]
		parsed_actions = {}
		saying = [None for _ in range(self.num_agents)]
		message_truncated = [False for _ in range(self.num_agents)]
		say = False

		# Stage 1: strict envelope, formal syntax, class/id, then active-carry whitelist.
		for agent_id in range(self.num_agents):
			envelope = envelopes[agent_id]
			envelope_reason = action_protocol.validate_communication_envelope(
				envelope,
				getattr(self, 'communication_mode', 'exclusive_action'),
			)
			if envelope_reason is not None and envelope_reason != 'physical_message_mutually_exclusive':
				self._mark_action_invalid(agent_id, envelope_reason)
				continue
			physical_action = envelope.physical_action
			if envelope_reason == 'physical_message_mutually_exclusive':
				physical_action = None
				self._mark_action_invalid(
					agent_id,
					envelope_reason,
					details={
						'physical_action_suppressed': True,
						'message_retained': True,
					},
				)
			if envelope.message not in [None, '']:
				saying[agent_id], message_truncated[agent_id] = self._prepare_outgoing_message(envelope.message)
				say = say or saying[agent_id] is not None
				if message_truncated[agent_id]:
					print("Message too long, truncating to {} characters".format(self.communication_max_message_chars))

			parsed, reason = action_protocol.parse_physical_action(physical_action)
			if reason is None:
				reason, details = action_protocol.validate_class_ids(self, parsed)
			else:
				details = None
			if reason is None:
				reason = action_protocol.validate_active_carrying_action(self, parsed)
				if reason == 'in_coop_carrying':
					details = {
						'allowed_actions': sorted(
							action_protocol.active_carry_allowed_actions(self)
						)
					}
			if reason is not None:
				self._mark_action_invalid(agent_id, reason, details=details)
				continue

			parsed = self._apply_navigation_stall_recovery(agent_id, parsed)
			if parsed is None:
				continue
			parsed_actions[agent_id] = parsed
			actions[agent_id] = None if parsed.kind == action_protocol.WAIT else parsed.kind
			action_dict_tobe_changed[agent_id] = action_protocol.format_physical_action(parsed)
			if (
				parsed.kind == action_protocol.WAIT
				and saying[agent_id] is not None
				and envelope_reason is None
			):
				self._set_action_feedback(agent_id, valid=True, status='message')

		self.last_message_feedback = {
			agent_id: {
				'sent': saying[agent_id] is not None,
				'truncated': bool(message_truncated[agent_id]),
				'char_count': len(saying[agent_id]) if saying[agent_id] is not None else 0,

			}
			for agent_id in range(self.num_agents)
		}
		implicit_pairing_changed = self._apply_implicit_coop_action_conversion(
			action_dict_tobe_changed,
			actions,
		)
		action_dict_verbose = copy.deepcopy(action_dict_tobe_changed)
		paired_action_changed = self._apply_coop_pair_precheck(action_dict_tobe_changed, actions)
		open_changed = self._apply_open_rules(action_dict_tobe_changed, actions)
		if implicit_pairing_changed or open_changed or paired_action_changed:
			action_dict_verbose = copy.deepcopy(action_dict_tobe_changed)
		coop_pending_finish = None
		coop_pending_grab = None
		if self._coop_enabled():
			active_carrying_ok = self._enforce_active_coop_carrying_constraint(
				action_dict_tobe_changed,
				stage='before_action'
			)
			coop_grab_changed, coop_pending_grab = self._apply_coop_grab_rules(action_dict_tobe_changed, actions)
			coop_put_changed, coop_pending_finish = self._apply_coop_put_rules(action_dict_tobe_changed, actions)
			if coop_grab_changed or coop_put_changed or not active_carrying_ok:
				action_dict_verbose = copy.deepcopy(action_dict_tobe_changed)

		script_list = utils.convert_action(action_dict_tobe_changed)
		script_list_verbose = utils.convert_action(action_dict_verbose)
		submitted_physical_actions = copy.deepcopy(action_dict_tobe_changed)
		charged_physical_actions = copy.deepcopy(submitted_physical_actions)
		physical_charge_reasons = {
			agent_id: ('unity_submission' if action is not None else None)
			for agent_id, action in submitted_physical_actions.items()
		}
		pre_physical_graph = copy.deepcopy(self.get_graph())
		failed_execution = False
		failed_execution_reasons = []
		failed_execution_details = []
		render_messages = []
		render_outcomes = []
		joint_render_success = None
		self._print_env_exec_log(script_list_verbose)

		if len(script_list[0]) > 0:
			if self.recording_options['recording']:
				raise AssertionError("Formal C-WAH calibrated PAET does not support recording")
			else:
				script_entries = self._script_entries_by_agent(script_list)
				success, message = self.comm.render_script(
					script_list,
					recording=False,
					image_synthesis=[],
					skip_animation=True,
				)
				joint_render_success = bool(success)
				for render_agent_id, agent_script in script_entries.items():
					if agent_script is None:
						continue
					render_outcomes.append({
						'agent_id': render_agent_id,
						'script': agent_script,
						'success': self._render_success_for_agent(success, message, render_agent_id),
						'message': self._render_message_for_agent(message, render_agent_id),
						'raw_message': message,
					})

				for outcome in render_outcomes:
					render_failed, render_failure_reasons = self._classify_render_execution_failure(
						outcome['success'],
						outcome['message'],
					)
					outcome['failed_exec'] = bool(render_failed)
					outcome['failed_exec_reasons'] = copy.deepcopy(render_failure_reasons)
					render_messages.append(outcome)
					if render_failed:
						failed_execution = True
						for reason in render_failure_reasons:
							if reason not in failed_execution_reasons:
								failed_execution_reasons.append(reason)
						failed_execution_details.append({
							'agent_id': outcome['agent_id'],
							'script': outcome['script'],
							'success': outcome['success'],
							'reasons': copy.deepcopy(render_failure_reasons),
							'message': outcome['message'],
							'raw_message': outcome['raw_message'],
						})
					else:
						self.changed_graph = True

		self._print_env_result_log(script_list_verbose, render_messages)

		if self._coop_enabled():
			self._confirm_coop_pending_grab(
				coop_pending_grab,
				render_success=(not failed_execution),
				render_failure_reasons=failed_execution_reasons,
			)

		# A valid solo grasp rejected only by the hidden cooperative property is a
		# real physical probe even though it is blocked before Unity submission.
		for agent_id, feedback in self.current_action_feedback.items():
			if feedback.get('invalid_reason') != 'requires_two_agent_cooperation':
				continue
			parsed = parsed_actions.get(agent_id)
			if parsed is not None:
				charged_physical_actions[agent_id] = action_protocol.format_physical_action(parsed)
				physical_charge_reasons[agent_id] = 'coop_property_probe'

		physical_action_frame_info = self._build_calibrated_physical_action_info(
			submitted_physical_actions,
			charged_physical_actions,
			physical_charge_reasons,
			pre_physical_graph,
			copy.deepcopy(self.get_graph()),
			joint_render_success,
		)

		postcondition_failures = self._check_action_postconditions(
			action_dict_tobe_changed,
			render_messages,
			pending_grab=coop_pending_grab,
		)
		if postcondition_failures:
			failed_execution = True
			if 'postcondition_failed' not in failed_execution_reasons:
				failed_execution_reasons.append('postcondition_failed')
			failed_execution_details.extend(postcondition_failures)

		if self._coop_enabled():
			self._commit_coop_pending_update(
				coop_pending_finish,
				execution_success=(
					joint_render_success is True and not failed_execution
				),
			)

		self._finalize_step_public_feedback(
			action_dict_tobe_changed,
			render_messages,
		)

		# Obtain goal progress before applying benchmark execution-time termination.
		reward, goal_done, info = self.reward()

		graph = self.get_graph()
		self.steps += 1
		execution_time_info = self._build_step_execution_time_info(
			previous_messages,
			saying,
			message_truncated,
			physical_action_frame_info,
		)
		step_embodied_time = float(execution_time_info.get('step_embodied_execution_time') or 0.0)
		previous_elapsed_time = float(getattr(self, 'elapsed_embodied_execution_time', 0.0) or 0.0)
		self.elapsed_embodied_execution_time = previous_elapsed_time + step_embodied_time
		execution_time_info['previous_elapsed_embodied_execution_time'] = previous_elapsed_time
		execution_time_info['elapsed_embodied_execution_time'] = self.elapsed_embodied_execution_time

		max_embodied_time = getattr(self, 'max_embodied_execution_time', None)
		# Benchmark: Only explicitly supplied episode limits participate in termination.
		done, task_success_within_time, self.termination_reason = resolve_episode_termination(
			goal_done, self.elapsed_embodied_execution_time, max_embodied_time,
			self.steps, self.max_episode_length,
		)

#		obs = self.get_observations()
		obs = None
		# already get it in env.get_observations()

		info['finished'] = task_success_within_time
		info['raw_goal_success'] = bool(goal_done)
		info['task_success_within_time'] = task_success_within_time
		info['termination_reason'] = self.termination_reason
		info['elapsed_embodied_execution_time'] = self.elapsed_embodied_execution_time
		info['max_embodied_execution_time'] = max_embodied_time
		info['graph'] = graph
		info['failed_exec'] = failed_execution
		info['failed_exec_reasons'] = failed_execution_reasons
		info['failed_exec_details'] = failed_execution_details
		info['last_render_messages'] = render_messages
		info['action_feedback'] = copy.deepcopy(self.last_action_feedback)
		info['action_status'] = {
			agent_id: feedback.get('action_status')
			for agent_id, feedback in self.last_action_feedback.items()
		}
		info['invalid_reasons'] = {
			agent_id: feedback.get('invalid_reason')
			for agent_id, feedback in self.last_action_feedback.items()
		}
		info['coop_action_feedback'] = {
			agent_id: copy.deepcopy(self.last_coop_events_by_agent.get(agent_id, []))
			for agent_id in range(self.num_agents)
		}
		if self.enable_physical_action_time:
			info['physical_action_frames'] = physical_action_frame_info
		info['communication'] = {
			'step': execution_time_info.get('step'),
			'messages': copy.deepcopy(execution_time_info.get('messages', [])),
			'communication_chars': execution_time_info.get('communication_chars'),
			'message_send_time_agent_sum': execution_time_info.get('message_send_time_agent_sum'),
			'communication_critical_path_overhead': execution_time_info.get('communication_critical_path_overhead'),
			'per_agent': copy.deepcopy(execution_time_info.get('per_agent', {})),
		}
		info['execution_time_step'] = execution_time_info
		satisfied, unsatisfied = utils.check_progress(self.get_graph(), self.goal_spec[0])
		info['progress'] = {'satisfied': satisfied, 'unsatisfied': unsatisfied}
		info['coop'] = self._build_coop_info()
		print('[execution_time] seconds={}'.format(execution_time_info.get('total_execution_time')))
		self._print_env_step_debug(graph, physical_action_frame_info)
		self._print_communication_step_debug(execution_time_info)
		self._print_coop_carry_position_debug(graph)
		
		messages = saying
		self.message_said = saying		

		if len(script_list[0]) == 0 and say == False and self.data_collection == True:
			#stacked, recollect data	
			print("Stacked")
			done = True

		return obs, reward, done, info, messages

	def get_observations(self):
		curr_graph = self.get_graph()
		curr_graph = utils.inside_not_trans(curr_graph)
		self.full_graph = curr_graph
		self.dict_graph = {node['id']: node for node in curr_graph['nodes']}

		self.id_map = []
		dict_observations = {}
		for agent_id in range(self.num_agents):
			obs_type = self.observation_types[agent_id]
			if ('image' in obs_type or self.data_collection):
				image_width, image_height = self.default_image_width, self.default_image_height
				camera_ids = [self.num_static_cameras + agent_id * self.num_camera_per_agent + self.CAMERA_NUM]
				s, seg_inst = self.comm.camera_image(camera_ids, mode='seg_inst', image_width=image_width,
													 image_height=image_height)
				s, inst_colors = self.comm.instance_colors()

				def find_id_by_colors(rgb):
					if rgb in self.cache_id_map.keys():
						return self.cache_id_map[rgb]
					ans_id = []
					for k, v in inst_colors.items():
						if abs(rgb[0] - v[2] * 255) + abs(rgb[1] - v[1] * 255) + abs(rgb[2] - v[0] * 255) < 10:
							if int(k) in self.all_relative_id:
								ans_id.append(int(k))
					ans = ans_id[0] if len(ans_id) == 1 else -1
					assert(len(ans_id) <= 1)
					self.cache_id_map[rgb] = ans
					return ans

				ids = np.empty((len(camera_ids), image_height, image_width))
				inside_info = {}
				for t in range(len(camera_ids)):
					for i in range(image_height):
						for j in range(image_width):
							ids[t, i, j] = find_id_by_colors(tuple(seg_inst[t][i, j]))
							if (ids[t, i, j] != -1):
								if (ids[t, i, j] not in inside_info.keys()):
									inside_info[ids[t, i, j]] = [edge['to_id'] for edge in self.full_graph['edges'] if edge['from_id'] == ids[t, i, j] and edge['relation_type'] == 'INSIDE'][0]
								if (self.dict_graph[inside_info[ids[t, i, j]]]['category'] not in ['Rooms'] and 'CLOSED' in self.dict_graph[inside_info[ids[t, i, j]]]['states']): 
									ids[t, i, j] = -1
				self.id_map.append(ids)
				colorids = np.stack(((ids % 10) * 10, ((ids // 10) % 10) * 10, ((ids // 100) % 10) * 10), axis=3)
				if self.save_image:
					import cv2
					for i in range(len(camera_ids)):
						rot_seg_inst = np.rot90(seg_inst[i], axes = (0, 1))
						rot_colorids = np.rot90(colorids[i], axes = (0, 1))
						cv2.imwrite(os.path.join('saved_images', f"{agent_id}_{self.steps:03}_seg.png"), rot_seg_inst)
						cv2.imwrite(os.path.join('saved_images', f"{agent_id}_{self.steps:03}_seg_reference.png"), rot_colorids)
			dict_observations[agent_id] = self.get_observation(agent_id, obs_type)
			self.location[agent_id].append(dict_observations[agent_id]['location'])
			if self.data_collection:
				def find_data_by_colors(rgb):
					if rgb in self.cache_data_map.keys():
						return self.cache_data_map[rgb]
					ans_id = []
					for k, v in inst_colors.items():
						if abs(rgb[0] - v[2] * 255) + abs(rgb[1] - v[1] * 255) + abs(rgb[2] - v[0] * 255) < 10:
							if (int(k) in self.all_detection_id):
								ans_id.append(int(k))
					ans = ans_id[0] if len(ans_id) == 1 else -1
					assert(len(ans_id) <= 1)
					self.cache_data_map[rgb] = ans
					return ans
				inside_info = {}
				ids_class = np.empty((len(camera_ids), image_height, image_width))
				ids_instance = np.empty((len(camera_ids), image_height, image_width))
				for t in range(len(camera_ids)):
					for i in range(image_height):
						for j in range(image_width):
							ids_class[t, i, j] = find_data_by_colors(tuple(seg_inst[t][i, j]))
							if ids_class[t, i, j] != -1:
								if ids_class[t, i, j] not in inside_info.keys():
									inside_info[ids_class[t, i, j]] = [edge['to_id'] for edge in self.full_graph['edges'] if edge['from_id'] == ids_class[t, i, j] and edge['relation_type'] == 'INSIDE'][0]
								if self.dict_graph[inside_info[ids_class[t, i, j]]]['category'] not in ['Rooms'] and 'CLOSED' in self.dict_graph[inside_info[ids_class[t, i, j]]]['states']:
									ids[t, i, j] = -1
							if ids_class[t, i, j] != -1:
								ids_instance[t, i, j] = ids_class[t, i, j]
								ids_class[t, i, j] = self.detection_name_id_map[self.id_to_name[ids_class[t, i, j]]]
							else:
								ids_instance[t, i, j] = -1
				color_id_instance = np.stack(((ids_instance % 10) * 10, ((ids_instance // 10) % 10) * 10, ((ids_instance // 100) % 10) * 10), axis=3)
				if 'bgr' not in dict_observations[agent_id].keys():
					s, dict_observations[agent_id]['bgr'] = self.comm.camera_image(camera_ids, mode='normal', image_width=image_width,
													 image_height=image_height)
				if os.path.exists(self.data_collection_dir + str(self.global_episode_id) + '_' + str(self.env_id) + '/') == False:
					os.mkdir(self.data_collection_dir + str(self.global_episode_id) + '_' + str(self.env_id) + '/')
				import cv2
				for t in range(len(camera_ids)):
					cv2.imwrite(self.data_collection_dir + str(self.global_episode_id) + '_' + str(self.env_id) + '/' + str(self.steps) + '_' + str(agent_id) + '_ids_instance.png', ids_instance[t])
					cv2.imwrite(self.data_collection_dir + str(self.global_episode_id) + '_' + str(self.env_id) + '/' + str(self.steps) + '_' + str(agent_id) + '_color_ids_instance.png', color_id_instance[t])
					cv2.imwrite(self.data_collection_dir + str(self.global_episode_id) + '_' + str(self.env_id) + '/' + str(self.steps) + '_' + str(agent_id) + '_ids_class.png', ids_class[t])
					cv2.imwrite(self.data_collection_dir + str(self.global_episode_id) + '_' + str(self.env_id) + '/' + str(self.steps) + '_' + str(agent_id) + '_rgb.png', dict_observations[agent_id]['bgr'][t])
		return dict_observations

	def get_agent_location(self, agent_id):
		graph = self.full_graph
		agent_node = \
			[node for node in graph['nodes'] if node['category'] == 'Characters' and node['id'] == agent_id + 1][0]
		return agent_node['obj_transform']['position']

	def bbox_3d_to_id(self, bbox):
		r'''
			Given a bounding box, return the id of the object in the scene
			api for agent to get the id of the object

			return: id and name of the object, if the main object is not relative object, return (-1, None).
		'''
		# deprecate now
		raise NotImplementedError

	def bbox_2d_to_id(self, bbox, agent_id, frame_id = 0, mask = None, threshold = 0.3):
		r'''
			Given a bounding box, return the id of the object in the scene
			api for agent to get the id of the object
			bbox: [x_min, y_min, x_max, y_max]

			return: id and name of the object, if the main object is not relative object, return (-1, None).
		'''
		id_array = copy.deepcopy(self.id_map[agent_id][frame_id, int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])])
		if (mask is not None):
			mask = mask[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]
			total_pixels = np.sum(mask)
			id_array[mask == False] = -1
		else:
			total_pixels = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
		id_array = id_array.reshape(-1)
		id_array = id_array[id_array != -1]
		if len(id_array) == 0:
			return -1, None
		count = np.bincount(id_array.astype(int))
		d = np.argmax(count)
		if (count[d] < total_pixels * threshold):
			return -1, None
		if (d in self.all_relative_id):
			return d, self.id_to_name[d]
		else:
			return -1, None
		
	def id_to_inside_room(self, id):
		r'''
			Given an id, return the id of the room it is inside.
			api for agent to get the id of the room it is inside.

			return: id and name of the room, if the main object is not relative object, return (-1, None).
		'''
		while (id not in self.all_room_id):
			id = [edge['to_id'] for edge in self.full_graph['edges'] if edge['from_id'] == id and edge['relation_type'] == 'INSIDE'][0]
		return id

	def get_properties(self, id):
		r'''
			return 'properties' of the object with id.
		'''
		return [node['properties'] for node in self.full_graph['nodes'] if node['id'] == id][0]
	
	def get_category(self, id):
		r'''
			return 'category' of the object with id.
		'''
		return [node['category'] for node in self.full_graph['nodes'] if node['id'] == id][0]
	
	def get_states(self, id):
		r'''
			return 'states' of the object with id.
		'''
		return [node['states'] for node in self.full_graph['nodes'] if node['id'] == id][0]
	
	def get_class_name(self, id):
		r'''
			return 'states' of the object with id.
		'''
		return [node['class_name'] for node in self.full_graph['nodes'] if node['id'] == id][0]

	def nodes_in_same_room(self, nodes, agent_id = 0):
		same_room_node =  [node for node in nodes if self.id_to_inside_room(node['id']) == self.id_to_inside_room(agent_id)]
		not_inside_node = []
		for node in same_room_node:								
			inside_info = [edge['to_id'] for edge in self.full_graph['edges'] if edge['from_id'] == node['id'] and edge['relation_type'] == 'INSIDE']
			assert (len(inside_info) <= 1)
			if len(inside_info) == 0:
				not_inside_node.append(node)
			elif self.dict_graph[inside_info[0]]['category'] in ['Rooms'] or 'CLOSED' not in self.dict_graph[inside_info[0]]['states']:
				not_inside_node.append(node)
		return not_inside_node

	def agent_env_api(self, agent_id) -> dict:
		# all the api the agent can use in the vision env.
		return {
			'bbox_2d_to_id': partial(self.bbox_2d_to_id, agent_id=agent_id),
			'get_properties': self.get_properties,
			'get_category': self.get_category,
			'get_states': self.get_states,
			'nodes_in_same_room': partial(self.nodes_in_same_room, agent_id = (agent_id + 1)),
			'get_class_name': self.get_class_name,
			'remove_duplicate_graph': self.remove_duplicate_graph,
		}

	def remove_duplicate_graph(self, graph):
		# remove duplicate nodes / edges
		allnode = [node['id'] for node in graph['nodes']]
		simplify_edge = []
		all_edge = []
		for edge in graph['edges']:
			if edge['from_id'] in allnode and (edge['to_id'] in allnode or self.id2node[edge['to_id']]['category'] == 'Rooms'):
				if (edge['from_id'], edge['to_id'], edge['relation_type']) not in all_edge:
					simplify_edge.append(edge)
					all_edge.append((edge['from_id'], edge['to_id'], edge['relation_type']))
		return {
			'nodes': graph['nodes'],
			'edges': simplify_edge,
		}

	def clean_object_relationship(self, graph):
		return {
			'nodes': [],
			'edges': [edge for edge in graph['edges'] if (self.id2node[edge['from_id']]['class_name'] == 'character' or self.id2node[edge['to_id']]['class_name'] == 'character') and edge['relation_type'] not in ['ON', 'BETWEEN', 'INSIDE']],
		}
	def get_observation(self, agent_id, obs_type):
		agent_location = self.get_agent_location(agent_id)

		if obs_type == 'partial':
			# agent 0 has id (0 + 1)
			obs = utils_env.get_visible_nodes(self.full_graph, agent_id=(agent_id + 1))
			if self.save_image:
				assert (
						self.num_static_cameras is not None and self.num_camera_per_agent is not None and self.CAMERA_NUM is not None)
				camera_ids = [self.num_static_cameras-1]
				# camera_ids = [self.num_static_cameras + agent_id * self.num_camera_per_agent + self.CAMERA_NUM]
                # camera_ids = [self.num_static_cameras - 1] has a top-down view
				image_width, image_height = self.default_image_width, self.default_image_height
				# All the types are "normal", "seg_inst", "seg_class", "depth", "flow", "albedo", "illumination", "surf_normals"
				s, bgr_images = self.comm.camera_image(camera_ids, mode='normal', image_width=image_width, image_height=image_height)
				s, depth_images = self.comm.camera_image(camera_ids, mode='depth', image_width=image_width, image_height=image_height)
				s, camera_info = self.comm.camera_data(camera_ids)
				if self.save_image:
					import cv2
					for i in range(len(camera_ids)):
						rot_bgr = np.rot90(bgr_images[i], axes = (0, 1))
						rot_depth = np.rot90(depth_images[i], axes = (0, 1))
						cv2.imwrite(os.path.join('saved_images', f"{agent_id}_{self.steps:03}_img.png"), rot_bgr)
						cv2.imwrite(os.path.join('saved_images', f"{agent_id}_{self.steps:03}_depth.png"), 50 / rot_depth) # for visualization
			return {
				'messages': self.message_said,
				**obs,
				'location': agent_location,
				'coop_observation': self._build_agent_coop_observation(agent_id),
				**self._build_observation_feedback_fields(agent_id),
			}

		elif obs_type == 'full':
			return self.full_graph

		elif obs_type == 'visible':
			# Only objects in the field of viedw of the agent
			raise NotImplementedError

		elif 'image' in obs_type:
			#we have 'full_image' and 'normal_image'
			assert (self.num_static_cameras != None and self.num_camera_per_agent != None and self.CAMERA_NUM != None)
			camera_ids = [self.num_static_cameras + agent_id * self.num_camera_per_agent + self.CAMERA_NUM]
			image_width, image_height = self.default_image_width, self.default_image_height
			# All the types are "normal", "seg_inst", "seg_class", "depth", "flow", "albedo", "illumination", "surf_normals"
			s, bgr_images = self.comm.camera_image(camera_ids, mode='normal', image_width=image_width, image_height=image_height)
			s, depth_images = self.comm.camera_image(camera_ids, mode='depth', image_width=image_width, image_height=image_height)
			s, camera_info = self.comm.camera_data(camera_ids)
			if self.save_image:
				import cv2
				for i in range(len(camera_ids)):
					rot_bgr = np.rot90(bgr_images[i], axes = (0, 1))
					rot_depth = np.rot90(depth_images[i], axes = (0, 1))
					cv2.imwrite(os.path.join('saved_images', f"{agent_id}_{self.steps:03}_img.png"), rot_bgr)
					cv2.imwrite(os.path.join('saved_images', f"{agent_id}_{self.steps:03}_depth.png"), 50 / rot_depth) # for visualization
			if not s:
				pdb.set_trace()
			if not self.gt_seg:
				return {
				'bgr': bgr_images,  # 1 * w * h * 3
				'depth': depth_images,  # 1 * w * h * 3
				'camera_info': camera_info,
				'room_info': self.room_info,
				'location': agent_location,  # [x, y, z]
				'messages': self.message_said,
				'coop_observation': self._build_agent_coop_observation(agent_id),
				**self._build_observation_feedback_fields(agent_id),
				'current_room': self.id_to_inside_room(agent_id + 1),
				**self.agent_env_api(agent_id),
				**self.clean_object_relationship(utils_env.get_visible_nodes(self.full_graph, agent_id=(
						agent_id + 1))),
				}
			else: return {
				'bgr': bgr_images,  # 1 * w * h * 3
				'depth': depth_images,  # 1 * w * h * 3
				'camera_info': camera_info,
				'seg_info': self.id_map[agent_id],
				'room_info': self.room_info,
				'messages': self.message_said,
				'location': agent_location,  # [x, y, z]
				'coop_observation': self._build_agent_coop_observation(agent_id),
				**self._build_observation_feedback_fields(agent_id),
				'current_room': self.id_to_inside_room(agent_id + 1),
				**self.agent_env_api(agent_id),
				**self.clean_object_relationship(utils_env.get_visible_nodes(self.full_graph, agent_id=(
						agent_id + 1))),
			}
		else:
			raise NotImplementedError

		# return updated_graph
