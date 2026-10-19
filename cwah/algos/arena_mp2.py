"""Run two DPGP agents and record the public benchmark trajectory."""

import os
import pickle
import copy
import atexit

try:
    from eval.metrics_forced_coop import build_episode_metrics
except Exception:
    build_episode_metrics = None

class ArenaMP(object):
    def __init__(self, arena_id, environment_fn, agent_fn, record_dir='out', debug=False, reset_max_attempts=None):
        self.agents = []
        self.env_fn = environment_fn
        self.arena_id = arena_id
        self.num_agents = len(agent_fn)
        self.task_goal = None
        self.record_dir = record_dir
        self.debug = debug
        print("Init Env")
        self.env = environment_fn(arena_id)
        for agent_type_fn in agent_fn:
            self.agents.append(agent_type_fn(arena_id, self.env))

        self.reset_max_attempts = reset_max_attempts
        atexit.register(self.close)

    def close(self):
        if self.env is None:
            return
        try:
            self.env.close()
        finally:
            self.env = None

    def get_port(self):
        return self.env.port_number


    def reset(self, task_id=None):
        ob = None
        reset_attempt = 0
        while ob is None:
            reset_attempt += 1
            ob = self.env.reset(task_id=task_id)
            if (
                ob is None
                and self.reset_max_attempts is not None
                and reset_attempt >= int(self.reset_max_attempts)
            ):
                raise RuntimeError(
                    'Environment reset returned None after {} attempts for task {}'.format(
                        reset_attempt, task_id
                    )
                )

        for it, agent in enumerate(self.agents):
            agent.reset(ob[it], self.env.all_containers_name,
                        self.env.all_goal_objects_name, self.env.all_room_name,
                        self.env.room_info, self.env.goal_spec[it])

    def get_actions(self, obs):
        dict_actions, dict_info = {}, {}
        for it, agent in enumerate(self.agents):
            task_goal = self.env.task_goal if self.task_goal is None else self.task_goal
            goal_spec = self.env.get_goal(task_goal[it], self.env.agent_goals[it])
            dict_actions[it], dict_info[it] = agent.get_action(obs[it], goal_spec)
        return dict_actions, dict_info

    def reset_env(self):
        self.env.close()
        self.env = self.env_fn(self.arena_id)


    def step(self):
        obs = self.env.get_observations()
        dict_actions, dict_info = self.get_actions(obs)
        try:
            step_info = self.env.step(dict_actions)
        except Exception:
            print("Exception occurs when performing action: ", dict_actions)
            raise
        return step_info, dict_actions, dict_info

    def run(self):
        self.task_goal = copy.deepcopy(self.env.task_goal)

        saved_info = {'task_id': self.env.task_id,
                      'env_id': self.env.env_id,
                      'task_name': self.env.task_name,
                      'gt_goals': self.env.task_goal[0],
                      'goals': self.task_goal,
                      'initial_progress': copy.deepcopy(getattr(self.env, 'initial_progress', {})),
                      'max_embodied_execution_time': getattr(self.env, 'max_embodied_execution_time', None),
                      'max_episode_length': self.env.max_episode_length,
                      'communication_count_unit': getattr(self.env, 'communication_count_unit', None),
                      'communication_send_chars_per_second': getattr(self.env, 'communication_send_chars_per_second', None),
                      'communication_mode': getattr(self.env, 'communication_mode', None),
                      'communication_max_message_chars': getattr(self.env, 'communication_max_message_chars', None),
                      'minimum_step_frames': getattr(self.env, 'minimum_step_frames', None),
                      'task_success_within_time': False,
                      'termination_reason': None,
                      'action': {0: [], 1: []},
                      'plan': {0: [], 1: []},
                      'method_diagnostics': {0: [], 1: []},
                      'finished': None,
                      'init_unity_graph': self.env.init_graph,
                      'goals_finished': [],
                      'LLM': {0: [], 1: []},
                      'graph': {0: [], 1: []},
                      'progress': [],
                      # ==================== Coop: Env Info Logging ====================
                      # Purpose: Record environment-level cooperative state and execution evidence.
                      'coop': [],
                      'failed_exec': [],
                      'failed_exec_reasons': [],
                      'failed_exec_details': [],
                      'physical_action_frames': [],
                      'action_feedback': [],
                      'action_status': [],
                      'invalid_reasons': [],
                      'coop_action_feedback': [],
                      'communication': [],
                      'execution_time_steps': [],
                      # ==================== End Coop: Env Info Logging ====================
                    }
        success = False
        while True:
            (obs, reward, done, infos, messages), actions, agent_info = self.step()
            success = infos['finished']
            if 'satisfied_goals' in infos:
                saved_info['goals_finished'].append(infos['satisfied_goals'])
            for agent_id, action in actions.items():
                saved_info['action'][agent_id].append(action)
            
            if 'progress' in infos:
                saved_info['progress'].append(infos['progress'])
            # ==================== Coop: Env Info Logging ====================
            # Purpose: Keep per-step cooperation traces for metrics and experiment analysis.
            if 'coop' in infos:
                saved_info['coop'].append(copy.deepcopy(infos['coop']))
            if 'failed_exec' in infos:
                saved_info['failed_exec'].append(infos['failed_exec'])
            if 'failed_exec_reasons' in infos:
                saved_info['failed_exec_reasons'].append(copy.deepcopy(infos['failed_exec_reasons']))
            if 'failed_exec_details' in infos:
                saved_info['failed_exec_details'].append(copy.deepcopy(infos['failed_exec_details']))
            if 'physical_action_frames' in infos:
                saved_info['physical_action_frames'].append(copy.deepcopy(infos['physical_action_frames']))
            if 'communication' in infos:
                saved_info['communication'].append(copy.deepcopy(infos['communication']))
            if 'execution_time_step' in infos:
                saved_info['execution_time_steps'].append(copy.deepcopy(infos['execution_time_step']))
            if 'task_success_within_time' in infos:
                saved_info['task_success_within_time'] = bool(infos['task_success_within_time'])
            if 'termination_reason' in infos:
                saved_info['termination_reason'] = infos['termination_reason']
            if 'action_feedback' in infos:
                saved_info['action_feedback'].append(copy.deepcopy(infos['action_feedback']))
            if 'action_status' in infos:
                saved_info['action_status'].append(copy.deepcopy(infos['action_status']))
            if 'invalid_reasons' in infos:
                saved_info['invalid_reasons'].append(copy.deepcopy(infos['invalid_reasons']))
            if 'coop_action_feedback' in infos:
                saved_info['coop_action_feedback'].append(copy.deepcopy(infos['coop_action_feedback']))
            # ==================== End Coop: Env Info Logging ====================
            for agent_id, info in agent_info.items():
                if 'plan' in info:
                    saved_info['plan'][agent_id].append(copy.deepcopy(info['plan']))
                # Optional generic diagnostics stay aligned with control steps.
                saved_info['method_diagnostics'][agent_id].append(copy.deepcopy(info.get('method_diagnostics')))
                if 'LLM' in info:
                    saved_info['LLM'][agent_id].append(info['LLM'])
                if 'graph' in info:
                    saved_info['graph'][agent_id].append(copy.deepcopy(info['graph']))
                if self.debug:
                    pickle.dump(saved_info, open(os.path.join(self.record_dir, 'log.pik'), 'wb'))
            if done:
                break
        saved_info['finished'] = success
        # ==================== Metrics: Forced-Coop Episode Summary ====================
        # Purpose: Call method-independent metrics outside arena logic and store the result.
        if build_episode_metrics is not None:
            try:
                saved_info['metrics'] = build_episode_metrics(saved_info)
            except Exception as exc:
                saved_info['metrics'] = {'error': repr(exc)}
        else:
            saved_info['metrics'] = {'error': 'metrics_forced_coop_unavailable'}
        # ==================== End Metrics: Forced-Coop Episode Summary ====================
        return success, self.env.steps, saved_info
