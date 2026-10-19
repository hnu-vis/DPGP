"""Online TDW adapter for deterministic and room-discovery planning."""

from __future__ import annotations

from dataclasses import replace
import json
import time
from typing import Any, Dict, Iterable

from eval.llm_usage import summarize_llm_calls

from .agent_memory import AgentMemory
from .astar_planner import plan_deterministic_graph
from .config import MethodConfig
from .coordination import evaluate_coordination_needs, merge_fact_graphs
from .discovery_belief import DiscoveryBelief, DiscoveryBeliefError
from .discovery_planner import plan_discovery
from .executor import ExecutionUpdate, MethodExecutor
from .memory import MethodMemory
from .prediction import (
    PredictionResult,
    SimpleLLMClient,
    is_llm_dependency_failure,
    predict_deterministic_graph,
)
from .protocol import (
    EXCLUSIVE_ACTION,
    CommunicationCoordinator,
    PlanningActionMessage,
    PlanningGraphMessage,
)


def _rooms_by_id(rooms_name: Any) -> Dict[int, str]:
    if isinstance(rooms_name, dict):
        items: Iterable = rooms_name.items()
    elif isinstance(rooms_name, (list, tuple)):
        items = enumerate(rooms_name)
    else:
        raise TypeError("rooms_name must be a dictionary, list, or tuple")
    return {int(room_id): name for room_id, name in items if name is not None}


DETERMINISTIC_LLM = "deterministic_llm"
DISCOVERY_PROBABILITY = "discovery_probability"
PREDICTION_MODES = (DETERMINISTIC_LLM, DISCOVERY_PROBABILITY)


class DPGP:
    """Coordinate probability or deterministic planning from public facts."""

    def __init__(
        self,
        agent_id,
        logger,
        max_frames,
        output_dir,
        llm_model="gpt-4o-mini",
        llm_temperature=0.0,
        llm_top_p=1.0,
        llm_max_tokens=4096,
        llm_n=1,
        communication_mode=EXCLUSIVE_ACTION,
        communication_max_message_chars=12000,
        prediction_mode=DISCOVERY_PROBABILITY,
        llm_client=None,
    ):
        self.agent_id = int(agent_id)
        if self.agent_id not in (0, 1):
            raise ValueError("agent_id must be 0 or 1")
        self.logger = logger
        self.max_frames = None if max_frames is None else int(max_frames)
        self.output_dir = output_dir
        self.communication_mode = communication_mode
        if prediction_mode not in PREDICTION_MODES:
            raise ValueError("unsupported prediction_mode")
        self.prediction_mode = prediction_mode
        self.config = replace(
            MethodConfig(),
            message_limit=int(communication_max_message_chars),
        )
        self.llm_client = llm_client if llm_client is not None else SimpleLLMClient(
            model=llm_model,
            temperature=llm_temperature,
            top_p=llm_top_p,
            max_tokens=llm_max_tokens,
            n=llm_n,
        )
        self.agent_memory = None
        self.method_memory = None
        self.executor = MethodExecutor(self.agent_id, self.config)
        self.coordinator = CommunicationCoordinator(
            self.agent_id, self.communication_mode, self.config
        )
        self._clear_episode_state()

    def _clear_episode_state(self) -> None:
        self.current_fact_graph = None
        self.previous_fact_graph = None
        self.last_fused_fact_graph = None
        self.last_deterministic_graph = None
        self.last_planning_result = None
        self.last_execution_update = ExecutionUpdate()
        self.last_effective_physical_action = None
        self.last_new_situations = ()
        self.last_global_events = ()
        self.last_local_events = ()
        self.last_planning_scope = None
        self.last_llm_info = {}
        self.last_discovery_diagnostics = {}
        self._last_planned_peer_key = None
        self._pending_global_event = False
        self._consecutive_idle_steps = 0
        self._last_public_frame = None
        self._assigned_idle = False
        self._awaiting_plan = False
        self._llm_failure_streak = 0
        self._llm_dependency_failure_streak = 0
        self._llm_circuit_open_until = 0
        self._method_dependency_error = None
        self._failed_prediction_graph = None
        self.step = 0
        self.llm_call_details = []
        self.discovery_belief = DiscoveryBelief()

    def reset(
        self,
        obs,
        goal_objects=None,
        output_dir=None,
        env_api=None,
        rooms_name=None,
        agent_color=None,
        agent_id=0,
        gt_mask=True,
        save_img=True,
    ):
        """Reset from the public runner contract; no planning is performed."""
        del save_img
        self.agent_id = int(agent_id)
        if self.agent_id not in (0, 1):
            raise ValueError("agent_id must be 0 or 1")
        if output_dir is not None:
            self.output_dir = output_dir
        if env_api is None:
            raise ValueError("env_api is required")
        self.agent_memory = AgentMemory(
            self.agent_id,
            agent_color,
            env_api=env_api,
            config=self.config,
            gt_mask=bool(gt_mask),
        )
        self.method_memory = MethodMemory(self.agent_id)
        self.executor = MethodExecutor(self.agent_id, self.config)
        self.coordinator = CommunicationCoordinator(
            self.agent_id, self.communication_mode, self.config
        )
        self._clear_episode_state()
        self.method_memory.reset(
            goal_objects=goal_objects,
            rooms_name=_rooms_by_id(rooms_name),
            env_api=env_api,
        )
        self.agent_memory.update(obs)
        self.method_memory.update(obs, agent_memory=self.agent_memory)
        self.current_fact_graph = self.method_memory.snapshot()
        self._last_public_frame = int(obs["current_frames"])

    def _require_reset(self) -> None:
        if (self.agent_memory is None or self.method_memory is None
                or self.current_fact_graph is None):
            raise RuntimeError("DPGP.reset() must be called before act()")

    def get_initial_snapshot(self):
        self._require_reset()
        return {
            "method_memory": self.method_memory.to_dict(),
            "planning_graph": self.current_fact_graph.to_dict(),
        }

    def _record_llm_call(self, usage: dict, prediction_failed: bool,
                         graph_step: int) -> None:
        self.last_llm_info = dict(usage)
        if not usage.get("llm_called", True):
            return
        attempts = usage.get("llm_attempt_details")
        if isinstance(attempts, list):
            for attempt in attempts:
                detail = dict(attempt)
                detail["prediction_failed"] = (
                    detail.get("prediction_status") != "applied"
                )
                detail.update(
                    graph_step=graph_step,
                    agent_id=self.agent_id,
                    env_step=self.step,
                )
                self.llm_call_details.append(detail)
            return
        detail = dict(usage)
        detail.update(
            prediction_failed=prediction_failed,
            graph_step=graph_step,
            agent_id=self.agent_id,
            env_step=self.step,
        )
        self.llm_call_details.append(detail)

    def get_llm_stats(self):
        return summarize_llm_calls(self.llm_call_details)

    def get_method_diagnostics(self):
        return {
            "step": self.step,
            "events": list(self.last_new_situations),
            "global_events": list(self.last_global_events),
            "local_events": list(self.last_local_events),
            "planning_scope": self.last_planning_scope,
            "plan_epoch": self.coordinator.current_plan_epoch,
            "protocol_error": self.coordinator.last_protocol_error,
            "prediction_status": self.last_llm_info.get("prediction_status"),
            "prediction_error": self.last_llm_info.get("prediction_error"),
            "prediction_mode": self.prediction_mode,
            "discovery_belief": self.discovery_belief.diagnostics,
            "discovery_planner": dict(self.last_discovery_diagnostics),
            "llm_failure_streak": self._llm_failure_streak,
            "llm_dependency_failure_streak": self._llm_dependency_failure_streak,
            "llm_circuit_open_until": self._llm_circuit_open_until,
            "method_dependency_error": self._method_dependency_error,
            "planner_status": getattr(self.last_planning_result, "status", None),
            "planner_reason": getattr(self.last_planning_result, "reason", None),
            "pending_action_count": len(self.executor.pending_actions),
            "executor_failure": self.last_execution_update.failure_reason,
            "navigation_recovery": self.executor.navigation_recovery,
            "navigation_event": self.executor.navigation_event,
            "effective_physical_action": self.last_effective_physical_action,
        }

    def get_method_dependency_error(self):
        return self._method_dependency_error

    def _update_llm_circuit(self, usage: dict) -> None:
        status = usage.get("prediction_status")
        if status == "applied":
            self._llm_failure_streak = 0
            self._llm_dependency_failure_streak = 0
            self._llm_circuit_open_until = 0
            self._method_dependency_error = None
        elif status == "failed" and usage.get("llm_called"):
            self._llm_failure_streak += 1
            self._llm_circuit_open_until = (
                self.step + self.config.prediction_cooldown_steps
            )
            self._llm_dependency_failure_streak = (
                self._llm_dependency_failure_streak + 1
                if is_llm_dependency_failure(usage) else 0
            )
            if (self._llm_dependency_failure_streak
                    >= self.config.dependency_failure_rounds):
                self._method_dependency_error = (
                    "LLM dependency failed for {} consecutive rounds: {}".format(
                        self._llm_dependency_failure_streak,
                        usage.get("prediction_error") or "unknown error",
                    )
                )

    def _log_planning(self, result, prediction, plan_epoch: int,
                      planning_scope: str, elapsed: float) -> None:
        if self.logger is None:
            return
        self.logger.info(
            "Method planning: agent=%s scope=%s local_step=%s epoch=%s "
            "prediction=%s status=%s reason=%s cost=%s expanded=%s "
            "elapsed_seconds=%.6f",
            self.agent_id,
            planning_scope,
            self.step,
            plan_epoch,
            prediction.usage.get("prediction_status"),
            getattr(result, "status", None),
            getattr(result, "reason", prediction.usage.get("prediction_error")),
            getattr(result, "total_cost", None),
            getattr(result, "expanded_states", 0),
            elapsed,
        )

    def _plan_with_peer_graph(
        self,
        peer_graph,
        peer_step: int,
        obs: dict,
        plan_epoch: int,
        planning_scope: str,
        peer_commitment=None,
    ):
        fused = merge_fact_graphs(
            self.current_fact_graph,
            peer_graph,
            self.agent_id,
            self.step,
            peer_step,
            coop_info=obs.get("coop_info") or {},
        )
        self.method_memory.apply_fused_facts(fused)
        self.current_fact_graph = self.method_memory.snapshot()
        self.executor.refresh_after_graph_update(obs, self.method_memory)
        llm_enabled = (
            self._method_dependency_error is None
            and self.step >= self._llm_circuit_open_until
        )
        commitments = (self.executor.commitment(obs.get("status") == 0), peer_commitment)
        if self.prediction_mode == DETERMINISTIC_LLM:
            if fused == self._failed_prediction_graph:
                prediction = PredictionResult(None, {
                    "llm_called": False,
                    "prediction_status": "deferred",
                    "prediction_error": "same_state_prediction_failed",
                }, True)
            else:
                prediction = predict_deterministic_graph(
                    fused, self.llm_client, self.agent_id, self.config,
                    llm_enabled=llm_enabled,
                )
                if (prediction.usage.get("prediction_status") == "failed"
                        and not is_llm_dependency_failure(prediction.usage)):
                    self._failed_prediction_graph = fused
                elif prediction.graph is not None:
                    self._failed_prediction_graph = None
            discovery = None
        else:
            try:
                usage = self.discovery_belief.synchronize(
                    fused, self.llm_client, self.agent_id, self.config,
                    llm_enabled=llm_enabled,
                )
            except DiscoveryBeliefError as error:
                usage = error.usage or {
                    "llm_called": False,
                    "prediction_status": "failed",
                    "prediction_kind": "discovery_support",
                    "prediction_error": str(error),
                }
                prediction = PredictionResult(None, usage, True)
                discovery = None
            else:
                discovery = plan_discovery(
                    fused, self.discovery_belief,
                    navigation=self.agent_memory, config=self.config,
                    commitments=commitments,
                )
                prediction = PredictionResult(
                    discovery.graph, usage, not discovery.result.succeeded
                )
        self._record_llm_call(prediction.usage, prediction.failed, peer_step)
        self._update_llm_circuit(prediction.usage)
        self.last_fused_fact_graph = fused
        self.last_deterministic_graph = prediction.graph
        self.last_discovery_diagnostics = (
            {} if discovery is None else discovery.diagnostics
        )
        self.last_planning_scope = planning_scope

        if prediction.graph is None:
            self.last_planning_result = None
            self._log_planning(None, prediction, plan_epoch, planning_scope, 0.0)
            return None, None
        if discovery is None:
            started = time.perf_counter()
            result = plan_deterministic_graph(
                prediction.graph, navigation=self.agent_memory, commitments=commitments
            )
            elapsed = time.perf_counter() - started
        else:
            result, elapsed = discovery.result, discovery.wall_time
        self.last_planning_result = result
        self._log_planning(result, prediction, plan_epoch, planning_scope, elapsed)
        return result, prediction.graph

    def _process_peer_graph(self, incoming, obs, communication_events) -> bool:
        key = incoming.source_agent, incoming.plan_epoch
        if key == self._last_planned_peer_key:
            return False
        simultaneous = self.coordinator.last_sent_graph_step == incoming.plan_epoch
        if simultaneous and self.agent_id == 1 and not communication_events:
            self.last_planning_scope = "deferred_to_agent_0"
            self._last_planned_peer_key = key
            return True

        result, graph = self._plan_with_peer_graph(
            incoming.graph,
            incoming.source_step,
            obs,
            incoming.plan_epoch,
            "global",
            incoming.commitment,
        )
        if (result is not None and graph is not None
                and result.succeeded):
            self.coordinator.prepare_actions(
                result, graph, self.step, incoming.plan_epoch,
                (self.discovery_belief.to_seed()
                 if self.prediction_mode == DISCOVERY_PROBABILITY else None),
            )
            self.coordinator.select_plan(self.coordinator.take_prepared_own_plan())
            self._awaiting_plan = False
        else:
            # A failed search is neither an assigned idle nor a completion.
            # Keep valid current work, but never spin forever on two empty queues.
            self.executor.clear_plan(preserve_current=self._preserve_execution(obs))
            peer_busy = bool(incoming.commitment and incoming.commitment["busy"])
            if (result is not None and result.status == "failed"
                    and not self.executor.has_plan_work() and not peer_busy):
                if result.reason == "no executable discovery candidate":
                    # Reuse the idle watchdog, not an immediate prefix retry.
                    # Persistent failure ends through the unchanged time budget.
                    self._awaiting_plan = True
                else:
                    raise RuntimeError("TDW joint planning failed: {}".format(result.reason))
            if peer_busy and not self.executor.has_plan_work():
                self._awaiting_plan = True
        self._last_planned_peer_key = key
        return True

    def _adopt_belief(self, incoming) -> None:
        if self.prediction_mode != DISCOVERY_PROBABILITY:
            return
        seed = getattr(incoming, "belief_seed", None)
        if seed is None:
            return
        candidate = DiscoveryBelief.from_seed(seed)
        if candidate.revision >= self.discovery_belief.revision:
            self.discovery_belief = candidate

    def _unfinished(self) -> bool:
        return any(not item.finished for item in self.current_fact_graph.objects)

    def _preserve_execution(self, obs: dict) -> bool:
        current = self.executor._unwrapped_action()
        if (current is not None and current.target_kind == "room"
                and not self.executor.exploring and obs.get("status") != 0):
            return self.executor.room_scan_pending(self.method_memory)
        return self.executor.commitment(obs.get("status") == 0)["busy"]

    def _idle_replan_due(self, incoming, events,
                         public_frame_advanced: bool) -> bool:
        return (
            self._consecutive_idle_steps >= self.config.idle_replan_steps - 1
            and not public_frame_advanced
            and incoming is None
            and not events
            and not self.executor.has_plan_work()
            and not self.coordinator.has_pending_work()
            and self._unfinished()
        )

    def _update_idle_counter(self, incoming, envelope,
                             public_frame_advanced: bool) -> None:
        quiet = (
            not public_frame_advanced
            and incoming is None
            and not self.last_new_situations
            and self.last_planning_scope is None
            and envelope["message"] is None
            and envelope["physical_action"].get("type") == 7
            and not self.executor.has_plan_work()
            and not self.coordinator.has_pending_work()
            and self._unfinished()
        )
        self._consecutive_idle_steps = self._consecutive_idle_steps + 1 if quiet else 0

    def _activate_plan(self, obs: dict) -> None:
        plan = self.coordinator.activate(self.step)
        if isinstance(plan, PlanningActionMessage):
            self.executor.refresh_after_graph_update(obs, self.method_memory)
            self.executor.set_action_sequence(
                plan.actions,
                plan.plan_epoch,
                preserve_current=self._preserve_execution(obs),
            )
            self._assigned_idle = not bool(plan.actions)
            self._awaiting_plan = False

    def act(self, obs):
        """Advance exactly one public-observation transaction."""
        self._require_reset()
        self.step += 1
        current_frame = int(obs["current_frames"])
        public_frame_advanced = (
            self._last_public_frame is not None
            and current_frame > self._last_public_frame
        )
        self._last_public_frame = current_frame
        self.last_planning_scope = None
        self.last_llm_info = {}
        self.agent_memory.update(obs)
        feedback = obs.get("action_feedback") or {}
        previous_action = feedback.get("effective_physical_action")
        if previous_action is None:
            previous_action = self.last_effective_physical_action
        self.method_memory.update(
            obs,
            agent_memory=self.agent_memory,
            previous_effective_action=previous_action,
        )
        self.last_execution_update = self.executor.observe(obs, self.method_memory)
        self.current_fact_graph = self.method_memory.snapshot()

        needs = evaluate_coordination_needs(
            self.previous_fact_graph,
            self.current_fact_graph,
            self.agent_id,
            (
                self.last_execution_update.completed_move_target_room_node_id
                if self.last_execution_update.completed_move_target_room_node_id
                is not None
                else self.executor.current_move_target_room_node_id()
            ),
            self.last_execution_update.planning_action_completed,
            self.last_execution_update.action_failed,
        )
        events = list(needs.all_events)
        communication_events = list(needs.communication_events)
        local_events = list(needs.local_events)
        if self._pending_global_event:
            events.append("deferred_global_event")
            communication_events.append("deferred_global_event")
        if (self.executor.last_plan_step is not None and not self.executor.has_plan_work()
                and not self._awaiting_plan
                and not self._assigned_idle and not self.coordinator.has_pending_work()
                and self._unfinished() and not communication_events):
            events.append("prefix_exhausted")
            communication_events.append("prefix_exhausted")

        messages = obs.get("messages") or [None, None]
        peer_message = messages[1 - self.agent_id]
        incoming = self.coordinator.receive(peer_message, self.step)
        if incoming is not None:
            self._adopt_belief(incoming)
        busy_exclusive = obs.get("status") == 0
        new_peer_graph = (
            isinstance(incoming, PlanningGraphMessage)
            and (incoming.source_agent, incoming.plan_epoch)
            != self._last_planned_peer_key
        )
        latest = self.coordinator.latest_peer_graph
        if (not new_peer_graph and latest is not None
                and (latest.source_agent, latest.plan_epoch)
                != self._last_planned_peer_key):
            new_peer_graph, incoming = True, latest
        if new_peer_graph and self.coordinator.is_stale_epoch(incoming.plan_epoch):
            self._last_planned_peer_key = incoming.source_agent, incoming.plan_epoch
            self.last_planning_scope = "stale_peer_graph_cached"
            new_peer_graph = False

        coop = obs.get("coop_info") or {}
        if (not new_peer_graph and not busy_exclusive
                and (not coop.get("active") or coop.get("holder_id") == self.agent_id)
                and self._idle_replan_due(incoming, events, public_frame_advanced)):
            events.append("idle_replan_timeout")
            communication_events.append("idle_replan_timeout")

        if new_peer_graph and not busy_exclusive:
            self._process_peer_graph(incoming, obs, communication_events)
            self._pending_global_event = False
        elif communication_events and not busy_exclusive:
            self.method_memory.refresh_navigation(self.agent_memory)
            self.current_fact_graph = self.method_memory.snapshot()
            self.coordinator.prepare_graph(
                self.current_fact_graph,
                self.step,
                (self.discovery_belief.to_seed()
                 if self.prediction_mode == DISCOVERY_PROBABILITY else None),
                self.executor.commitment(obs.get("status") == 0),
            )
            self.executor.clear_plan(preserve_current=self._preserve_execution(obs))
            self._assigned_idle = False
            self._awaiting_plan = True
            self.last_planning_scope = "global_graph_sent"
            self._pending_global_event = False
        elif communication_events:
            self._pending_global_event = True

        self._activate_plan(obs)
        proposal = self.executor.choose_physical_action(
            obs, self.method_memory, self.agent_memory
        )
        can_send = proposal.get("type") != "ongoing"
        outgoing = self.coordinator.choose_outgoing(can_send)
        envelope = self.coordinator.schedule_envelope(proposal, outgoing)
        self.executor.record_submitted_action(envelope["physical_action"])
        self.last_effective_physical_action = envelope["physical_action"]
        navigation_event = self.executor.navigation_recovery or self.executor.navigation_event
        if navigation_event is not None and self.logger is not None:
            self.logger.info(
                "Method navigation: agent=%s step=%s event=%s physical=%s",
                self.agent_id, self.step,
                json.dumps(navigation_event, separators=(",", ":")),
                envelope["physical_action"],
            )

        self.last_new_situations = tuple(events)
        self.last_global_events = tuple(communication_events)
        self.last_local_events = tuple(local_events)
        self.previous_fact_graph = self.current_fact_graph
        self._update_idle_counter(
            incoming, envelope, public_frame_advanced
        )
        return envelope
