"""Persistent semantic queue and one-step TDW physical execution."""

from dataclasses import dataclass, field
from math import hypot
from typing import Optional

from .config import (MethodConfig, EXPLORE_CENTER_RADIUS,
                     OBJECT_APPROACH_RADIUS, GOAL_APPROACH_RADIUS)


MOVE = "move"
GRAB = "grab"
COOP_GRAB = "coop_grab"
PUT = "put"
COOP_PUT = "coop_put"
WAIT = "wait"
COOP_RESUME = "coop_resume"
SEMANTIC_KINDS = frozenset((
    MOVE, GRAB, COOP_GRAB, PUT, COOP_PUT, WAIT, COOP_RESUME,
))

ROOM = "room"
OBJECT = "object"
GOAL = "goal"

MOVE_FORWARD = 0
TURN_LEFT = 1
TURN_RIGHT = 2
GRASP = 3
COOP_GRASP = 4
RELEASE = 5
COOP_RELEASE = 6
WAIT_PHYSICAL = 7

SCAN_REQUIRED_TURNS = 16
DEFAULT_COOP_RECOVERY_DISTANCE = 3.0
COOP_APPROACH_MARGIN = 0.5
COOP_MIN_STANDOFF = 0.25
NAVIGATION_STALL_STEPS = 24
NAVIGATION_RECOVERY_ATTEMPTS = 2


@dataclass
class SemanticAction:
    """One persistent semantic action, optionally wrapping its successor."""

    kind: str
    object_index: Optional[int] = None
    object_id: Optional[int] = None
    target_room_node_id: Optional[int] = None
    target_kind: Optional[str] = None
    required_revisit_count: Optional[int] = None
    wait_reason: Optional[str] = None
    remaining_wait_steps: Optional[int] = None
    next_action: Optional["SemanticAction"] = None

    def __post_init__(self) -> None:
        if self.kind not in SEMANTIC_KINDS:
            raise ValueError("unknown semantic action kind: {}".format(self.kind))
        if self.required_revisit_count is not None:
            if (self.kind != MOVE or self.target_kind != ROOM
                    or type(self.required_revisit_count) is not int
                    or self.required_revisit_count < 0):
                raise ValueError(
                    "required_revisit_count requires a room move and non-negative integer"
                )

    @classmethod
    def wait(cls, steps: Optional[int], reason: str,
             next_action: Optional["SemanticAction"] = None):
        if steps is not None and (type(steps) is not int or steps <= 0):
            raise ValueError("finite wait steps must be a positive integer")
        return cls(WAIT, wait_reason=str(reason), remaining_wait_steps=steps,
                   next_action=next_action)

    @classmethod
    def resume(cls, target_room_node_id: Optional[int],
               next_action: Optional["SemanticAction"]):
        return cls(COOP_RESUME, target_room_node_id=target_room_node_id,
                   next_action=next_action)


@dataclass
class ExploreState:
    room_node_id: int
    turn_count: int = 0
    turn_pending_confirmation: bool = False
    phase: str = "approach"
    required_turns: int = SCAN_REQUIRED_TURNS
    required_revisit_count: Optional[int] = None
    viewpoint: Optional[tuple] = None
    start_frame: Optional[int] = None
    observed_target_ids: set = field(default_factory=set)
    scan_pass: int = 0


@dataclass(frozen=True)
class ExecutionUpdate:
    planning_action_completed: bool = False
    completed_move_target_room_node_id: Optional[int] = None
    action_failed: bool = False
    failure_reason: Optional[str] = None


RECOVERABLE_FAILURE_REASONS = frozenset((
    "collision",
    "detected_obstacle",
    "coop_action_requires_pair",
    "coop_drop_agents_too_far",
    "requires_two_agent_cooperation",
    "coop_grasp_non_coop_target",
    "new_action_while_ongoing",
    "no_ongoing_action",
))


def _position(value):
    if value is None or len(value) < 3:
        return None
    return float(value[0]), float(value[1]), float(value[2])


def _distance(left, right):
    left, right = _position(left), _position(right)
    if left is None or right is None:
        return float("inf")
    return hypot(left[0] - right[0], left[2] - right[2])


class MethodExecutor:
    """Keep execution state independent of factual memory and planning."""

    def __init__(self, agent_id: int, config: Optional[MethodConfig] = None):
        if agent_id not in (0, 1):
            raise ValueError("agent_id must be 0 or 1")
        self.agent_id = int(agent_id)
        self.config = config or MethodConfig()
        self.reset()

    def reset(self) -> None:
        self.semantic_action: Optional[SemanticAction] = None
        self.explore_state: Optional[ExploreState] = None
        self.last_mode: Optional[str] = None
        self.last_physical_action: Optional[dict] = None
        self.pending_interaction = None
        self.last_plan_step: Optional[int] = None
        self.pending_actions = []
        self.deferred_actions = None
        self.last_submitted_physical_action = None
        self.last_failure_reason = None
        self.unknown_target_wait_count = 0
        self.coop_pair_retry_count = 0
        self.last_observed_frame = None
        self._navigation_progress = {}
        self.navigation_recovery = None
        self.navigation_event = None
        self._navigation_failure = None

    @property
    def exploring(self) -> bool:
        return self.explore_state is not None

    def current_move_target_room_node_id(self):
        current = self._unwrapped_action()
        return (current.target_room_node_id if current is not None
                and current.kind == MOVE and current.target_kind == ROOM else None)

    def room_scan_pending(self, method_memory) -> bool:
        current = self.semantic_action
        if (current is None or current.kind != MOVE
                or current.target_kind != ROOM):
            return False
        room = method_memory.rooms.get(current.target_room_node_id)
        return room is not None and (
            not room.explored
            or (current.required_revisit_count is not None
                and room.revisit_count < current.required_revisit_count)
        )

    def has_plan_work(self) -> bool:
        return (self.semantic_action is not None or bool(self.pending_actions)
                or self.deferred_actions is not None)

    def commitment(self, physical_ongoing: bool = False) -> dict:
        """Expose only the current non-preemptible execution summary."""
        current = self._unwrapped_action()
        kind = None if current is None else current.kind
        if kind in (COOP_GRAB, COOP_PUT):
            kind = GRAB if kind == COOP_GRAB else PUT
        if kind not in (MOVE, GRAB, PUT):
            kind = None
        return {
            "agent_id": self.agent_id,
            "busy": bool(physical_ongoing
                         or (self.pending_interaction and kind == GRAB)
                         or (kind in (GRAB, PUT) and self.last_mode == MOVE)
                         or self.explore_state is not None
                         or (current is not None and current.kind == MOVE)),
            "action": (None if kind is None else {
                "kind": kind,
                "object_id": current.object_id,
                "target_kind": current.target_kind if kind == MOVE else None,
                "target_room_node_id": current.target_room_node_id if kind == MOVE else None,
                **({"required_revisit_count": current.required_revisit_count or 0,
                    "scan_started": bool(physical_ongoing or (self.explore_state is not None
                                         and self.explore_state.start_frame is not None))}
                   if kind == MOVE and current.target_kind == ROOM else {}),
            }),
        }

    def clear_plan(self, preserve_current: bool = False) -> None:
        self.pending_actions = []
        self.deferred_actions = None
        if not preserve_current:
            self.semantic_action = None
            self.pending_interaction = None
            self.unknown_target_wait_count = 0
            self.coop_pair_retry_count = 0

    def set_action_sequence(self, actions, plan_epoch: int,
                            preserve_current: bool = False) -> None:
        if type(plan_epoch) is not int or plan_epoch < 0:
            raise ValueError("plan_epoch must be a non-negative integer")
        if self.last_plan_step is not None and plan_epoch < self.last_plan_step:
            return
        converted = [self._from_action_dict(item) for item in actions]
        self.last_plan_step = plan_epoch
        if preserve_current and self.semantic_action is not None:
            current = self._unwrapped_action()
            def identity(action):
                return None if action is None else (
                    action.kind, action.object_id, action.target_kind,
                    action.target_room_node_id, action.required_revisit_count or 0)
            # A replan may spell out the approach currently executed under
            # an interaction intent. Keep that intent, not a second Move queue.
            if (converted and current is not None
                    and current.kind in (GRAB, COOP_GRAB, PUT, COOP_PUT)
                    and converted[0].kind == MOVE
                    and converted[0].object_id == current.object_id
                    and converted[0].target_kind == (
                        OBJECT if current.kind in (GRAB, COOP_GRAB) else GOAL)):
                converted.pop(0)
            if converted and identity(converted[0]) == identity(current):
                converted.pop(0)
            elif (converted and self.pending_interaction
                  and current is not None and current.kind == COOP_GRAB):
                raise ValueError("new plan does not continue the active pair request")
            self.deferred_actions, self.pending_actions = converted, []
            return
        self.semantic_action = converted.pop(0) if converted else None
        self.pending_actions, self.deferred_actions = converted, None
        self.pending_interaction = None
        self.unknown_target_wait_count = self.coop_pair_retry_count = 0

    def activate_next(self) -> bool:
        if self.semantic_action is not None or not self.pending_actions:
            return False
        self.semantic_action = self.pending_actions.pop(0)
        return True

    def observe(self, obs: dict, method_memory) -> ExecutionUpdate:
        """Consume the previous public action result and advance the queue."""
        current_frame = int(obs["current_frames"])
        frame_advanced = (
            None if self.last_observed_frame is None
            else current_frame > self.last_observed_frame
        )
        self.last_observed_frame = current_frame
        self.navigation_recovery = None
        self.navigation_event = None
        self._confirm_previous_explore_turn(obs, method_memory)
        completed = self._complete_interaction(obs, method_memory)
        self._sync_explore_state(obs, method_memory)
        move_completed, completed_room = self._complete_move(
            method_memory, obs.get("coop_info") or {})
        completed = completed or move_completed

        feedback = obs.get("action_feedback") or {}
        rejection = feedback.get("reason_code")
        if (obs.get("status") != 0 and rejection is not None
                and rejection != "coop_action_requires_pair"):
            self.pending_interaction = None

        reason = self._navigation_failure if obs.get("status") != 0 else None
        if reason is not None:
            self._navigation_failure = None
            self.explore_state = None
        reason = reason or self._bounded_recovery_reason(
            obs, method_memory, frame_advanced
        )
        if reason is None:
            reason = self._unrecoverable_failure_reason(obs)
        if reason is not None:
            self.clear_plan(preserve_current=reason == "coop_pairing_retry_exhausted")
            self.coop_pair_retry_count = 0
            self.last_failure_reason = reason
            return ExecutionUpdate(action_failed=True, failure_reason=reason)

        if (obs.get("status") != 0 and self.deferred_actions is not None
                and self.semantic_action is None and self.explore_state is None):
            actions, self.deferred_actions = self.deferred_actions, None
            self.semantic_action = actions.pop(0) if actions else None
            self.pending_actions = actions
            self.pending_interaction = None
        self.activate_next()
        # A delayed prefix can contain the scan that has just completed.
        # Retire it before choose_physical_action can restart its navigation.
        while self.semantic_action is not None:
            current = self._unwrapped_action()
            if current is None or current.kind != MOVE or current.target_kind != ROOM:
                break
            done, room = self._complete_move(method_memory)
            if not done:
                break
            completed, completed_room = True, room
        return ExecutionUpdate(completed, completed_room)

    def refresh_after_graph_update(self, obs: dict, method_memory) -> None:
        self._sync_explore_state(obs, method_memory)
        current = self._unwrapped_action()
        if current is not None and current.kind == MOVE and current.target_kind == ROOM:
            if (obs.get("status") != 0
                    and not (self.explore_state is not None and self.explore_state.start_frame is not None)
                    and method_memory.goal_anchor.known
                    and not any(not o.finished and not o.holder_agent_ids and o.room_node_id is None
                                for o in method_memory.objects.values())):
                self.explore_state = None
                self.clear_plan()
            else:
                self._complete_move(method_memory, obs.get("coop_info") or {})
        elif (current is not None and current.kind == MOVE and current.target_kind == OBJECT
              and obs.get("status") != 0):
            item = self._object_memory(current, method_memory)
            if item is not None and (item.finished or item.holder_agent_ids):
                self.clear_plan()

    def choose_physical_action(self, obs: dict, method_memory, agent_memory):
        """Return exactly one TDW action without consuming persistent intent."""
        coop_info = obs.get("coop_info") or {}
        self._sync_coop_resume(coop_info, method_memory)
        if obs.get("status") == 0:
            action, mode = {"type": "ongoing"}, "ongoing"
        elif (coop_info.get("active")
              and coop_info.get("self_role") == "holder"
              and float(coop_info.get("distance", 0.0))
              > self._coop_recovery_distance(coop_info)):
            action, mode = {"type": WAIT_PHYSICAL}, "coop_holder_recovery_wait"
        elif self.explore_state is not None:
            action, mode = self._explore_action(method_memory, agent_memory)
        elif self.semantic_action is None:
            action, mode = {"type": WAIT_PHYSICAL}, "idle"
        elif self.semantic_action.kind == WAIT:
            action, mode = {"type": WAIT_PHYSICAL}, "semantic_wait"
        elif self.semantic_action.kind == COOP_RESUME:
            action, mode = self._resume_action(coop_info, method_memory, agent_memory)
        else:
            action, mode = self._ordinary_action(
                self.semantic_action, obs, method_memory, agent_memory
            )
        self._advance_finite_wait()
        self.last_mode = mode
        return action

    def record_submitted_action(self, action: dict) -> None:
        """Record the post-envelope physical action actually sent to TDW."""
        self.last_physical_action = self.last_submitted_physical_action = action
        if (self.explore_state is not None
                and self.explore_state.turn_pending_confirmation
                and action.get("type") not in (TURN_RIGHT, "ongoing")):
            self.explore_state.turn_count -= 1
            self.explore_state.turn_pending_confirmation = False
        if action.get("type") in (GRASP, COOP_GRASP, RELEASE, COOP_RELEASE):
            kind = {GRASP: GRAB, COOP_GRASP: COOP_GRAB,
                    RELEASE: PUT, COOP_RELEASE: COOP_PUT}[action["type"]]
            self.pending_interaction = kind, action.get("object")

    @staticmethod
    def _from_action_dict(raw: dict) -> SemanticAction:
        if not isinstance(raw, dict) or raw.get("kind") not in SEMANTIC_KINDS:
            raise ValueError("invalid semantic action payload")
        if raw["kind"] == WAIT:
            return SemanticAction.wait(
                raw.get("remaining_wait_steps", 1),
                raw.get("wait_reason") or "planner_wait",
            )
        return SemanticAction(
            kind=raw["kind"], object_index=raw.get("object_index"),
            object_id=raw.get("object_id"),
            target_room_node_id=raw.get("target_room_node_id"),
            target_kind=raw.get("target_kind"),
            required_revisit_count=raw.get("required_revisit_count"),
        )

    def _unwrapped_action(self):
        current = self.semantic_action
        while current is not None and current.kind in (WAIT, COOP_RESUME):
            current = current.next_action
        return current

    def _advance_finite_wait(self) -> None:
        current = self.semantic_action
        if (current is None or current.kind != WAIT
                or current.remaining_wait_steps is None):
            return
        current.remaining_wait_steps -= 1
        if current.remaining_wait_steps <= 0:
            self.semantic_action = current.next_action

    def _confirm_previous_explore_turn(self, obs: dict, method_memory) -> None:
        state = self.explore_state
        if state is not None and state.start_frame is not None:
            state.observed_target_ids.update(
                method_memory.current_visible_target_ids
            )
        if state is None or not state.turn_pending_confirmation:
            return
        if self._previous_action_invalid(obs):
            state.turn_count -= 1
            state.turn_pending_confirmation = False
            return
        if (obs.get("status") == 0
                or str(obs.get("action_status") or "").lower() == "ongoing"):
            return
        state.turn_pending_confirmation = False
        if state.turn_count >= state.required_turns:
            method_memory.record_inspection(
                state.room_node_id,
                state.viewpoint,
                state.scan_pass,
                state.start_frame,
                int(obs["current_frames"]),
                state.observed_target_ids,
            )
            if not state.required_revisit_count:
                method_memory.mark_room_explored(state.room_node_id)
            else:
                method_memory.mark_room_revisited(
                    state.room_node_id, state.required_revisit_count
                )
            self.explore_state = None

    def _sync_explore_state(self, obs: dict, method_memory) -> None:
        own_room_id = method_memory.agents[self.agent_id].room_node_id
        if own_room_id is None:
            self.explore_state = None
            return
        room = method_memory.rooms[own_room_id]
        current = self._unwrapped_action()
        scan_requested = (
            current is not None and current.kind == MOVE
            and current.target_kind == ROOM
            and current.target_room_node_id == own_room_id
        )
        required_revisit_count = (
            current.required_revisit_count
            if scan_requested
            else None
        )
        needs_revisit = (
            required_revisit_count is not None
            and room.revisit_count < required_revisit_count
        )
        coop_info = obs.get("coop_info") or {}
        holder = coop_info.get("active") and coop_info.get("self_role") == "holder"
        if holder:
            self.explore_state = None
        elif (self.explore_state is not None
              and self.explore_state.room_node_id != own_room_id):
            self.explore_state = None
        elif self.explore_state is not None:
            return
        elif room.explored and not needs_revisit:
            self.explore_state = None
        elif scan_requested:
            self.explore_state = ExploreState(
                own_room_id,
                required_turns=SCAN_REQUIRED_TURNS,
                required_revisit_count=required_revisit_count,
                scan_pass=(required_revisit_count or 0),
            )

    def _explore_action(self, method_memory, agent_memory):
        state = self.explore_state
        room = method_memory.rooms[state.room_node_id]
        if state.viewpoint is None:
            if state.scan_pass == 0:
                state.viewpoint = room.center
            else:
                selector = getattr(agent_memory, "select_scan_viewpoint", None)
                state.viewpoint = (
                    selector(
                        room.room_name,
                        room.center,
                        method_memory.inspection_viewpoints(state.room_node_id),
                    ) if callable(selector) else room.center
                )
        if _distance(agent_memory.position, state.viewpoint) > EXPLORE_CENTER_RADIUS:
            state.phase = "approach"
            return self._navigate(agent_memory, state.viewpoint,
                                  radius=EXPLORE_CENTER_RADIUS,
                                  room_name=room.room_name), "explore_approach"
        state.phase = "turn"
        if state.start_frame is None:
            state.start_frame = method_memory.frame
            state.observed_target_ids.update(
                method_memory.current_visible_target_ids
            )
        state.turn_count += 1
        state.turn_pending_confirmation = True
        return {"type": TURN_RIGHT}, "explore_turn"

    def _sync_coop_resume(self, coop_info: dict, method_memory) -> None:
        current = self.semantic_action
        if not coop_info.get("active") or coop_info.get("self_role") != "supporter":
            if current is not None and current.kind == COOP_RESUME:
                self.semantic_action = current.next_action
            return
        if current is not None and current.kind == WAIT:
            return
        recovery_distance = self._coop_recovery_distance(coop_info)
        if current is not None and current.kind == COOP_RESUME:
            if self._delivery_reached(method_memory, method_memory.agents[self.agent_id].position,
                                      coop_info) or float(coop_info.get("distance", 0.0)) <= max(
                COOP_MIN_STANDOFF, recovery_distance - COOP_APPROACH_MARGIN
            ):
                self.semantic_action = current.next_action
            return
        if float(coop_info.get("distance", 0.0)) > recovery_distance:
            target_room = (current.target_room_node_id if current is not None
                           else method_memory.agents[self.agent_id].room_node_id)
            self.semantic_action = SemanticAction.resume(target_room, current)

    def _resume_action(self, coop_info, method_memory, agent_memory):
        if not coop_info.get("active") or coop_info.get("self_role") != "supporter":
            current = self.semantic_action
            self.semantic_action = current.next_action
            return {"type": WAIT_PHYSICAL}, "coop_resume_complete"
        action = self._follow_holder(coop_info, agent_memory)
        return action, "coop_resume"

    def _follow_holder(self, coop_info, agent_memory):
        radius = max(COOP_MIN_STANDOFF,
                     self._coop_recovery_distance(coop_info) - COOP_APPROACH_MARGIN)
        return self._navigate(agent_memory, coop_info.get("holder_pos"), radius=radius)

    @staticmethod
    def _coop_recovery_distance(coop_info: dict) -> float:
        threshold = coop_info.get(
            "soft_threshold", coop_info.get("hard_threshold", 4.0)
        )
        return min(DEFAULT_COOP_RECOVERY_DISTANCE, float(threshold))

    def _ordinary_action(self, action, obs, method_memory, agent_memory):
        coop_info = obs.get("coop_info") or {}
        if action.kind == MOVE:
            if action.target_kind == ROOM:
                target = method_memory.rooms[action.target_room_node_id].center
            elif action.target_kind == OBJECT:
                item = self._object_memory(action, method_memory)
                target = item.position if item is not None else None
            elif action.target_kind == GOAL:
                target = method_memory.goal_anchor.position
                if target is None:
                    room = method_memory.rooms.get(action.target_room_node_id)
                    target = room.center if room is not None else None
            else:
                target = None
            if target is None:
                return {"type": WAIT_PHYSICAL}, "move_target_unknown"
            reached = (self._delivery_reached(method_memory, agent_memory.position, coop_info)
                       if action.target_kind == GOAL else
                       _distance(agent_memory.position, target) <= (
                           OBJECT_APPROACH_RADIUS if action.target_kind == OBJECT
                           else EXPLORE_CENTER_RADIUS
                       ))
            if reached:
                return {"type": WAIT_PHYSICAL}, "move_complete"
            if action.target_kind == GOAL:
                if coop_info.get("active") and coop_info.get("self_role") == "supporter":
                    if _distance(agent_memory.position, coop_info.get("holder_pos")) <= max(
                            COOP_MIN_STANDOFF,
                            self._coop_recovery_distance(coop_info) - COOP_APPROACH_MARGIN):
                        return {"type": WAIT_PHYSICAL}, "coop_follow_wait"
                    proposal = self._follow_holder(coop_info, agent_memory)
                    return proposal, MOVE
            radius = {ROOM: EXPLORE_CENTER_RADIUS, OBJECT: OBJECT_APPROACH_RADIUS,
                      GOAL: GOAL_APPROACH_RADIUS}[action.target_kind]
            room = method_memory.rooms.get(action.target_room_node_id)
            proposal = self._navigate(
                agent_memory, target, radius=radius,
                room_name=(room.room_name if room is not None and action.target_kind != OBJECT else None))
            return proposal, MOVE

        object_memory = self._object_memory(action, method_memory)
        if action.kind in (GRAB, COOP_GRAB):
            if object_memory is not None and object_memory.finished:
                self.clear_plan()
                return {"type": WAIT_PHYSICAL}, "finished_target_skipped"
            if (object_memory is None or object_memory.object_id is None
                    or object_memory.position is None):
                return {"type": WAIT_PHYSICAL}, "grab_target_unknown"
            if _distance(agent_memory.position, object_memory.position) > OBJECT_APPROACH_RADIUS:
                return self._navigate(agent_memory, object_memory.position,
                                      radius=OBJECT_APPROACH_RADIUS), MOVE
            action_type = COOP_GRASP if action.kind == COOP_GRAB else GRASP
            return {"type": action_type, "object": object_memory.object_id}, action.kind

        if action.kind in (PUT, COOP_PUT):
            if not self._delivery_reached(method_memory, agent_memory.position, coop_info):
                if (not (coop_info.get("active") and coop_info.get("self_role") == "supporter")
                        and method_memory.goal_anchor.position is None):
                    return {"type": WAIT_PHYSICAL}, "put_target_unknown"
                # Reuse exactly the goal Move's holder/supporter mapping, while
                # retaining the original interaction and its queued successor.
                return self._ordinary_action(SemanticAction(
                    MOVE, action.object_index, action.object_id,
                    method_memory.goal.target_room_node_id, GOAL,
                ), obs, method_memory, agent_memory)
            if action.kind == COOP_PUT:
                return {"type": COOP_RELEASE, "object": action.object_id}, COOP_PUT
            return {"type": RELEASE, "object": action.object_id}, PUT
        return {"type": WAIT_PHYSICAL}, "unknown_semantic_action"

    @staticmethod
    def _goal_reached(method_memory, agent_position) -> bool:
        """Match the Benchmark's spatial precondition before releasing."""
        anchor = _position(method_memory.goal_anchor.position)
        target_room = method_memory.goal.target_room_node_id
        own = method_memory.agents.get(method_memory.agent_id)
        return bool(
            anchor is not None
            and target_room is not None
            and own is not None
            and own.room_node_id == target_room
            and _distance(agent_position, anchor) < GOAL_APPROACH_RADIUS
        )

    @classmethod
    def _delivery_reached(cls, method_memory, agent_position, coop_info):
        """The holder audits delivery; the supporter only prepares its pairing.

        A supporter may request Put early: release still requires the holder's
        matching request. Its own partial-view bed anchor must not veto that.
        """
        if not (coop_info.get("active") and coop_info.get("self_role") == "supporter"):
            return cls._goal_reached(method_memory, agent_position)
        holder = _position(coop_info.get("holder_pos"))
        return bool(holder is not None
                    and method_memory.goal.target_room_node_id is not None
                    and method_memory._room_id(holder) == method_memory.goal.target_room_node_id
                    and _distance(agent_position, holder) <= min(
                        2.0, cls._coop_recovery_distance(coop_info)))

    def _complete_interaction(self, obs: dict, method_memory) -> bool:
        current = self._unwrapped_action()
        if current is None:
            return False
        coop_info = obs.get("coop_info") or {}
        held_ids = {int(item["id"]) for item in (obs.get("held_objects") or [])
                    if isinstance(item, dict) and item.get("id") is not None}
        fallback = next((event for event in reversed(
            obs.get("coop_action_feedback") or []
        ) if isinstance(event, dict)
            and event.get("type") in ("ordinary_grasp_fallback_confirmed",
                                      "ordinary_grasp_fallback_failed")
            and event.get("object_id") is not None
            and current.object_id is not None
            and int(event["object_id"]) == int(current.object_id)), None)
        if current.kind == COOP_GRAB and fallback is not None:
            method_memory.mark_object_property(
                current.object_id, 1, "coop_grasp_ordinary_fallback"
            )
            self._replace_action_reference(current, None)
            self.pending_actions = []
            self.pending_interaction = None
            return True
        if (current.kind == COOP_GRAB and coop_info.get("active")
                and coop_info.get("object_id") == current.object_id):
            self._complete_action(current)
            return True
        if current.kind == GRAB and current.object_id in held_ids:
            method_memory.mark_object_property(
                current.object_id, 1, "ordinary_grasp_success"
            )
            self._complete_action(current)
            return True
        released_ids = {int(event["object_id"])
                        for event in (obs.get("coop_action_feedback") or [])
                        if isinstance(event, dict)
                        and event.get("type") == "coop_carrying_released"
                        and event.get("object_id") is not None}
        if current.kind == COOP_PUT and current.object_id in released_ids:
            method_memory.mark_object_finished(current.object_id)
            self._complete_action(current)
            self.pending_interaction = None
            return True
        transaction = (obs.get("action_feedback") or {}).get(
            "action_transaction"
        ) or {}
        if (current.kind == PUT and current.object_id is not None
                and self.pending_interaction == (PUT, current.object_id)
                and transaction.get("state") == "completed"
                and transaction.get("action_type") == RELEASE
                and transaction.get("object_id") is not None
                and int(transaction["object_id"]) == int(current.object_id)
                and current.object_id not in held_ids):
            method_memory.mark_object_finished(current.object_id)
            self._complete_action(current)
            self.pending_interaction = None
            return True
        return False

    def _bounded_recovery_reason(self, obs: dict, method_memory,
                                 frame_advanced):
        current = self._unwrapped_action()
        if self.last_mode == "grab_target_unknown" and current is not None:
            target = self._object_memory(current, method_memory)
            self.unknown_target_wait_count = (
                self.unknown_target_wait_count + 1
                if target is None or target.position is None else 0
            )
        else:
            self.unknown_target_wait_count = 0
        if self.unknown_target_wait_count >= self.config.target_position_wait_steps:
            return "grab_target_position_unavailable"

        feedback = obs.get("action_feedback") or {}
        reason = feedback.get("reason_code")
        submitted = (self.last_submitted_physical_action or {}).get("type")
        if reason == "coop_action_requires_pair" and submitted in (
            COOP_GRASP, COOP_RELEASE
        ):
            if frame_advanced is False:
                self.coop_pair_retry_count += 1
            else:
                self.coop_pair_retry_count = 0
        else:
            self.coop_pair_retry_count = 0
        if self.coop_pair_retry_count >= self.config.coop_pair_retry_steps:
            return "coop_pairing_retry_exhausted"
        return None

    def _complete_move(self, method_memory, coop_info=None):
        current = self._unwrapped_action()
        if current is None or current.kind != MOVE:
            return False, None
        if self.explore_state is not None:
            return False, None
        own = method_memory.agents[self.agent_id]
        if current.target_kind == ROOM:
            room = method_memory.rooms.get(current.target_room_node_id)
            reached = (room is not None and room.explored
                       and (current.required_revisit_count is None
                            or room.revisit_count
                            >= current.required_revisit_count))
            completed_room = current.target_room_node_id
        elif current.target_kind == OBJECT:
            item = self._object_memory(current, method_memory)
            reached = (item is not None and item.position is not None
                       and _distance(own.position, item.position)
                       <= OBJECT_APPROACH_RADIUS)
            completed_room = None
        elif current.target_kind == GOAL:
            reached = self._delivery_reached(method_memory, own.position, coop_info or {})
            completed_room = None
        else:
            reached, completed_room = False, None
        if not reached:
            return False, None
        self._complete_action(current)
        return True, completed_room

    def _complete_action(self, action) -> None:
        if self.pending_interaction == (action.kind, action.object_id):
            self.pending_interaction = None
        replacement = self.pending_actions.pop(0) if self.pending_actions else None
        self._replace_action_reference(action, replacement)

    @staticmethod
    def _unrecoverable_failure_reason(obs: dict):
        feedback = obs.get("action_feedback") or {}
        reason = feedback.get("reason_code")
        failed = (obs.get("valid") is False
                  or str(obs.get("action_status", "")).lower()
                  in ("invalid", "failed", "rejected"))
        if not failed or reason in RECOVERABLE_FAILURE_REASONS:
            return None
        return reason or "unclassified_execution_failure"

    def _replace_action_reference(self, old_action, new_action) -> None:
        current = self.semantic_action
        if current is old_action:
            self.semantic_action = new_action
            return
        while current is not None:
            if current.next_action is old_action:
                current.next_action = new_action
                return
            current = current.next_action

    @staticmethod
    def _object_memory(action, method_memory):
        if action.object_id is not None:
            for item in method_memory.objects.values():
                if item.object_id == action.object_id:
                    return item
        return method_memory.objects.get(action.object_index)

    @staticmethod
    def _previous_action_succeeded(obs: dict) -> bool:
        if obs.get("valid") is False:
            return False
        return (obs.get("status") == 1
                or str(obs.get("action_status", "")).lower().endswith("success"))

    @staticmethod
    def _previous_action_invalid(obs: dict) -> bool:
        return (obs.get("valid") is False
                or str(obs.get("action_status", "")).lower() == "invalid")

    def _navigate(self, agent_memory, target, *, radius=0.0, room_name=None):
        if target is None:
            return {"type": WAIT_PHYSICAL}
        obs = getattr(agent_memory, "obs", None) or {}
        frame = obs.get("current_frames")
        position = _position(agent_memory.position)
        current = self._unwrapped_action()
        key = (None if current is None else (current.kind, current.object_id,
               current.target_kind, current.target_room_node_id, current.required_revisit_count))
        state = self._navigation_progress
        distance = _distance(position, target)
        if (frame is not None and state and state["key"] == key
                and _distance(target, state["target"]) > 2 * self.config.cell_size):
            # A reselected endpoint changes the distance reference, not the
            # agent's progress. Preserve its stall window and recovery budget.
            state.update(target=tuple(target), best=distance)
        if frame is not None and (not state or state["key"] != key):
            reset = getattr(agent_memory, "reset_navigation", None)
            if callable(reset):
                reset()
            state = self._navigation_progress = dict(
                key=key, frame=frame, origin=position, target=tuple(target), best=distance,
                stalled=0, attempts=0)
        elif frame is not None and (frame > state["frame"] or state.get("path_missing")):
            # Count completed navigation opportunities, never message-only or
            # scan turns. A one-metre detour also counts as spatial progress.
            effective = (obs.get("action_feedback") or {}).get("effective_physical_action")
            effective = effective or self.last_submitted_physical_action or {}
            distance = _distance(position, state["target"])
            navigated = effective.get("type") in (0, 1, 2, "ongoing") and self.last_mode in (
                MOVE, "explore_approach", "coop_resume", "ongoing")
            envelope = (obs.get("action_feedback") or {}).get("requested_envelope") or {}
            navigated |= (state.get("path_missing", False) and effective.get("type") == 7
                          and not envelope.get("message"))
            if distance < state["best"] - 0.125 or _distance(position, state["origin"]) >= 1.0:
                state.update(origin=position, best=distance, stalled=0, attempts=0)
            elif navigated:
                state["stalled"] += 1
            state["frame"] = frame
            if state["stalled"] >= NAVIGATION_STALL_STEPS:
                state["attempts"] += 1
                state["stalled"] = 0
                self.navigation_recovery = dict(frame=frame, target=list(target),
                                               position=list(position), attempt=state["attempts"],
                                               navigation=getattr(agent_memory, "navigation_diagnostics", {}))
                if state["attempts"] > NAVIGATION_RECOVERY_ATTEMPTS:
                    self._navigation_failure = "navigation_no_progress"
                else:
                    recover = getattr(agent_memory, "recover_navigation", None)
                    if callable(recover):
                        recover()
                return {"type": WAIT_PHYSICAL}
        try:
            result = agent_memory.move_to_pos(target, radius=radius, room_name=room_name)
            action = result[0] if isinstance(result, tuple) else result
        except (IndexError, RuntimeError, TypeError, ValueError):
            action = None
        if state:
            state["path_missing"] = action is None
            previous_turn, previous_position = state.get("last_turn", (None, None))
            turn = (action or {}).get("type")
            alternating = (turn in (1, 2) and previous_turn in (1, 2)
                           and turn != previous_turn
                           and (self.last_submitted_physical_action or {}).get("type") == previous_turn
                           and _distance(position, previous_position) < self.config.cell_size)
            state["turn_reversals"] = state.get("turn_reversals", 0) + 1 if alternating else 0
            state["last_turn"] = (turn, position)
            if state["turn_reversals"] == 2:
                self.navigation_event = dict(
                    kind="turn_oscillation", frame=frame, target=list(target),
                    position=list(position), attempt=state["attempts"],
                    navigation=getattr(agent_memory, "navigation_diagnostics", {}))
        return action or {"type": WAIT_PHYSICAL}
