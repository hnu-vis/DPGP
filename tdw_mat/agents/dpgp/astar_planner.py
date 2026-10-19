"""One deterministic event-driven joint A* over ``PlanningDomain``."""

from dataclasses import dataclass, replace
import heapq
from itertools import count
from typing import Any, Dict, Iterable, Optional, Tuple

from .models import DeterministicGraph, ObjectAssumption
from .planning_domain import (
    HighLevelAction,
    JointHighLevelAction,
    PlannerState,
    PlanningDomain,
    PlanningDomainError,
)


STATUS_ACTION = "action"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
COST_EPSILON = 1e-9


@dataclass(frozen=True)
class PlanningResult:
    status: str
    selected_action: Optional[JointHighLevelAction]
    total_cost: Optional[float]
    joint_actions: Tuple[JointHighLevelAction, ...]
    expanded_states: int
    assumptions: Tuple[ObjectAssumption, ...]
    reopened_room_ids: Tuple[int, ...]
    reason: Optional[str]
    # Absolute scan requirements of continuing actions, not new inspections.
    continued_scans: Tuple[Tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in (STATUS_ACTION, STATUS_COMPLETE, STATUS_FAILED):
            raise ValueError("unknown planning status: {}".format(self.status))
        if self.status == STATUS_ACTION and self.selected_action is None:
            raise ValueError("action status requires selected_action")
        if self.status != STATUS_ACTION and self.selected_action is not None:
            raise ValueError("only action status may contain selected_action")
        if self.expanded_states < 0:
            raise ValueError("expanded_states must be non-negative")

    @property
    def should_execute(self) -> bool:
        return self.status == STATUS_ACTION

    @property
    def succeeded(self) -> bool:
        return self.status in (STATUS_ACTION, STATUS_COMPLETE)

    def agent_action_prefix(self, agent_id: int,
                            max_actions: int) -> Tuple[HighLevelAction, ...]:
        """Project a bounded executable prefix; protocol adds runtime IDs later."""
        if agent_id not in (0, 1):
            raise ValueError("agent_id must be 0 or 1")
        if type(max_actions) is not int or max_actions <= 0:
            raise ValueError("max_actions must be a positive integer")
        result = []
        for joint in self.joint_actions:
            action = joint.agent_0 if agent_id == 0 else joint.agent_1
            if action is None:
                continue
            result.append(action)
            if len(result) >= max_actions:
                break
        return tuple(result)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "selected_action": (self.selected_action.to_dict()
                                if self.selected_action is not None else None),
            "total_cost": self.total_cost,
            "joint_actions": [item.to_dict() for item in self.joint_actions],
            "expanded_states": self.expanded_states,
            "assumptions": [
                {"graph_index": item.graph_index,
                 "room_node_id": item.room_node_id,
                 "ordinary_property": item.ordinary_property}
                for item in self.assumptions
            ],
            "reopened_room_ids": list(self.reopened_room_ids),
            "reason": self.reason,
            **({"continued_scans": [list(item) for item in self.continued_scans]}
               if self.continued_scans else {}),
        }


@dataclass(frozen=True, eq=False)
class _SearchNode:
    state: PlannerState
    parent: Optional["_SearchNode"]
    action: Optional[JointHighLevelAction]


class AStarPlanner:
    def __init__(self, graph: DeterministicGraph, navigation=None,
                 root_state: Optional[PlannerState] = None, commitments=()):
        self.graph = graph
        self.domain = PlanningDomain(graph, navigation, commitments)
        self.root_state = root_state or self.domain.initial_state
        expected = {item.graph_index for item in self.domain.initial_state.objects}
        if ({item.graph_index for item in self.root_state.objects} != expected
                or tuple(item.agent_id for item in self.root_state.agents) != (0, 1)):
            raise PlanningDomainError("root state does not match planning scope")

    @staticmethod
    def _reconstruct(node: _SearchNode) -> Tuple[JointHighLevelAction, ...]:
        actions = []
        while node.parent is not None:
            if node.action is None:
                raise PlanningDomainError("non-root search node has no action")
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return tuple(actions)

    def _result(self, status: str, selected_action=None, total_cost=None,
                joint_actions=(), expanded_states=0, reason=None):
        return PlanningResult(
            status, selected_action, total_cost, tuple(joint_actions),
            expanded_states, self.graph.assumptions,
            self.graph.reopened_room_ids, reason,
        )

    def solve(self, allowed_root_actions: Optional[
              Iterable[JointHighLevelAction]] = None,
              max_expansions: Optional[int] = None) -> PlanningResult:
        """Run one exact A* with a feasible greedy upper bound."""
        start = self.root_state
        if start.is_goal():
            return self._result(STATUS_COMPLETE, total_cost=0.0)
        root_filter = (None if allowed_root_actions is None
                       else set(allowed_root_actions))

        def actions_for(node):
            actions = self.domain.legal_joint_actions(node.state)
            if node.parent is None and root_filter is not None:
                actions = tuple(item for item in actions if item in root_filter)
            return actions

        def greedy_incumbent():
            node, cost = _SearchNode(start, None, None), 0.0
            seen = {self.domain.state_key(start)}
            for _ in range(256):
                if node.state.is_goal():
                    return cost, self._reconstruct(node)
                candidates = []
                for action in actions_for(node):
                    next_state, elapsed = self.domain.advance(node.state, action)
                    next_key = self.domain.state_key(next_state)
                    if next_key in seen:
                        continue
                    edge = self.domain.transition_cost(node.state, elapsed)
                    candidates.append((
                        edge + self.domain.heuristic(next_state),
                        sum(not item.finished for item in next_state.objects),
                        self.domain._joint_key(action), action, next_state, edge,
                    ))
                if not candidates:
                    return None
                _, _, _, action, next_state, edge = min(
                    candidates, key=lambda item: item[:3]
                )
                node = _SearchNode(next_state, node, action)
                cost += edge
                seen.add(self.domain.state_key(next_state))
            return None

        incumbent = greedy_incumbent()
        upper = float("inf") if incumbent is None else incumbent[0]
        order = count()
        root = _SearchNode(start, None, None)
        queue = [(self.domain.heuristic(start), -0.0, next(order), root)]
        best = {self.domain.state_key(start): 0.0}
        expanded = 0
        bounded = False
        while queue:
            estimate, negative_cost, _, node = heapq.heappop(queue)
            cost = -negative_cost
            if estimate >= upper - COST_EPSILON:
                break
            if max_expansions is not None and expanded >= max_expansions:
                bounded = True
                break
            key = self.domain.state_key(node.state)
            best_cost = best.get(key)
            if best_cost is None or cost > best_cost + COST_EPSILON:
                continue
            if node.state.is_goal():
                actions = self._reconstruct(node)
                if not actions:
                    return self._result(
                        STATUS_FAILED, expanded_states=expanded,
                        reason="unfinished scenario returned no first action",
                    )
                return self._result(
                    STATUS_ACTION,
                    selected_action=actions[0],
                    total_cost=cost,
                    joint_actions=actions,
                    expanded_states=expanded,
                )
            expanded += 1
            for action in actions_for(node):
                next_state, elapsed = self.domain.advance(node.state, action)
                next_cost = cost + self.domain.transition_cost(node.state, elapsed)
                next_key = self.domain.state_key(next_state)
                previous = best.get(next_key)
                estimate = next_cost + self.domain.heuristic(next_state)
                if (previous is not None
                        and next_cost >= previous - COST_EPSILON):
                    continue
                if estimate >= upper - COST_EPSILON:
                    continue
                best[next_key] = next_cost
                child = _SearchNode(next_state, node, action)
                heapq.heappush(queue, (
                    estimate,
                    -next_cost,
                    next(order),
                    child,
                ))
        if incumbent is not None:
            cost, actions = incumbent
            return self._result(
                STATUS_ACTION, selected_action=actions[0], total_cost=cost,
                joint_actions=actions, expanded_states=expanded,
                reason="search_budget_feasible" if bounded else None,
            )
        reason = ("search_budget_exhausted" if bounded else
                  "no plan reaches all unfinished objects")
        if root_filter is not None:
            reason += " under the allowed root actions"
        return self._result(
            STATUS_FAILED, expanded_states=expanded, reason=reason
        )


def plan_deterministic_graph(
    graph: DeterministicGraph,
    allowed_root_actions: Optional[Iterable[JointHighLevelAction]] = None,
    navigation=None,
    root_state: Optional[PlannerState] = None,
    max_expansions: Optional[int] = None,
    commitments=(),
) -> PlanningResult:
    """Validate and solve exactly one already-completed deterministic graph."""
    try:
        planner = AStarPlanner(graph, navigation, root_state, commitments)
        result = planner.solve(
            allowed_root_actions, max_expansions=max_expansions
        )
        if result.should_execute and any(planner.domain.initial_actions):
            first = result.joint_actions[0]
            initial = planner.domain.initial_actions
            first = JointHighLevelAction(first.agent_0 or initial[0], first.agent_1 or initial[1])
            result = replace(result, selected_action=first,
                             joint_actions=(first,) + result.joint_actions[1:],
                             continued_scans=planner.domain.continued_scans)
        return result
    except (PlanningDomainError, ValueError, TypeError) as error:
        assumptions = graph.assumptions if isinstance(graph, DeterministicGraph) else ()
        reopened = graph.reopened_room_ids if isinstance(graph, DeterministicGraph) else ()
        return PlanningResult(
            STATUS_FAILED, None, None, (), 0, assumptions, reopened,
            "deterministic planning failed: {}".format(error),
        )


def plan_known_fact_graph(facts, navigation=None, revealed_rooms=None,
                          reopened_room_ids=(), root_state=None):
    """Solve only factually located or explicitly revealed target slots."""
    graph = DeterministicGraph.for_known_facts(
        facts, revealed_rooms, reopened_room_ids
    )
    return graph, plan_deterministic_graph(
        graph, navigation=navigation, root_state=root_state
    )
