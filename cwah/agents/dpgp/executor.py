"""Feedback-driven grounding. Only submitted physical actions can fail/complete."""

from collections import deque
from dataclasses import dataclass
import numpy as np

from .models import Action, Commitment

RECOVERABLE = {
    "object_not_close", "target_not_close", "coop_action_requires_pair", "mismatched_coop_action",
    "coop_grab_too_far", "coop_grab_agents_too_far", "collaborators_not_close_for_coop_interaction",
    "holder_not_close_to_target", "supporter_not_close_for_coop_put", "coop_carrying_paused_distance",
    "target_closed", "target_already_open", "coop_grab_target_mismatch", "coop_put_target_mismatch",
}


@dataclass(frozen=True)
class ExecutionUpdate:
    completed: object = None
    failure: object = None


class MethodExecutor:
    def __init__(self, index):
        self.index = index
        self.reset()

    def reset(self):
        self.current = None
        self.queue = deque()
        self.epoch = -1
        self.phase = "idle"
        self.submitted = None
        self.grounding = {}
        self.failure = None

    @property
    def has_work(self):
        return self.current is not None or bool(self.queue)

    def commitment(self):
        return Commitment(self.current, self.phase, max(0, self.epoch)) if self.current else None

    def _non_preemptive(self):
        return bool(self.current and (
            self.current.kind == "Move" or
            self.current.kind == "Grab" and self.current.cooperative and self.phase == "coop_wait"
        ))

    def invalidate(self, *, keep_current=True):
        self.queue.clear()
        # Keep physical navigation and an established public pair request only.
        if not keep_current or not self._non_preemptive():
            self.current = None
            self.phase = "idle"

    def install(self, actions, epoch, entities=()):
        if epoch < self.epoch:
            return False
        actions = [a for a in actions if a.kind not in ("Wait", "Continue")]
        self.grounding.update({e.id: e for e in entities})
        if self.current and actions and actions[0] == self.current:
            actions.pop(0)
        elif self.current and self.current.kind == "Move":
            # Search can regard the target coordinate as reached while Unity has
            # not confirmed CLOSE. Finish this real Move before the new prefix.
            pass
        elif self._non_preemptive():
            # Prefix projection carries valid commitments. Refuse replacement
            # instead of quietly abandoning an established coop request.
            if actions:
                raise ValueError("new plan does not continue the active commitment")
        else:
            self.current = None
            self.phase = "idle"
        self.queue = deque(actions)
        self.epoch = epoch
        self.failure = None
        return True

    def entity(self, memory, key):
        local = next((e for e in memory.graph.entities if e.id == key), None)
        remote = self.grounding.get(key)
        if local is None or (remote is not None and remote.stamp.step > local.stamp.step):
            return remote
        return local

    def _done(self, memory, action):
        facts, own = memory.graph, memory.graph.agents[self.index]
        entities = {e.id: e for e in facts.entities}
        if action.kind == "Move":
            return own.room_id == action.target_id if action.target_id in {r.id for r in facts.rooms} else action.target_id in own.close
        if action.kind == "CheckContainer":
            entity = entities.get(action.target_id)
            return bool(entity and entity.open_state == "open")
        if action.kind == "Grab":
            return (action.target_id in own.held or facts.cooperation.object_id == action.target_id)
        if action.kind == "Put":
            return any(g.target_id == action.target_id and action.object_id in facts.satisfied(g) for g in facts.goals)
        return False

    def observe(self, memory):
        if self.current is None:
            self.submitted = None
            return ExecutionUpdate()
        feedback = memory.last_feedback
        requested = feedback.get("requested_physical_action") or (
            feedback.get("requested_envelope") or {}).get("physical_action")
        matched = bool(self.submitted and requested == self.submitted)
        put_succeeded = (self.current.kind == "Put" and self.phase == "submit_put" and matched
                         and feedback.get("valid") is True
                         and feedback.get("action_status") == "success")
        if self._done(memory, self.current) or put_succeeded:
            finished = self.current
            self.current, self.phase, self.submitted = None, "idle", None
            return ExecutionUpdate(completed=finished)
        recovery = feedback.get("navigation_recovery")
        if (self.current.kind == "Move" and not self.current.cooperative and matched
                and isinstance(recovery, dict)
                and recovery.get("original_physical_action") == self.submitted):
            # The public environment has executed a different navigation action.
            # Release only this local Move commitment; its target remains a fact
            # and can be selected again from the observed post-redirect pose.
            self.current, self.phase, self.submitted = None, "idle", None
            self.queue.clear()
            self.failure = "navigation_redirected"
            return ExecutionUpdate(failure=self.failure)
        if self.current.kind == "Grab" and self.current.cooperative:
            entity = self.entity(memory, self.current.target_id)
            property_value = (entity.coop or dict(memory.graph.class_coop).get(entity.name)) if entity else None
            if property_value == 1 and entity.holders and self.index not in entity.holders:
                self.current, self.phase, self.submitted = None, "idle", None
                self.queue.clear()
                return ExecutionUpdate(failure="ordinary_fallback_supporter_released")
        reason = None
        if matched and (feedback.get("valid") is False or
                feedback.get("action_status") in ("failed", "invalid", "rejected")):
            reason = feedback.get("invalid_reason") or feedback.get("reason_code") or "action_failed"
        if reason == "coop_action_requires_pair" and self.current.kind == "Grab":
            self.phase = "coop_wait"
        if reason == "navigation_stalled" and memory.graph.rooms:
            reason = None  # Keep the action for one random-room recovery proposal.
        if reason and reason not in RECOVERABLE:
            self.current, self.phase = None, "failed"
            self.queue.clear()
            self.failure = reason
        else:
            reason = None
        self.submitted = None
        return ExecutionUpdate(failure=reason)

    def _activate(self, memory):
        # Skip actions already satisfied by a concurrent open or just-completed
        # commitment. This does not fabricate success for an unexecuted grab/put.
        while self.current is None and self.queue:
            action = self.queue.popleft()
            if not self._done(memory, action):
                self.current, self.phase = action, "ready"

    @staticmethod
    def _one(kind, name, key):
        return "[{}] <{}> ({})".format(kind, name, key)

    def _walk(self, memory, key):
        rooms = {r.id: r for r in memory.graph.rooms}
        if key in rooms:
            own = memory.graph.agents[self.index]
            pending, seen = deque([(own.room_id,)]), {own.room_id}
            while pending:
                path = pending.popleft()
                if path[-1] == key:
                    target = path[1] if len(path) > 1 else key
                    return self._one("walktowards", rooms[target].name, target)
                neighbors = sorted({p.target if p.source == path[-1] else p.source for p in memory.graph.portals
                                    if path[-1] in (p.source, p.target)})
                for neighbor in neighbors:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        pending.append(path + (neighbor,))
            raise ValueError("no known adjacent-room route")
        entity = self.entity(memory, key)
        if entity is None:
            raise ValueError("missing executable entity grounding")
        if entity.room_id != memory.graph.agents[self.index].room_id:
            return self._walk(memory, entity.room_id)
        return self._one("walktowards", entity.name, key)

    def propose(self, memory):
        self._activate(memory)
        action = self.current
        if action is None:
            self.phase = "idle"
            return None
        if memory.last_feedback.get("invalid_reason") == "navigation_stalled":
            rooms = memory.graph.rooms
            if rooms:
                room = rooms[np.random.randint(0, max(1, len(rooms) - 1))]
                self.phase = "navigation_redirect"
                return self._one("walktowards", room.name, room.id)
        coop = memory.last_coop_info
        if coop.get("active") and (coop.get("distance_level") == "hard" or coop.get("status") == "paused_distance"):
            if coop.get("holder_agent") == self.index:
                self.phase = "coop_holder_wait"
                return None
            self.phase = "coop_supporter_recover"
            return self._one("walktowards", "character", int(coop["holder_agent"]) + 1)
        own = memory.graph.agents[self.index]
        if action.kind == "Move":
            self.phase = "navigate"
            return self._walk(memory, action.target_id)
        target = self.entity(memory, action.target_id)
        if target is None:
            raise ValueError("missing interaction grounding")
        if action.kind == "Grab":
            parent = self.entity(memory, target.inside)
            if parent and parent.container and parent.open_state == "closed":
                if parent.id not in own.close:
                    self.phase = "navigate_container"
                    return self._walk(memory, parent.id)
                self.phase = "open_container"
                return self._one("open", parent.name, parent.id)
            if action.target_id not in own.close:
                if parent and parent.container and parent.id not in own.close:
                    self.phase = "navigate_container"
                    return self._walk(memory, parent.id)
                self.phase = "navigate_object"
                return self._walk(memory, target.id)
            cooperative = action.cooperative or target.coop == 2 or dict(memory.graph.class_coop).get(target.name) == 2
            self.phase = "coop_wait" if cooperative and self.phase == "coop_wait" else "submit_grab"
            return self._one("coop_grab" if cooperative else "grab", target.name, target.id)
        if target.id not in own.close:
            self.phase = "navigate_target"
            return self._walk(memory, target.id)
        if action.kind == "CheckContainer":
            self.phase = "open"
            return self._one("open", target.name, target.id)
        if action.kind == "Put":
            obj = self.entity(memory, action.object_id)
            if obj is None:
                raise ValueError("missing held object grounding")
            goal = next(g for g in memory.graph.goals if g.target_id == target.id and g.object_class == obj.name)
            if goal.relation == "INSIDE" and target.open_state == "closed":
                self.phase = "open_put_target"
                return self._one("open", target.name, target.id)
            cooperative = memory.graph.cooperation.object_id == obj.id
            # Public ordinary fallback releases the supporter and leaves the
            # actual holder responsible; stale planned coop flags are not facts.
            if not cooperative and obj.id not in own.held:
                raise ValueError("Put has no publicly confirmed holder")
            kind = "putin" if goal.relation == "INSIDE" else "putback"
            if cooperative:
                kind = "coop_" + kind
            self.phase = "submit_put"
            return "[{}] <{}> ({}) <{}> ({})".format(kind, obj.name, obj.id, target.name, target.id)
        raise ValueError("unsupported executable action")

    def record_submitted(self, physical):
        self.submitted = physical
