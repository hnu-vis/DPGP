"""One public Agent loop: facts -> delivered synchronization -> LLM -> A* -> execution."""

from dataclasses import asdict, replace
import json

from .astar_planner import PlanningError, project_prefixes, solve
from .config import MethodConfig
from .coordination import classify_changes
from .executor import MethodExecutor
from .memory import MethodMemory
from .prediction import OpenAIPredictionClient, PredictionError, aggregate_usage, resolve_with_llm
from .protocol import (decode_actions, decode_graph, decode_status, encode_actions,
                       encode_graph, encode_status, message_kind, attach_belief, split_belief, BaselineMismatch,
                       WireTemplate, attach_snapshot, split_snapshot, recovery_requested, _load)
from .discovery_belief import DiscoveryBelief
from .discovery_planner import EventPlanner


def llm_info(usage):
    attempts = usage.get("llm_attempts", [])
    return {"llm_call_count": len(attempts), "llm_prompt_tokens": usage.get("prompt_tokens", 0),
            "llm_completion_tokens": usage.get("completion_tokens", 0),
            "llm_total_tokens": usage.get("total_tokens", 0), "llm_wall_time": usage.get("wall_time", 0.),
            "llm_inference_time_estimated": usage.get("total_tokens", 0) / 50.,
            "llm_calls": attempts, "prediction_status": attempts[-1]["status"] if attempts else "not_called",
            "prediction_output": attempts[-1].get("output") if attempts else None}


class DPGP:
    agent_type = "DPGP"

    def __init__(self, agent_id, char_index, args, llm_client=None):
        mode = getattr(args, "dpgp_prediction_mode", "discovery_probability")
        if mode not in ("deterministic_llm", "discovery_probability"):
            raise ValueError("unsupported DPGP prediction mode")
        self.mode = mode
        self.agent_id, self.char_index, self.peer = agent_id, char_index, 1 - char_index
        self.config = MethodConfig(action_horizon=getattr(args, "planning_action_horizon", 8),
                                   message_limit=getattr(args, "communication_max_message_chars", 12000),
                                   discovery_candidates=getattr(args, "discovery_candidates", 6),
                                   discovery_samples=getattr(args, "discovery_samples", 8),
                                   discovery_time_horizon=getattr(args, "discovery_time_horizon", 120.),
                                   discovery_astar_expansions=getattr(args, "discovery_astar_expansions", 2000))
        self.send_rate = getattr(args, "communication_send_chars_per_second", 2500.)
        self.communication_mode = getattr(args, "communication_mode", "exclusive_action")
        if self.communication_mode not in ("exclusive_action", "parallel_side_channel"):
            raise ValueError("unsupported communication mode")
        self.seed = getattr(args, "seed", 1)
        self.method_memory = MethodMemory(agent_id, char_index, self.config)
        self.executor = MethodExecutor(char_index)
        self.client = llm_client or OpenAIPredictionClient(getattr(args, "lm_id", "gpt-4o-mini"),
            getattr(args, "t", 0.), getattr(args, "top_p", 1.), getattr(args, "max_tokens", 4096))
        self.initialized = False

    def reset(self, observation, containers_name, goal_objects_name, rooms_name, room_info, goal_spec):
        self.method_memory.reset(observation, containers_name, goal_objects_name, rooms_name, room_info, goal_spec)
        self.executor.reset()
        facts = self.method_memory.snapshot()
        self.template = WireTemplate(facts.rooms, facts.goals, tuple(sorted(set(containers_name))),
                                     self.config.boundary_tolerance)
        self.resync = False
        self.rx_round = -1
        self.goal_spec = goal_spec
        self.first = True
        self.epoch = -1
        self.awaiting = None
        self.wait_started = None
        self.assigned_idle = False
        self.tx_base = self.rx_base = None
        self.sent_graph = None  # Grounding basis survives an outbound delta reset.
        self.tx_seed = self.rx_seed = None
        self.pending_message = None
        self.previous = self.method_memory.snapshot()
        self.pending_events = set()
        self.stall_signature = None
        self.stall_steps = 0
        self.last_envelope = None
        self.last_plan = None
        self.belief = DiscoveryBelief()
        self.belief_epoch = -1
        self.initialized = True

    def _commitment_to_facts(self):
        graph = self.method_memory.graph
        agents = list(graph.agents)
        agents[self.char_index] = replace(agents[self.char_index], commitment=self.executor.commitment())
        self.method_memory.graph = replace(graph, agents=tuple(agents))

    def _request(self):
        self.executor.invalidate(keep_current=True)
        self._commitment_to_facts()
        self.epoch = max(self.epoch + 1, 2 * self.method_memory.step + self.char_index)
        self.awaiting, self.wait_started = self.epoch, self.method_memory.step
        self.assigned_idle = False
        facts = self.method_memory.snapshot()
        message = encode_graph(facts, round_id=self.epoch, base=self.tx_base, config=self.config,
                               template=self.template, recovery=self.resync)
        if self.mode == "discovery_probability":
            message = attach_belief(message, self.belief.to_seed(), model_epoch=self.belief_epoch,
                                    base=self.tx_seed, config=self.config, template=self.template)
        self.pending_events.clear()
        return message, facts

    def _plan(self, epoch):
        facts = self.method_memory.snapshot()
        probability = self.mode == "discovery_probability"
        if probability:
            usage = self.belief.synchronize(facts, self.client, agent_index=self.char_index, config=self.config)
            result = EventPlanner(facts, self.belief, self.config, seed=self.seed,
                                  communication_mode=self.communication_mode, send_rate=self.send_rate, template=self.template).plan()
            prefixes = result.prefixes
            self.belief_epoch = epoch
            assumptions = []
        else:
            resolved, usage = resolve_with_llm(facts, self.client, agent_index=self.char_index, config=self.config)
            result = solve(resolved, self.config)
            prefixes = project_prefixes(result, resolved, facts, self.config.action_horizon)
            assumptions = [asdict(a) for a in resolved.assumptions]
        self.executor.install(prefixes[self.char_index], epoch, facts.entities)
        self.pending_events.clear()
        self.assigned_idle = not bool(prefixes[self.char_index])
        self.awaiting = None
        self.epoch = epoch
        self.last_plan = {
            "status": "complete" if not any(facts.remaining(g) for g in facts.goals) else "planned",
            "epoch": epoch, "objective": result.objective, "makespan": result.makespan,
            "expanded": result.expanded, "planner_wall_time": result.wall_time,
            "prefixes": [[asdict(a) for a in p] for p in prefixes],
            "assumptions": assumptions,
            "planning_input": facts.to_dict(), "cost_model": self.config.to_dict(self.mode),
        }
        if probability:
            self.last_plan.update(discovery=result.diagnostics, belief=self.belief.to_seed(),
                                  belief_update=self.belief.last_diagnostics)
        print("[CWAHPlan] " + json.dumps({"agent": self.char_index, "step": facts.step, "epoch": epoch,
              "expanded": result.expanded, "wall_time": result.wall_time, "objective": result.objective,
              "prefix_lengths": [len(p) for p in prefixes], "llm_calls": len(usage.get("llm_attempts", ()))},
              ensure_ascii=False), flush=True)
        snapshot = facts if self.tx_base is None else None
        if prefixes[self.peer]:
            message = encode_actions(prefixes[self.peer], facts, round_id=epoch, recipient=self.peer,
                                     base=snapshot if snapshot is not None else self.rx_base, config=self.config, template=self.template)
        else:
            message = encode_status("complete" if self.last_plan["status"] == "complete" else "idle",
                                    round_id=epoch, recipient=self.peer, config=self.config, template=self.template)
        if snapshot is not None:
            message = attach_snapshot(message, snapshot, template=self.template, config=self.config)
        if probability:
            message = attach_belief(message, self.belief.to_seed(), model_epoch=self.belief_epoch,
                                    base=self.tx_seed, config=self.config, template=self.template)
        return message, usage, snapshot

    def _receive_belief(self, seed, epoch):
        if self.mode == "discovery_probability" and seed is not None and epoch >= self.belief_epoch:
            incoming = DiscoveryBelief.from_seed(seed, self.method_memory.snapshot())
            if epoch == self.belief_epoch and self.belief.task and incoming.to_seed() != self.belief.to_seed():
                raise ValueError("conflicting probability models in one planning epoch")
            self.belief, self.belief_epoch = incoming, epoch

    def get_action(self, observation, goal_spec):
        if not self.initialized:
            raise RuntimeError("DPGP.reset must be called before get_action")
        if goal_spec != self.goal_spec:
            raise ValueError("public goal changed during episode")
        memory = self.method_memory
        initial = self.first
        if self.first:
            self.first = False
        else:
            memory.update_from_observation(observation, self.last_envelope, commitment=self.executor.commitment())
        update = self.executor.observe(memory)
        if update.completed and update.completed.kind == "Put":
            memory.confirm_placement(update.completed)
        redirected = update.failure == "navigation_redirected"
        if redirected:
            # Replies to the superseded Move were planned from its old pose and
            # commitment. Start the existing resynchronization transaction from
            # the public post-redirect observation instead of reinstalling them.
            self.pending_message = self.awaiting = None
            self.tx_base = self.tx_seed = None
            self.resync = True
            self.epoch = max(self.epoch + 1, 2 * memory.step + self.char_index)
        self._commitment_to_facts()
        current = memory.snapshot()
        changes = classify_changes(self.previous, current)
        current_events = set(changes.categories) & {
            "room_exploration", "topology", "discovery", "container_state",
            "cooperation_property", "entity_removed",
        }
        if "container_state" in current_events:
            targets = {g.target_id for g in current.goals}
            old = {e.id: e for e in self.previous.entities}
            changed_search = any(e.id not in targets and e.searchable and e.id in old and
                                 (old[e.id].explored, old[e.id].open_state) != (e.explored, e.open_state)
                                 for e in current.entities)
            if not changed_search:
                current_events.remove("container_state")
        self.pending_events.update(current_events)
        usage = aggregate_usage(())
        outgoing = outgoing_graph = outgoing_seed = None
        request_reasons = []
        planned = received_plan = False
        # Promote a delta baseline only after public delivery feedback confirms it.
        if self.pending_message:
            text, graph, seed = self.pending_message
            feedback = memory.last_message_feedback
            if feedback.get("truncated"):
                raise RuntimeError("DPGP message was truncated despite its length check")
            if feedback.get("sent"):
                if graph is not None:
                    self.tx_base = self.sent_graph = graph
                self.tx_seed = seed
                self.pending_message = None
            elif "sent" in feedback:
                outgoing, outgoing_graph, outgoing_seed = text, graph, seed
                request_reasons = ["delivery_retry"]
            else:
                # An absent acknowledgement never establishes a delta baseline.
                self.pending_message = None
                self.tx_base = self.tx_seed = None
        incoming = memory.last_messages[self.peer] if len(memory.last_messages) > self.peer else None
        recovery = False
        if incoming:
            header = _load(incoming, self.config.message_limit, self.template)
            if header[1] > 2 * memory.step + 1:
                raise ValueError("message round is in the future")
            if header[1] <= self.rx_round or (redirected and header[1] < self.epoch):
                incoming = None
        if incoming:
            try:
                incoming, belief_seed, model_epoch = split_belief(incoming, self.config, base=self.rx_seed,
                                                                template=self.template)
                snapshot = split_snapshot(incoming, template=self.template, base=self.rx_base,
                                          expected_source=self.peer, config=self.config)
                kind = message_kind(incoming, self.config)
                if kind in ("F", "D"):
                    decoded = decode_graph(incoming, base=self.rx_base, expected_source=self.peer, config=self.config, template=self.template)
                elif kind == "A":
                    decoded = decode_actions(incoming, expected_recipient=self.char_index,
                                             base=snapshot if snapshot is not None else self.sent_graph, config=self.config, template=self.template)
                elif kind == "S":
                    decoded = decode_status(incoming, recipient=self.char_index, config=self.config, template=self.template)
                else:
                    raise ValueError("unsupported DPGP message")
            except BaselineMismatch:
                # Existing full request/reply transaction repairs either basis;
                # never apply a partial message or open an extra ACK channel.
                incoming = None
                self.tx_base = self.tx_seed = self.awaiting = self.pending_message = None
                outgoing = outgoing_graph = outgoing_seed = None
                recovery = self.resync = True
        if incoming:
            self.rx_round = header[1]
            self.rx_seed = belief_seed
            if snapshot is not None:
                memory.merge_received(snapshot)
                self.rx_base = snapshot
            if kind in ("F", "D"):
                epoch, peer_graph = decoded
                self.rx_base = peer_graph
                if recovery_requested(incoming, self.config):
                    self.tx_seed = self.tx_base = None
                self.resync = False
                memory.merge_received(peer_graph)
                self._commitment_to_facts()
                if epoch >= self.executor.epoch:
                    self._receive_belief(belief_seed, model_epoch)
                    collision = self.awaiting is not None
                    epoch = max(epoch, self.awaiting if collision else epoch)
                    outgoing = outgoing_graph = outgoing_seed = None
                    self.pending_message = None
                    if collision and self.char_index == 1:
                        self.awaiting, self.epoch = epoch, max(epoch, self.epoch)
                    else:
                        try:
                            outgoing, usage, outgoing_graph = self._plan(epoch)
                        except PredictionError as error:
                            # Fail visibly; an invalid model output must not turn
                            # into an indefinitely repeated empty action episode.
                            self.last_plan = {"status": "failed", "reason": str(error), "LLM": llm_info(error.usage)}
                            raise
                        planned = True
            elif kind == "A":
                packet = decoded
                if packet.round_id >= self.epoch and (self.awaiting is None or packet.round_id >= self.awaiting):
                    self._receive_belief(belief_seed, model_epoch)
                    memory.merge_grounding(packet.entities)
                    self.executor.install(packet.actions, packet.round_id, packet.entities)
                    self.epoch, self.awaiting, self.assigned_idle = packet.round_id, None, False
                    received_plan = True
                    self.resync = False
                    outgoing = outgoing_graph = outgoing_seed = None
                    self.pending_message = None
            elif kind == "S":
                epoch, status = decoded
                if epoch >= self.epoch:
                    self._receive_belief(belief_seed, model_epoch)
                    self.executor.install((), epoch)
                    self.epoch, self.awaiting, self.assigned_idle = epoch, None, True
                    received_plan = True
                    self.resync = False
                    outgoing = outgoing_graph = outgoing_seed = None
                    self.pending_message = None
            else:
                raise ValueError("unsupported DPGP message")
        facts = memory.snapshot()
        complete = not any(facts.remaining(g) for g in facts.goals)
        own = facts.agents[self.char_index]
        signature = (own.room_id, own.position, own.held, own.close,
                     self.executor.current, self.executor.phase,
                     tuple((e.id, e.room_id, e.inside, e.on, e.open_state, e.coop) for e in facts.entities))
        self.stall_steps = self.stall_steps + 1 if signature == self.stall_signature and self.executor.has_work else 0
        self.stall_signature = signature
        expected_wait = self.executor.phase in ("coop_wait", "coop_holder_wait", "coop_supporter_recover")
        stalled = self.stall_steps >= 5 and not expected_wait
        events = set(self.pending_events)
        global_change = bool(events)
        need = recovery or initial or update.failure or global_change or stalled or (
            not self.executor.has_work and not self.assigned_idle and not complete)
        # A holder owns new planning rounds while carrying. The supporter can
        # still receive graphs and execute the same deterministic planning call.
        supporter = facts.cooperation.object_id is not None and facts.cooperation.supporter == self.char_index
        if self.awaiting is not None and memory.step - self.wait_started >= 5 and not incoming:
            self.tx_base = self.tx_seed = None
            self.resync = True
            self.awaiting = None
            need = True
        # Agent 1 owns the initial request. The first reply carries agent 0's
        # full dynamic snapshot, so both directional graph bases are established.
        if initial and self.char_index == 0 and not complete:
            self.epoch = self.awaiting = 1
            self.wait_started = memory.step
        if need and not complete and self.awaiting is None and not (planned or received_plan or supporter) and outgoing is None:
            request_reasons = sorted(events) + [name for name, condition in (
                ("initial", initial), ("recovery", recovery or self.resync), ("failure", update.failure),
                ("stalled", stalled), ("prefix_exhausted", not self.executor.has_work and not self.assigned_idle))
                if condition]
            outgoing, outgoing_graph = self._request()
            self.stall_steps = 0
        if complete:
            self.executor.invalidate(keep_current=False)
        # A plan reply starts on the next step on both sides, including parallel
        # messaging mode. Sending a graph can preserve an existing physical action.
        physical = None if planned else self.executor.propose(memory)
        if outgoing and (self.communication_mode == "exclusive_action" or message_kind(outgoing, self.config) in ("A", "S")):
            physical = None
        envelope = {"physical_action": physical, "message": outgoing}
        if outgoing:
            seed = outgoing_seed if outgoing_seed is not None else (
                self.belief.to_seed() if self.mode == "discovery_probability" else None)
            self.pending_message = (outgoing, outgoing_graph, seed)
        self.executor.record_submitted(physical)
        self.last_envelope = envelope
        self._commitment_to_facts()
        self.previous = memory.snapshot()
        info = {"LLM": llm_info(usage), "graph": self.previous.to_dict(),
                "plan": self.last_plan, "graph_changes": list(changes.categories),
                "execution": {"phase": self.executor.phase, "action": asdict(self.executor.current) if self.executor.current else None,
                              "completed": asdict(update.completed) if update.completed else None, "failure": update.failure},
                "coordination": {"epoch": self.epoch, "awaiting": self.awaiting, "assigned_idle": self.assigned_idle,
                                 "planned_this_step": planned, "stall_steps": self.stall_steps,
                                 "planning_events": sorted(events), "request_reasons": request_reasons}}
        info["method_diagnostics"] = {"coordination": info["coordination"],
                                      "message_kind": message_kind(outgoing, self.config) if outgoing else None}
        return envelope, info
