"""RLT hooks for Evan's existing ARX5 evaluation script.

This module keeps the robot control implementation unchanged.  It augments
policy requests with decision feedback and maps S/F to terminal RL rewards.
"""

from __future__ import annotations

import os
import socket
import sys
import uuid

import numpy as np

from openpi_client.websocket_client_policy import WebsocketClientPolicy as _OpenPIClient
from peripherals.keystroke_counter import KeystrokeCounter as _KeystrokeCounter
from peripherals.keystroke_counter import KeyCode
from modules.arx5_env import Arx5Env as _Arx5Env


class _EvanRLTCoordinator:
    def __init__(self):
        self.client: EvanRLTWebsocketClientPolicy | None = None
        self.session_id = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.episode_id = "unknown"
        self.episode_active = False

    def bind(self, client: "EvanRLTWebsocketClientPolicy") -> None:
        self.client = client

    def start_episode(self, episode_id: str) -> None:
        if self.client is not None:
            try:
                self.client.cancel_pending()
            except Exception as exc:
                print(f"[RLT Evan] stale decision cleanup failed: {exc}")
        self.episode_id = episode_id
        self.episode_active = True
        if self.client is not None:
            self.client.start_episode(self.session_id, episode_id)

    def finish_episode(self, success: bool) -> None:
        if not self.episode_active:
            return
        self.episode_active = False
        if self.client is not None:
            try:
                self.client.finish_episode(success)
            except Exception as exc:
                # RL logging must never prevent the robot environment from
                # ending and returning control to the operator.
                print(f"[RLT Evan] terminal feedback failed: {exc}")

    def record_executed_physical_action(self, actions: np.ndarray) -> None:
        if self.episode_active and self.client is not None:
            self.client.record_executed_physical_action(actions)

    def cancel_episode(self) -> None:
        self.episode_active = False
        if self.client is not None:
            try:
                self.client.cancel_pending()
            except Exception as exc:
                print(f"[RLT Evan] cancel feedback failed: {exc}")


_COORDINATOR = _EvanRLTCoordinator()


def _steps_per_inference_from_cli(default: int = 8) -> int:
    """Read the same Click option used by the base Evan script."""

    environment_value = os.environ.get("RLT_EVAN_EXECUTED_STEPS")
    if environment_value is not None:
        return int(environment_value)
    option_names = ("--steps_per_inference", "--steps-per-inference", "-si")
    for index, argument in enumerate(sys.argv):
        if argument in option_names and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        for option_name in option_names[:2]:
            prefix = f"{option_name}="
            if argument.startswith(prefix):
                return int(argument[len(prefix) :])
    return default


class EvanRLTWebsocketClientPolicy:
    """Drop-in OpenPI client that carries previous-decision feedback."""

    def __init__(self, host: str = "0.0.0.0", port: int | None = None, api_key: str | None = None):
        host = os.environ.get("RLT_EVAN_POLICY_HOST", host)
        configured_port = os.environ.get("RLT_EVAN_POLICY_PORT")
        if configured_port is not None:
            port = int(configured_port)
        if api_key is None:
            self._client = _OpenPIClient(host=host, port=port)
        else:
            self._client = _OpenPIClient(host=host, port=port, api_key=api_key)
        self._session_id = _COORDINATOR.session_id
        self._episode_id = "unknown"
        self._pending_decision_id: str | None = None
        self._pending_executed_physical_action: np.ndarray | None = None
        self._first_decision = True
        self._executed_steps = _steps_per_inference_from_cli()
        if self._executed_steps < 1:
            raise ValueError("steps_per_inference must be positive.")
        self._step_reward = float(os.environ.get("RLT_EVAN_STEP_REWARD", "0.0"))
        self._success_reward = float(os.environ.get("RLT_EVAN_SUCCESS_REWARD", "1.0"))
        self._failure_reward = float(os.environ.get("RLT_EVAN_FAILURE_REWARD", "0.0"))
        self._debug_actions = os.environ.get("RLT_EVAN_DEBUG_ACTIONS", "0") == "1"
        _COORDINATOR.bind(self)

    def get_server_metadata(self):
        return self._client.get_server_metadata()

    def reset(self) -> None:
        self.cancel_pending()

    def start_episode(self, session_id: str, episode_id: str) -> None:
        self._session_id = session_id
        self._episode_id = str(episode_id)
        self._pending_decision_id = None
        self._pending_executed_physical_action = None
        self._first_decision = True

    def record_executed_physical_action(self, actions: np.ndarray) -> None:
        """Attach the final post-processed commands submitted for this decision."""

        if self._pending_decision_id is None:
            return
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Executed physical actions must be rank 2, got {values.shape}.")
        if values.shape[0] < self._executed_steps:
            raise ValueError(
                f"The rollout submitted {values.shape[0]} physical actions, but "
                f"steps_per_inference={self._executed_steps}."
            )
        # The production rollout submits the full 16-step trajectory to the
        # controller, then replans after N steps and supersedes its tail. Only
        # the first N commands belong to this RLT decision transition.
        self._pending_executed_physical_action = np.array(
            values[: self._executed_steps], copy=True
        )

    def _nonterminal_feedback(self) -> dict | None:
        if self._pending_decision_id is None:
            return None
        feedback = {
            "decision_id": self._pending_decision_id,
            "rewards": np.full(self._executed_steps, self._step_reward, dtype=np.float32),
            "terminal": False,
        }
        if self._pending_executed_physical_action is not None:
            feedback["executed_physical_action"] = self._pending_executed_physical_action
        return feedback

    def infer(self, obs: dict) -> dict:
        request = dict(obs)
        if not _COORDINATOR.episode_active:
            request["rlt_warmup"] = True
        else:
            request["rlt_context"] = {
                "session_id": self._session_id,
                "episode_id": self._episode_id,
                "execution_horizon": self._executed_steps,
                "episode_start": self._first_decision,
            }
            feedback = self._nonterminal_feedback()
            if feedback is not None:
                request["rlt_feedback"] = feedback
        response = self._client.infer(request)
        if self._debug_actions and "actions" in response:
            actions = np.asarray(response["actions"])
            gripper = actions[: self._executed_steps, 9] if actions.ndim == 2 and actions.shape[1] > 9 else None
            print(
                "[RLT Evan debug]"
                f" source={response.get('rlt_action_source')}"
                f" actor_active={response.get('rlt_actor_active')}"
                f" policy_timing={response.get('policy_timing')}"
                f" physical_10d_gripper={gripper}"
            )
        if _COORDINATOR.episode_active:
            decision_id = response.get("rlt_decision_id")
            if decision_id is None:
                raise RuntimeError("Evan RLT server response is missing rlt_decision_id.")
            self._pending_decision_id = str(decision_id)
            self._pending_executed_physical_action = None
            self._first_decision = False
            action_count = len(np.asarray(response["actions"]))
            if self._executed_steps > action_count:
                raise ValueError(
                    f"steps_per_inference={self._executed_steps} exceeds the "
                    f"server action horizon {action_count}."
                )
        return response

    def finish_episode(self, success: bool) -> None:
        if self._pending_decision_id is None:
            return
        rewards = np.full(self._executed_steps, self._step_reward, dtype=np.float32)
        rewards[-1] = self._success_reward if success else self._failure_reward
        feedback = {
            "decision_id": self._pending_decision_id,
            "rewards": rewards,
            "terminal": True,
            "success": bool(success),
        }
        if self._pending_executed_physical_action is not None:
            feedback["executed_physical_action"] = self._pending_executed_physical_action
        response = self._client.infer(
            {
                "rlt_command": "finish_episode",
                "rlt_context": {
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                },
                "rlt_feedback": feedback,
            }
        )
        print(
            f"[RLT Evan] episode={self._episode_id} success={success} "
            f"status={response.get('rlt_feedback_status')}"
        )
        self._pending_decision_id = None
        self._pending_executed_physical_action = None

    def cancel_pending(self) -> None:
        if self._pending_decision_id is None:
            return
        self._client.infer(
            {
                "rlt_command": "cancel_episode",
                "rlt_context": {"session_id": self._session_id},
            }
        )
        self._pending_decision_id = None
        self._pending_executed_physical_action = None


class EvanRLTArx5Env(_Arx5Env):
    """Marks environment episode boundaries without changing ARX5 control."""

    def start_episode(self, start_time=None):
        episode_id = str(self.replay_buffer.n_episodes)
        result = super().start_episode(start_time)
        _COORDINATOR.start_episode(episode_id)
        return result

    def end_episode(self):
        # Any end not already labelled by S/F is a timeout, error, or interrupt.
        _COORDINATOR.finish_episode(success=False)
        return super().end_episode()

    def exec_actions(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        compensate_latency=False,
        dynamic_latency=False,
    ):
        result = super().exec_actions(
            actions=actions,
            timestamps=timestamps,
            compensate_latency=compensate_latency,
            dynamic_latency=dynamic_latency,
        )
        _COORDINATOR.record_executed_physical_action(np.asarray(actions))
        return result


class EvanRLTKeystrokeCounter(_KeystrokeCounter):
    """Adds explicit success/failure labels to the original key handling."""

    def get_press_events(self):
        events = super().get_press_events()
        mapped_events = []
        for event in events:
            if event == KeyCode(char="s"):
                _COORDINATOR.finish_episode(success=True)
                mapped_events.append(event)
            elif event == KeyCode(char="f"):
                _COORDINATOR.finish_episode(success=False)
                # The base Evan script already stops on S, so map F to S after
                # recording the different terminal label.
                mapped_events.append(KeyCode(char="s"))
            elif event == KeyCode(char="q"):
                _COORDINATOR.cancel_episode()
                mapped_events.append(event)
            else:
                mapped_events.append(event)
        return mapped_events
