import torch
import heapq
from dataclasses import dataclass, field
from typing import Optional
from .console import Console
from .utils import progress


@dataclass(order=True)
class Event:
    event_tensor: torch.Tensor = field(compare=False)
    is_user: bool = field(compare=False)


class OfflineSession:

    ABS_DELTA, VOICE_DELTA, VOICE_ID, MIDICENTS, VELOCITY, DURATION = range(6)
    USER_VOICE = 0

    @classmethod
    def run(cls, model, user_events: torch.Tensor) -> torch.Tensor:
        model.eval()
        display = Console.get_display()
        device = user_events.device
        user_abs_times = user_events[..., cls.ABS_DELTA].clone().cumsum(0)
        max_time = int(user_abs_times[-1].item())
        queue: list[tuple[float, int, Event]] = []
        counter = 0

        def push(abs_time: float, event: Event):
            nonlocal counter
            heapq.heappush(queue, (abs_time, counter, event))
            counter += 1

        for i, abs_time in enumerate(user_abs_times):
            push(abs_time=abs_time,
                 event=Event(
                     event_tensor=user_events[i].clone(),
                     is_user=True
                 ))

        emitted: list[tuple[float, torch.Tensor]] = []
        last_abs_time: float = 0.0
        last_voice_time: dict[int, float] = {}

        def compute_deltas(abs_time: float, voice_id: int) -> tuple[float, float]:
            abs_delta = abs_time - last_abs_time
            voice_delta = abs_time - last_voice_time.get(voice_id, abs_time)
            return abs_delta, voice_delta

        def record_event(abs_time: float, feat6: torch.Tensor):
            nonlocal last_abs_time
            emitted.append((abs_time, feat6.clone()))
            last_abs_time = abs_time
            last_voice_time[int(feat6[cls.VOICE_ID].item())] = abs_time

        def run_model(feat5: torch.Tensor) -> Optional[torch.Tensor]:
            with torch.no_grad():
                inp = feat5.view(1, 1, 5).float()
                out = model(inp)
            return out

        while queue:

            abs_time, _, event = heapq.heappop(queue)
            feat = event.event_tensor
            voice_id = int(feat[cls.VOICE_ID].item())

            abs_delta, voice_delta = compute_deltas(abs_time, voice_id)
            feat5 = feat[:5].clone()
            feat5[cls.ABS_DELTA] = abs_delta
            feat5[cls.VOICE_DELTA] = voice_delta

            duration = feat[cls.DURATION].item()
            feat6 = torch.cat([feat5, torch.tensor([duration]).to(device)])
            record_event(abs_time, feat6)
            display.update(progress=progress(int(abs_time), max_time))
            pred = run_model(feat5)

            if pred is not None:
                pred_feat = pred.squeeze()
                pred_dt = pred_feat[cls.ABS_DELTA].item()
                scheduled_time = abs_time + pred_dt
                push(scheduled_time, Event(
                    event_tensor=pred_feat.clone(),
                    is_user=False
                ))

        return torch.stack([f for _, f in emitted])
