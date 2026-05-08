import mido
import torch
from .utils import validate_path


class MidiParser:
    def __init__(self, file: str):
        file_path = validate_path(file, ['.mid', '.midi'])
        midi = mido.MidiFile(filename=file_path)
        if midi.type == 2:
            raise TypeError(
                'Invalid MIDI file type 2. Expected type 0 (single track) or 1 (multi-track).')

        midi.tracks = [mido.merge_tracks(midi.tracks)]
        self._midi = midi
        self._events = self._parse()

    def _parse(self) -> torch.Tensor:
        onset = 0
        events = []
        active_notes = {}
        channels = set()
        for msg in self._midi:
            onset += msg.time
            if not msg.type.startswith('note_'):
                continue
            channel = msg.channel
            channels.add(channel)
            pitch = msg.note
            key = (channel, pitch)
            if msg.velocity > 0:
                active_notes[key] = (onset, channel, pitch, msg.velocity)
            elif key in active_notes:
                note = active_notes.pop(key)
                duration = onset - note[0]
                events.append((*note, duration))

        self.channels = list(channels)
        events.sort(key=lambda x: x[0])

        prev_onset = 0
        rows = []
        for onset, chan, pitch, velocity, duration in events:
            ioi = onset - prev_onset
            prev_onset = onset
            rows.append([ioi,
                         chan,
                         pitch,
                         velocity,
                         duration,
                         ])

        return torch.tensor(rows, dtype=torch.float32)

    @classmethod
    def render(cls,
               events: torch.Tensor,
               output_path: str,
               tempo: int = 500_000,
               ticks_per_beat: int = 480) -> None:
        mid = mido.MidiFile(type=0,
                            ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo',
                                      tempo=tempo,
                                      time=0))

        ticks_per_ms = ticks_per_beat / (tempo / 1000)

        messages = []
        current_onset_ms = 0.0
        for row in events:
            ioi, chan, pitch, velocity, duration = row.tolist()
            current_onset_ms += ioi
            duration = 1000
            note = int(round(pitch))
            vel = int(round(velocity))
            channel = int(round(chan))
            onset_tick = int(round(current_onset_ms * ticks_per_ms))
            off_tick = int(round((current_onset_ms + duration) * ticks_per_ms))
            messages.append((onset_tick, mido.Message(type='note_on',
                                                      channel=channel,
                                                      note=note,
                                                      velocity=vel,
                                                      time=0)))
            messages.append((off_tick,   mido.Message(type='note_on',
                                                      channel=channel,
                                                      note=note,
                                                      velocity=0,
                                                      time=0)))

        messages.sort(key=lambda x: x[0])
        prev_tick = 0
        for abs_tick, msg in messages:
            msg.time = abs_tick - prev_tick
            prev_tick = abs_tick
            track.append(msg)

        mid.save(output_path)

    def events(self):
        return self._events
