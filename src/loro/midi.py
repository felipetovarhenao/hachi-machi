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
        self._numvoices = 0
        self._events = self._parse()

    def _parse(self) -> torch.Tensor:
        onset = 0
        last_onset = 0
        last_onset_per_voice = {}
        events = []
        voice_map = {}
        numvoices = 0
        active_notes = {}

        for msg in self._midi:
            onset += int(round(msg.time * 1000))
            if not msg.type.startswith('note_'):
                continue
            channel = msg.channel
            if channel not in voice_map:
                voice_map[channel] = numvoices
                numvoices += 1
            pitch = int(round(msg.note * 100))
            velocity = msg.velocity
            voice = voice_map[channel]
            key = (voice, pitch)
            if velocity > 0:
                ioi = onset - last_onset
                last_onset = onset
                voice_ioi = onset - last_onset_per_voice.get(voice, 0)
                last_onset_per_voice[voice] = onset
                active_notes[key] = [onset,
                                     ioi,
                                     voice_ioi,
                                     voice,
                                     pitch,
                                     velocity]
            elif key in active_notes:
                event: list = active_notes.pop(key)
                # duration = onset - event[0]
                # event.append(duration)
                events.append((event[0], event[1:]))
        self._numvoices = numvoices
        events.sort(key=lambda x: x[0])
        events.sort(key=lambda x: x[0])
        # Recompute ioi from sorted onsets
        corrected = []
        prev_onset = 0
        for onset, data in events:
            ioi = onset - prev_onset
            prev_onset = onset
            corrected.append([ioi] + data[1:])  # replace stale ioi
        return torch.tensor(corrected).to(torch.float32)

    def serialize(self, events: torch.Tensor, output_path: str, tempo: int = 500000) -> None:
        """
        Serialize parsed MIDI events back into a MIDI file.

        Expected tensor columns: [ioi, voice_ioi, voice, pitch, velocity, duration]
        Pitch values are assumed to be in units of 100 (as produced by _parse).
        """
        mid = mido.MidiFile(type=0, ticks_per_beat=self._midi.ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

        ticks_per_ms = self._midi.ticks_per_beat / (tempo / 1000)

        # Build flat list of (absolute_tick, msg) from events
        messages = []
        current_onset_ms = 0.0
        for row in events:
            ioi, _, voice, pitch, velocity = row.tolist()
            current_onset_ms += ioi
            duration = 1000
            note = int(round(pitch / 100))
            vel = int(round(velocity))
            channel = int(round(voice))
            onset_tick = int(round(current_onset_ms * ticks_per_ms))
            off_tick = int(round((current_onset_ms + duration) * ticks_per_ms))
            messages.append((onset_tick, mido.Message(
                'note_on',  channel=channel, note=note, velocity=vel,   time=0)))
            messages.append((off_tick,   mido.Message(
                'note_on',  channel=channel, note=note, velocity=0,     time=0)))

        messages.sort(key=lambda x: x[0])

        # Convert absolute ticks to delta ticks
        prev_tick = 0
        for abs_tick, msg in messages:
            msg.time = abs_tick - prev_tick
            prev_tick = abs_tick
            track.append(msg)

        mid.save(output_path)

    def events(self):
        return self._events

    def numvoices(self):
        return self._numvoices
