import os
import mido
import torch


class MidiParser:
    def __init__(self, file: str):
        file_path = os.path.abspath(file)
        ext = os.path.splitext(file_path)[1]
        if ext not in ['.mid', '.midi']:
            raise TypeError(
                f"Invalid file extension: {ext}. Expected .mid or .midi")
        midi = mido.MidiFile(filename=file)
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
                                     voice,
                                     pitch,
                                     velocity,
                                     ioi,
                                     voice_ioi]
            elif key in active_notes:
                event: list = active_notes.pop(key)
                duration = onset - event[0]
                event.append(duration)
                events.append((event[0], event[1:]))
        self._numvoices = numvoices
        events.sort(key=lambda x: x[0])
        return torch.tensor([e for _, e in events]).to(torch.float32)

    def events(self):
        return self._events

    def numvoices(self):
        return self._numvoices
