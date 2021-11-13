def mido_to_sustained_events(midi):
    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time

        if (
            message.type == "control_change"
            and message.control == 64
            and (message.value >= 64) != sustain
        ):
            # sustain pedal state has just changed
            sustain = message.value >= 64
            event_type = "sustain_on" if sustain else "sustain_off"
            event = dict(
                index=len(events), time=time, type=event_type, note=None, velocity=0
            )
            events.append(event)

        if "note" in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == "note_on" else 0
            event = dict(
                index=len(events),
                time=time,
                type="note",
                note=message.note,
                velocity=velocity,
                sustain=sustain,
            )
            events.append(event)
    return events


def events_to_notes(events):
    notes = []
    for i, onset in enumerate(events):
        if onset["velocity"] == 0:
            continue

        if onset == events[-1]:
            # TODO 왜 note-off되지 않는 note가 생겼지?
            continue

        # find the next note_off message
        offset = next(
            n for n in events[i + 1 :] if n["note"] == onset["note"] or n is events[-1]
        )

        if offset is not events[-1] and offset["sustain"]:
            # if the sustain pedal is active at offset, find when the sustain ends
            offset = next(
                n
                for n in events[offset["index"] + 1 :]
                if n["type"] == "sustain_off"
                or n["note"] == onset["note"]
                or n is events[-1]
            )

        note = (onset["time"], offset["time"], onset["note"], onset["velocity"])
        notes.append(note)
    return notes
