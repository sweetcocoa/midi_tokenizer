import pretty_midi
import numpy as np
from numba import jit
import mido
import midi_utils

TOKEN_SPECIAL: int = 0
TOKEN_NOTE: int = 1
TOKEN_VELOCITY: int = 2
TOKEN_TIME: int = 3

EOS: int = 1
PAD: int = 0


@jit(nopython=True, cache=True)
def fast_detokenize(
    idx, n_special, n_note, n_velocity, time_resolution, time_offset, decimal
):
    """
    integer -> corresponding token

    idx : integer
    n_special, n_note, n_velocity : word vocabulary size
    time_resolution
    time_offset :

    return : token type, value(absolute time | velocity | pitch)
    """
    if idx >= n_special + n_note + n_velocity:
        # TODO decimals
        return (
            TOKEN_TIME,
            np.round(
                (idx - (n_special + n_note + n_velocity)) * time_resolution
                + time_offset,
                decimal,
            ),
        )
    elif idx >= n_special + n_note:
        return TOKEN_VELOCITY, idx - (n_special + n_note)
    elif idx >= n_special:
        return TOKEN_NOTE, idx - n_special
    else:
        return TOKEN_SPECIAL, idx


@jit(nopython=True, cache=True)
def fast_split(
    tokens: np.array,
    time_from,
    time_to,
    n_special,
    n_note,
    n_velocity,
    time_resolution,
    decimal,
):
    idx_from = -1
    idx_to = len(tokens) + 1
    index_offset = int(time_from / time_resolution)

    for i in range(len(tokens)):
        type, time = fast_detokenize(
            idx=tokens[i],
            n_special=n_special,
            n_note=n_note,
            n_velocity=n_velocity,
            time_resolution=time_resolution,
            time_offset=0.0,
            decimal=decimal,
        )
        idx_to = i
        if type == TOKEN_TIME:
            if time >= time_from and idx_from == -1:
                # start
                idx_from = i
            if time >= time_to:
                # end
                break
            tokens[i] -= index_offset
    tokens[idx_to] = EOS
    return tokens[idx_from : idx_to + 1]


@jit(nopython=True, cache=True)
def fast_tokenize(idx, type, n_special, n_note, n_velocity):
    if type == TOKEN_TIME:
        return n_special + n_note + n_velocity + idx
    elif type == TOKEN_VELOCITY:
        return n_special + n_note + idx
    elif type == TOKEN_NOTE:
        return n_special + idx
    elif type == TOKEN_SPECIAL:
        return idx
    else:
        return -1


@jit(nopython=True, cache=True)
def get_reduced_velocity(n_velocity, value):
    if n_velocity == 2:  # binary velocity
        if value > 0:
            return 1
        else:
            return value
    else:
        return value


@jit(nopython=True, cache=True)
def get_original_velocity(n_velocity, value):
    if n_velocity == 2:  # binary velocity
        if value > 0:
            return 64
        else:
            return value
    else:
        return value


class Tokenizer:
    def __init__(self, tokenizer_config):
        self.config = tokenizer_config

    def split(self, tokens: np.array, time_from, time_to):
        """
        Split tokens from {time_from} to {time_to}

        And also modify
        - Shift time tokens' value (by time_from)
        - Adds EOS Token at last
        """
        return fast_split(
            tokens=tokens.copy(),
            time_from=time_from,
            time_to=time_to,
            n_special=self.config.vocab_size.special,
            n_note=self.config.vocab_size.note,
            n_velocity=self.config.vocab_size.velocity,
            time_resolution=self.config.time_resolution,
            decimal=self.config.float_decimal,
        )

    def detokenize(self, token, time_offset):
        type, value = fast_detokenize(
            token,
            time_offset=time_offset,
            n_special=self.config.vocab_size.special,
            n_note=self.config.vocab_size.note,
            n_velocity=self.config.vocab_size.velocity,
            time_resolution=self.config.time_resolution,
            decimal=self.config.float_decimal,
        )
        if type != TOKEN_TIME:
            value = int(value)
        return [type, value]

    def to_string(self, tokens, time_offset=0.0):
        nums = [self.detokenize(token, time_offset=time_offset) for token in tokens]
        for i in range(len(nums)):
            type = nums[i][0]
            value = nums[i][1]
            if type == TOKEN_TIME:
                nums[i][0] = "time"
            elif type == TOKEN_SPECIAL:
                if value == EOS:
                    nums[i][0] = "EOS"
                elif value == PAD:
                    nums[i][0] = "PAD"
                else:
                    nums[i][0] = "Unknown Special"
            elif type == TOKEN_NOTE:
                nums[i][0] = "note"
            elif type == TOKEN_VELOCITY:
                nums[i][0] = "velocity"
        return nums

    def tokenize(self, idx, type):
        rt = fast_tokenize(
            idx,
            type,
            self.config.vocab_size.special,
            self.config.vocab_size.note,
            self.config.vocab_size.velocity,
        )
        if rt == -1:
            raise ValueError(f"type {type} is not a predefined token type.")
        else:
            return rt

    def mido_to_token(self, midi: mido.midifiles.midifiles.MidiFile):
        events = midi_utils.mido_to_sustained_events(midi)

        notes = midi_utils.events_to_notes(
            events
        )  # (N, 4) : onset, offset, pitch, velocity

        primitive_notes = np.array(notes)
        end_time = primitive_notes[:, 1].max()

        n_special = self.config.vocab_size.special
        n_note = self.config.vocab_size.note
        n_velocity = self.config.vocab_size.velocity
        time_resolution = self.config.time_resolution

        times = [[] for i in range(int(end_time / time_resolution) + 1)]
        for i in range(len(primitive_notes)):
            start, end, pitch, velocity = primitive_notes[i]
            # print(int(start / time_resolution), start, time_resolution)
            velocity = get_reduced_velocity(n_velocity=n_velocity, value=velocity)
            times[int(start / time_resolution)].append([pitch, velocity])
            times[int(end / time_resolution)].append([pitch, 0])

        tokens = []
        current_velocity = 0
        for i, time in enumerate(times):
            if len(time) == 0:
                continue
            tokens.append(fast_tokenize(i, TOKEN_TIME, n_special, n_note, n_velocity))
            for pitch, velocity in time:
                if current_velocity != velocity:
                    current_velocity = velocity
                    tokens.append(
                        fast_tokenize(
                            velocity, TOKEN_VELOCITY, n_special, n_note, n_velocity
                        )
                    )

                tokens.append(
                    fast_tokenize(pitch, TOKEN_NOTE, n_special, n_note, n_velocity)
                )

        tokens.append(fast_tokenize(0, TOKEN_SPECIAL, n_special, n_note, n_velocity))

        return np.array(tokens, dtype=int)

    def token_to_word(self, tokens, is_batch=False, time_per_sequence=8.192):
        n_special = self.config.vocab_size.special
        n_note = self.config.vocab_size.note
        n_velocity = self.config.vocab_size.velocity
        time_resolution = self.config.time_resolution
        decimal = self.config.float_decimal
        words = list()

        time_offset = time_per_sequence if is_batch else 0.0

        if is_batch:
            for i in range(len(tokens)):
                time_offset = i * time_per_sequence
                for token in tokens[i]:
                    ttype, value = fast_detokenize(
                        token,
                        time_offset=time_offset,
                        n_special=n_special,
                        n_note=n_note,
                        n_velocity=n_velocity,
                        time_resolution=time_resolution,
                        decimal=decimal,
                    )
                    # ignore tokens after the melspec segment's time
                    if ttype == TOKEN_TIME:
                        if time_offset <= value <= time_offset + time_per_sequence:
                            words.append((ttype, value))
                        else:
                            break
                    else:
                        if ttype == TOKEN_VELOCITY:
                            value = get_original_velocity(
                                n_velocity=n_velocity, value=value
                            )

                        words.append((ttype, value))

        else:
            for token in tokens:
                ttype, value = fast_detokenize(
                    token,
                    time_offset=time_offset,
                    n_special=n_special,
                    n_note=n_note,
                    n_velocity=n_velocity,
                    time_resolution=time_resolution,
                    decimal=decimal,
                )
                if ttype == TOKEN_VELOCITY:
                    value = get_original_velocity(n_velocity=n_velocity, value=value)
                words.append((ttype, value))

        return words

    def token_to_midi(
        self,
        tokens,
        force_offset=False,
        ignore_eos=False,
        ignore_pad=True,
        is_batch=False,
        time_per_sequence=8.16,
    ):
        """
        Convert list of integer tokens to a pretty midi object

        force_offset : release all notes which don't have its offset messages.
        ignore_eos :
            if False, stop decoding when meet an EOS token.
            if True, ignore EOS token
        ignore_pad :
            same as ignore_eos.

        is_batch :
            if shape of token is 2-dim (batch, sequence),
                When decoding the (batch)th time token,
                the offset time of (batch)th audio segment added.

        time_per_sequence :
            used when batch decoding.
        """

        words = self.token_to_word(
            tokens, is_batch=is_batch, time_per_sequence=time_per_sequence
        )
        new_pm = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=120.0)
        new_inst = pretty_midi.Instrument(program=0)

        current_time = 0
        current_velocity = 0
        note_queue = [None] * (
            self.config.vocab_size.note + 1
        )  # note_queue[pitch] = (on, off, pitch, vel)
        notes = list()
        for type, number in words:
            if type == TOKEN_SPECIAL:
                if number == 0 and ignore_pad:  # pad
                    pass
                elif number == 0 and not ignore_pad:
                    break
                elif number == 1 and ignore_eos:
                    pass
                elif number == 1 and not ignore_eos:  # eos
                    break
            elif type == TOKEN_NOTE:
                pitch = int(number)
                if current_velocity == 0:  # offset
                    if note_queue[pitch] is not None:
                        onset, velocity = note_queue[pitch]
                        if onset >= current_time:
                            pass
                        elif velocity == 0:
                            pass
                        elif pitch == 0:
                            pass
                        else:
                            new_note = pretty_midi.Note(
                                velocity=int(velocity),
                                pitch=int(pitch),
                                start=onset,
                                end=current_time,
                            )
                            notes.append(new_note)
                        note_queue[pitch] = None
                    else:  # offset without onset
                        pass
                else:
                    if note_queue[pitch] is not None:  # note-on already exists.
                        # then release immediately and add another note-on??
                        # or ignore?
                        # onset, velocity = note_queue[pitch]
                        # new_note = pretty_midi.Note(
                        #     velocity=int(velocity),
                        #     pitch=int(pitch),
                        #     start=onset,
                        #     end=current_time,
                        # )
                        # notes.append(new_note)
                        pass

                    else:
                        note_queue[pitch] = (
                            current_time,
                            current_velocity,
                        )

            elif type == TOKEN_VELOCITY:
                current_velocity = number
            elif type == TOKEN_TIME:
                if current_time < number:  # time token only grows up
                    current_time = number
            else:
                raise ValueError
        #     print(note_queue)

        if force_offset:
            for pitch, note in enumerate(note_queue):
                if note is None:
                    pass
                else:
                    onset, velocity = note
                    new_note = pretty_midi.Note(
                        velocity=int(velocity),
                        pitch=pitch,
                        start=onset,
                        end=current_time + self.config.time_resolution,
                    )
                    notes.append(new_note)

        notes = sorted(notes, key=lambda x: x.start)
        new_inst.notes = notes
        new_pm.instruments.append(new_inst)
        new_pm.remove_invalid_notes()
        return new_pm
