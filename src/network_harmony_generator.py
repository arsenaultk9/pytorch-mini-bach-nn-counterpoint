import torch

import src.constants as constants

from src.networks.forward_network import ForwardNetwork
from src.models.voices import voices
from src.models.voice import Voice

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class NetworkHarmonyGenerator:
    def __init__(self, network: ForwardNetwork):
        self.network = network

    def get_note_number(self, output: torch.Tensor, slice_index: int, voice: Voice):
        output_slice = output[slice_index]
        on_position = output_slice.argmax()

        return voice.get_note_from_tensor_position(on_position.item())

    def get_voice_note(self, output: torch.Tensor, slice_index: int, voice: Voice):
        on_position = output[slice_index]

        return voice.get_note_from_tensor_position(on_position.item())


    def generate_harmony(self, x_soprano: torch.Tensor):
        harmony_notes = []

        x_soprano = x_soprano.to(device)
        y_alto, y_tenor, y_bass = self.network(x_soprano)

        # Only use first batch result and flip dimensions so notes are last dimension and time first dimension for model output
        x_soprano = x_soprano[0]
        y_alto = y_alto[0].transpose(0, 1)
        y_tenor = y_tenor[0].transpose(0, 1)
        y_bass = y_bass[0].transpose(0, 1)

        for slice_index in range(constants.SEQUENCE_LENGTH):
            soprano_note = self.get_note_number(
                x_soprano, slice_index, voices['soprano'])
                
            alto_note = self.get_note_number(
                y_alto, slice_index, voices['alto'])
            tenor_note = self.get_note_number(
                y_tenor, slice_index, voices['tenor'])
            bass_note = self.get_note_number(
                y_bass, slice_index, voices['bass'])

            notes_tuple = (soprano_note, alto_note, tenor_note, bass_note)
            harmony_notes.append(notes_tuple)

        return harmony_notes

    def imitate_harmony(self,
                        x_soprano: torch.Tensor,
                        y_alto: torch.Tensor,
                        y_tenor: torch.Tensor,
                        y_bass: torch.Tensor):
        harmony_notes = []

        x_soprano = x_soprano.to(device)[None, :]

        for slice_index in range(constants.SEQUENCE_LENGTH):
            soprano_note = self.get_note_number(
                x_soprano[0], slice_index, voices['soprano'])

            alto_note = self.get_voice_note(
                y_alto, slice_index, voices['alto'])
            tenor_note = self.get_voice_note(
                y_tenor, slice_index, voices['tenor'])
            bass_note = self.get_voice_note(
                y_bass, slice_index, voices['bass'])

            notes_tuple = (soprano_note, alto_note, tenor_note, bass_note)
            harmony_notes.append(notes_tuple)

        return harmony_notes
