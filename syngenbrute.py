#!/usr/bin/env python3

import sys
import time
import mido
import argparse
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from lib.simpleccpatch import SimpleCCPatch
from lib.analyser import MFCCComparator
from lib.genetic import GeneticModel

sd.default.device = 'system'


def trimpad(sample, startThreshold, length):
    start = 0
    end = length
    for (idx, smp) in enumerate(sample):
        if smp[0] > startThreshold:
            start = idx
            break
    return sample[start:end]


def record(length, sample_rate, threshold):
    # recording = sd.rec(3, sample_rate, channels=2, blocking=True)
    recording = sd.rec(length, sample_rate, channels=1)
    sd.wait()
    recording /= np.max(np.abs(recording), axis=0)
    time.sleep(length/sample_rate)
    return trimpad(recording, threshold, length)

best_score = 0.0
best_patch = None
best_count = 0
def try_speciman(synth, midi_device, channel, comp, genetic_model, speciman, sample_length):
    global best_score
    global best_patch
    global best_count
    for (param, value) in zip(genetic_model.params, speciman.values):
        synth.set_param(param['key'], int(round(value)))
        # print(f'{param["key"]} = {int(round(value))}; ', end='')
    patch = synth.get_patch()
    # print('zxz', midi_device)
    outport = midi_device
    # outport = mido.open_output(midi_device)
    # send patch
    # midi.send(patch)
    # record wav
    # wav = record(sample_length, comp.sample_rate, 100)
    for cmd in mido.parse_all(patch):
        outport.send(cmd)
    outport.send(mido.Message('note_on', channel=channel, note=48))
    wav = sd.rec(sample_length, comp.sample_rate, channels=1)
    # time.sleep(.001)
    sd.wait()
    outport.send(mido.Message('note_off', channel=channel, note=48))
    # wav = trimpad(wav, 1, comp.target_length)

    score = comp.compare_to(wav)
    if score > best_score:
        wavfile.write(f'specimen/best{best_count}.wav', 48000, wav)
        best_score = score
        best_patch = patch
        print(f'{best_count}. ', end='')
        best_count += 1
        for (param, value) in zip(genetic_model.params, speciman.values):
            synth.set_param(param['key'], int(round(value)))
            print(f'{param["key"]} = {int(round(value))}; ', end='')
        print(' Score -> ', score)
    return score


def bruteforce_sound(path_to_wav, synth_def, params_def,
                     target_score=0.9, pop_size=30, max_generations=100,
                     mutation_rate=0.05, midi_channel=10, midi_device=None):
    global best_patch
    # print('z', synth_def)
    # synth = SimpleCCPatch(midi_channel, synth_def, ['Feedback', 'Mix'])
    synth = SimpleCCPatch(midi_channel, synth_def, ['Feedback'])
    outport = mido.open_output(midi_device)
    comp = MFCCComparator(path_to_wav)
    length = comp.target_length
    gm = GeneticModel(pop_size, mutation_rate, epoch=30)
    gm.test_function = lambda S: try_speciman(synth, outport, midi_channel, comp, gm, S, length)
    for param in params_def:
        gm.add_param(param['name'], param['min'], param['max'])
    best_result = -1
    for i in range(max_generations):
        print(f"Generation {i+1}/{max_generations}")
        generation = gm.iterate()
    # Send best patch
    for cmd in mido.parse_all(best_patch):
        outport.send(cmd)



def midi_test(device_name, channel):
    # test_sequence = [144, 60, 64, 144, 48, 64]
    # for cmd in mido.parse_all(test_sequence):
    #     outport.send(mido.Message('note_off', channel=channel, note=60))
    #     outport.send(cmd)
    #     time.sleep(0.2)
    outport = mido.open_output(device_name)
    recording = record(9600, 48000, 5)
    outport.send(mido.Message('note_on', channel=channel, note=48))
    time.sleep(.001)
    outport.send(mido.Message('note_off', channel=channel, note=48))
    sd.wait()
    # print(recording)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="", add_help=False)
    ap.add_argument('-S', '--synth',
                    type=str,
                    # required=True,
                    help="synth.json file to use")
    ap.add_argument('-L', '--list-controls',
                    type=str,
                    help="List all available synthesizer controls from synth.json")
    ap.add_argument('-t', '--tweak',
                    type=str,
                    help=''),
    ap.add_argument('-a', '--audio-in',
                    type=str,
                    default="",
                    help="Audio input device (--help for list of devices)")
    ap.add_argument('-m', '--midi-out',
                    type=str,
                    default="",
                    help="MIDI output device (--help for list of devices)")
    ap.add_argument('-c', '--channel',
                    type=int,
                    default=0,
                    help="MIDI channel (1 to 16)")
    ap.add_argument('-i', '--input',
                    type=str,
                    # required=True,
                    help="WAV sample to try and bruteforce")
    ap.add_argument('-h', '--help',
                    default=False,
                    action='store_true',
                    help="Print help message and exit")
    args = ap.parse_args()
    # print(args)
    if args.help:
        ap.print_help()
        print("\nMIDI outputs:")
        for out in mido.get_output_names():
            print(f'"{out}"')
        print("\nAudio inputs:")
        for out in mido.get_output_names():
            print(f'"{out}"')
    if args.midi_out and args.channel:
        midi_outs = mido.get_output_names()
        if args.midi_out not in midi_outs:
            print(f"Can't find \033[1m{args.midi_out} MIDI output. Try these:")
            for out in midi_outs:
                print(f'"{out}"')
            sys.exit(0)
        if args.channel < 1 or args.channel > 16:
            print("MIDI channel can be 1 to 16")
            sys.exit(0)
        args.channel -= 1
        if args.input:
            params_def = [
                # {
                #   "name": "Ratio C",
                #   "min": 0,
                #   "max": 127,
                #   "cc": [91],
                #   "category": "syn1"
                # },
                # {
                #   "name": "Ratio B1",
                #   "min": 0,
                #   "max": 18,
                #   "cc": [16, 48],
                #   "filter": "(ratio_b & 0x3E0) | value",
                #   "store": "ratio_b",
                #   "category": "syn1"
                # },
                # {
                #   "name": "Ratio B2",
                #   "min": 0,
                #   "max": 18,
                #   "cc": [16, 48],
                #   "filter": "(ratio_b & 0x1F) | (value << 5)",
                #   "store": "ratio_b",
                #   "category": "syn1"
                # },
                # {
                #   "name": "Harmonics",
                #   "min": 0,
                #   "max": 16383,
                #   "cc": [17, 49],
                #   "category": "syn1"
                # },
                {
                  "name": "Feedback",
                  "min": 0,
                  "max": 127,
                  "cc": [19],
                  "category": "syn1"
                },
                # {
                #   "name": "Mix",
                #   "min": 0,
                #   "max": 16383,
                #   "cc": [20, 52],
                #   "category": "syn1"
                # }
                ]
            bruteforce_sound(args.input, args.synth, params_def, midi_device=args.midi_out, midi_channel=args.channel, max_generations=3, mutation_rate=0.25)
        else:
            midi_test(args.midi_out, args.channel)
