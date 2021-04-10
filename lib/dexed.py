#!/usr/bin/env python3
import json
from collections import defaultdict
# A: SYSEX Message: Bulk Data for 1 Voice

class Dexed:
    def __init__(self, channel=0, preset_filename=None, enable_only=None):
        self.params = []
        self.vars = {}
        self.param_map = {}
        self.init_params()
        self.enabled_params = set()
        self.categories = defaultdict(list)
        if preset_filename is not None:
            self.load(preset_filename, enable_only)

    def init_params(self):
        p = self.params
        for op in range(6):
            opid = 6 - op
            for i in range(4):
                p.append((f"OP{opid} EG RATE {i+1}", 0, 99))
            for i in range(4):
                p.append((f"OP{opid} EG LEVEL {i+1}", 0, 99))
            p.append((f"OP{opid} KBD LEV SCL BRK PT", 0, 99, 0x27))
            p.append((f"OP{opid} KBD LEV SCL LFT DEPTH", 0, 99, 0x0))
            p.append((f"OP{opid} KBD LEV SCL RHT DEPTH", 0, 99, 0x0))
            p.append((f"OP{opid} KBD LEV SCL LFT CURVE", 0, 3, 0x0))
            p.append((f"OP{opid} KBD LEV SCL RHT CURVE", 0, 3, 0x3))
            p.append((f"OP{opid} KBD RATE SCALING", 0, 7, 0x0))
            p.append((f"OP{opid} AMP MOD SENSITIVITY", 0, 3, 0x0))
            p.append((f"OP{opid} KEY VEL SENSITIVITY", 0, 7, 0x0))
            p.append((f"OP{opid} OPERATOR OUTPUT LEVEL", 0, 99, 0x0))
            p.append((f"OP{opid} OSC MODE", 0, 1, 0x0))
            p.append((f"OP{opid} OSC FREQ COARSE", 0, 31, 0x1))
            p.append((f"OP{opid} OSC FREQ FINE", 0, 99, 0x0))
            p.append((f"OP{opid} OSC DETUNE", 0, 14, 0x7))
        for i in range(4):
            p.append((f"PITCH EG RATE {i}", 0, 99, 99))
        for i in range(4):
            p.append((f"PITCH EG LEVEL {i}", 0, 99, 99))
        p.append((f"ALGORITHM", 0, 31, 0))
        p.append((f"FEEDBACK", 0, 7, 0))
        p.append((f"OSCILLATOR SYNC", 0, 1, 0))
        p.append((f"LFO SPEED", 0, 99, 0))
        p.append((f"LFO DELAY", 0, 99, 0))
        p.append((f"LFO PITCH MOD DEPTH", 0, 99, 0))
        p.append((f"LFO AMP MOD DEPTH", 0, 99, 0))
        p.append((f"LFO SYNC", 0, 1, 0))
        p.append((f"LFO WAVEFORM", 0, 5, 0))
        p.append((f"PITCH MOD SENSITIVITY", 0, 7, 0))
        p.append((f"TRANSPOSE", 0, 48, 0))
        for i in range(10):
            p.append((f"VOICE NAME CHAR {i}", 0, 127, 0x30))
        p.append((f"OPERATOR ON/OFF", 0, 63, 63))
        for idx in range(len(p)):
            self.param_map[p[0]] = p

    def load(self, _):
        pass

    def set_param(self, param_name, value):
