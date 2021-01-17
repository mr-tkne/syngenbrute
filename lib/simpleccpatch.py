import json
from collections import defaultdict


class SimpleCCPatch:
    def __init__(self, channel=0, preset_filename=None, enable_only=None):
        self.channel = channel
        self.vars = {}
        self.params = {}
        self.enabled_params = set()
        self.categories = defaultdict(list)
        if preset_filename is not None:
            self.load(preset_filename, enable_only)

    def load(self, preset_filename, enable_only=None):
        with open(preset_filename, 'r') as f:
            data = f.read()
        patch = json.loads(data)
        for var in patch['vars']:
            if enable_only and var not in enable_only:
                continue
            self.vars[var] = patch['vars'][var]
        for param in patch['params']:
            if enable_only and param['name'] not in enable_only:
                continue
            # TODO validation
            if 'filter' not in param:
                param['filter'] = 'value'
            if 'default' not in param:
                param['default'] = param['min']
            if 'store' not in param:
                param['store'] = param['name']
            self.vars[param['store']] = param['default']
            self.params[param['name']] = param
        if enable_only:
            for key in enable_only:
                self.enabled_params.add(key)

    def set_param(self, param_name, value):
        # print(self.params)
        filter = self.params[param_name]['filter']
        computed = eval(filter, self.vars, {'value': value})
        self.vars[self.params[param_name]['store']] = computed

    def get_patch(self):
        status_byte = 0xB0 | self.channel
        send_bytes = []
        for var in self.enabled_params:
            raw_value = self.vars[self.params[var]['store']]
            if len(self.params[var]['cc']) == 1:
                cc = self.params[var]['cc'][0]
                pkt = [status_byte, cc, raw_value]
            elif len(self.params[var]['cc']) == 2:
                cc_msb = self.params[var]['cc'][0]
                cc_lsb = self.params[var]['cc'][1]
                pkt = [status_byte, cc_msb, ((raw_value & 0x3F80) >> 7),
                       status_byte, cc_lsb, (raw_value & 0x7F)]
            send_bytes += pkt
        return send_bytes
