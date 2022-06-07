import math
import random
from typing import Tuple

from simulator.network_simulator.constants import BITS_PER_BYTE, BYTES_PER_PACKET
from simulator.trace import Trace


class Link():

    def __init__(self, trace: Trace):
        self.trace = trace
        self.trace.reset()
        self.queue_delay_update_time = 0.0
        self.queue_size = self.trace.get_queue_size()
        self.pkt_in_queue = 0

    def get_cur_queue_delay(self, event_time: float) -> float:
        self.pkt_in_queue = max(
            0, self.pkt_in_queue - self.trace.get_avail_bits2send(
                self.queue_delay_update_time, event_time) / BITS_PER_BYTE / BYTES_PER_PACKET)
        self.queue_delay_update_time = event_time

        cur_queue_delay = self.trace.get_sending_t_usage(
            self.pkt_in_queue * BYTES_PER_PACKET * BITS_PER_BYTE, event_time)
        return cur_queue_delay

    def get_cur_propagation_latency(self, event_time: float) -> float:
        return self.trace.get_delay(event_time) / 1000.0

    def get_cur_latency(self, event_time: float) -> Tuple[float, float]:
        q_delay = self.get_cur_queue_delay(event_time)
        return self.trace.get_delay(event_time) / 1000.0, q_delay

    def packet_enters_link(self, event_time: float) -> bool:
        if (random.random() < self.trace.get_loss_rate()):
            return False
        if 1 + self.pkt_in_queue > self.queue_size:
            return False
        self.pkt_in_queue += 1
        return True

    def reset(self):
        self.trace.reset()
        self.queue_delay_update_time = 0.0
        self.pkt_in_queue = 0

    def get_bandwidth(self, ts):
        return self.trace.get_bandwidth(ts) * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET
