# Original acquisition header plus the same-host CLOCK_MONOTONIC timeline.
# sensor identifies left, right, or odor; header.stamp + sensor is the sample ID.
# source_monotonic_ns=0 means source instrumentation is unavailable.
std_msgs/Header header
string sensor
string clock_domain
uint64 source_monotonic_ns
uint64 ingress_monotonic_ns
uint64 forward_monotonic_ns
