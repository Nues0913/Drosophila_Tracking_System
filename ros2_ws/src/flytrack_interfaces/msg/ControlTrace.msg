# Causal samples used for this command; zero times indicate missing metadata.
std_msgs/Header header
uint64 command_sequence
string clock_domain
uint64 control_started_monotonic_ns
uint64 command_published_monotonic_ns
bool inference_performed
bool[3] valid
SensorTiming[3] samples
uint64[3] policy_received_monotonic_ns
