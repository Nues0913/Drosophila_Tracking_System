# Paired arbitrary concentration units; stamp is the original sensor-production time.
std_msgs/Header header
float32 left
float32 right
bool valid
# Source message-production time, before transport; same-host clock domain.
uint64 source_monotonic_ns
string clock_domain
