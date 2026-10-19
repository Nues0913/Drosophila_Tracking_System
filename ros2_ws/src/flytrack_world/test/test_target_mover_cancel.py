from types import SimpleNamespace

from flytrack_world.target_mover import TargetMoverNode
from rclpy.task import Future


def node_with_errors():
    node = object.__new__(TargetMoverNode)
    node.pending = None
    errors = []
    node.get_logger = lambda: SimpleNamespace(error=errors.append)
    return node, errors


def test_inactive_cancel_detaches_before_immediate_done_callback():
    node, errors = node_with_errors()
    node.active = False

    def consume_event(_):
        return False, {}

    node.consume_event = consume_event
    future = Future()
    node.pending = future
    future.add_done_callback(node._complete)

    node._scenario(object())

    assert future.cancelled()
    assert node.pending is None
    assert errors == []


def test_late_callback_from_cancelled_request_does_not_clear_new_pending():
    node, errors = node_with_errors()
    old = Future()
    node.pending = old
    node._cancel_pending()
    new = Future()
    node.pending = new

    node._complete(old)

    assert node.pending is new
    assert errors == []


def test_cancelled_current_future_is_not_reported_as_service_failure():
    node, errors = node_with_errors()
    future = Future()
    node.pending = future

    future.cancel()
    node._complete(future)

    assert node.pending is None
    assert errors == []


def test_service_rejection_still_reports_error():
    node, errors = node_with_errors()
    future = Future()
    node.pending = future
    future.set_result(SimpleNamespace(success=False))

    node._complete(future)

    assert errors == ["Gazebo rejected target SetEntityPose"]


def test_service_exception_still_reports_error():
    node, errors = node_with_errors()
    future = Future()
    node.pending = future
    future.set_exception(RuntimeError("service transport failed"))

    node._complete(future)

    assert errors == ["target SetEntityPose failed: service transport failed"]
