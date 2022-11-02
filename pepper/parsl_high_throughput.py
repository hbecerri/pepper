import parsl
import zmq
import threading
import queue
from multiprocessing import Queue
from parsl.multiprocessing import ForkProcess
from parsl.executors.errors import ScalingFailed
from parsl.process_loggers import wrap_with_logs


# This serves as a workaround for the connection limit issue
# https://github.com/Parsl/parsl/issues/2165
# Using the workaround just means using the HighThroughputExecutor implemented
# here instead of Parsl's.
# The connection limit issue is circumvented by spawning proxy processes
# that get all the connections from the workers and send their data to the
# interchange process through just a very limited amount of connections.


MAX_WORKERS_PER_PROC = 400
# 400 workers means 800 connections which is well below the default
# limit of 1024
executor_logger = parsl.executors.high_throughput.executor.logger


class InterchangeProxy:
    def __init__(self, interchange_address, interchange_ports,
                 worker_port_range=(54000, 60000), poll_period=10,
                 identity=None):
        self.context = zmq.Context()
        self.task_incoming = self.context.socket(zmq.DEALER)
        self.task_incoming.set_hwm(0)
        self.task_incoming.RCVTIMEO = 10  # in milliseconds
        if identity is not None:
            self.task_incoming.setsockopt(zmq.IDENTITY, identity)
        self.task_incoming.connect(
            f"tcp://{interchange_address}:{interchange_ports[0]}")
        self.results_outgoing = self.context.socket(zmq.DEALER)
        self.results_outgoing.set_hwm(0)
        if identity is not None:
            self.results_outgoing.setsockopt(zmq.IDENTITY, identity)
        self.results_outgoing.connect(
            f"tcp://{interchange_address}:{interchange_ports[1]}")

        self.task_outgoing = self.context.socket(zmq.ROUTER)
        self.task_outgoing.set_hwm(0)
        self.results_incoming = self.context.socket(zmq.ROUTER)
        self.results_incoming.set_hwm(0)

        self.worker_task_port = self.task_outgoing.bind_to_random_port(
            "tcp://*", min_port=worker_port_range[0],
            max_port=worker_port_range[1], max_tries=1000)
        self.worker_result_port = self.results_incoming.bind_to_random_port(
            "tcp://*", min_port=worker_port_range[0],
            max_port=worker_port_range[1], max_tries=1000)

        self.kill_event = None

    @classmethod
    def starter(cls, comm_q, *args, **kwargs):
        icp = cls(*args, **kwargs)
        comm_q.put((icp.worker_task_port,
                    icp.worker_result_port))
        comm_q.close()
        icp.start()

    def forward_tasks(self):
        zmq.proxy(self.task_outgoing, self.task_incoming)

    def forward_results(self):
        zmq.proxy(self.results_incoming, self.results_outgoing)

    def start(self):
        self.kill_event = threading.Event()

        self._task_thread = threading.Thread(
            target=self.forward_tasks, args=tuple(), name="Task-Forwarder")
        self._result_thread = threading.Thread(
            target=self.forward_results, args=tuple(), name="Result-Forwarder")
        self._task_thread.start()
        self._result_thread.start()
        self.kill_event.wait()
        self.context.term()


class ProxiedConnection:
    def __init__(self, socket):
        self.socket = socket
        self.proxy_map = {}
        # Polling a non-socket class, even if it has fileno() like ourself,
        # seems to not work properly, making it impossible to use this
        # as a drop in replacement of the socket. Workaround: Overwrite methods
        self.socket_recv_multipart = self.socket.recv_multipart
        self.socket_send_multipart = self.socket.send_multipart
        self.socket.recv_multipart = self.recv_multipart
        self.socket.send_multipart = self.send_multipart

    def recv_multipart(self, *args, **kwargs):
        data = self.socket_recv_multipart(*args, **kwargs)
        proxy_idn = data[0]
        idn = data[1]
        self.proxy_map[idn] = proxy_idn
        return data[1:]

    def send_multipart(self, msg_parts, *args, **kwargs):
        idn = msg_parts[0]
        msg_parts = [self.proxy_map[idn]] + msg_parts
        return self.socket_send_multipart(msg_parts=msg_parts, *args, **kwargs)

    def fileno(self):
        return self.socket.fileno()


@wrap_with_logs(target="interchange")
def interchange_starter(comm_q, *args, **kwargs):
    """Modified version of parsl.executors.high_throughput.interchange.starter
    This overwrites the relevant methods of the sockets, so that the Parsl
    code works without further modifications, even though the workers are not
    directly connected to the Interchange instance.
    """
    ic = parsl.executors.high_throughput.interchange.Interchange(
        *args, **kwargs)
    ic.task_outgoing_proxy = ProxiedConnection(ic.task_outgoing)
    ic.results_incoming_proxy = ProxiedConnection(ic.results_incoming)
    comm_q.put((ic.worker_task_port, ic.worker_result_port))
    ic.start()


parsl.executors.high_throughput.interchange.starter = interchange_starter


class HighThroughputExecutor(parsl.executors.HighThroughputExecutor):
    def __init__(self, *args, allow_scalein=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._proxy_procs = []
        self._worker_counts = {}
        self.interchange_task_port = None
        self.interchange_result_port = None
        self.allow_scalein = allow_scalein

    def _start_local_interchange_process(self):
        super()._start_local_interchange_process()
        self.interchange_task_port = self.worker_task_port
        self.interchange_result_port = self.worker_result_port
        self.worker_task_port = "{worker_task_port}"
        self.worker_result_port = "{worker_result_port}"

    def _spawn_new_proxy(self):
        comm_q = Queue()
        num = len(self._proxy_procs)
        proc = ForkProcess(
            target=InterchangeProxy.starter,
            kwargs={
                "comm_q": comm_q,
                "interchange_address": "127.0.0.1",
                "interchange_ports": (
                    self.interchange_task_port,
                    self.interchange_result_port),
                "worker_port_range": self.worker_port_range,
                "identity": f"proxy-{num}".encode(),
            },
            daemon=True,
            name="HTEX-InterchangeProxy"
        )
        proc.start()
        self._proxy_procs.append(proc)
        try:
            ports = comm_q.get(block=True, timeout=120)
        except queue.Empty:
            executor_logger.error("Interchange Proxy has not completed "
                                  "initialization in 120s. Aborting")
            raise Exception("Interchange Proxy failed to start")

        return ports

    def _get_worker_ports(self):
        for ports, worker_counts in self._worker_counts.items():
            if worker_counts < MAX_WORKERS_PER_PROC:
                break
        else:
            ports = self._spawn_new_proxy()
        if ports in self._worker_counts:
            self._worker_counts[ports] += 1
        else:
            self._worker_counts[ports] = 1
        return ports

    def _launch_block(self, block_id):
        if self.launch_cmd is None:
            raise ScalingFailed(self.provider.label, "No launch command")
        ports = self._get_worker_ports()
        launch_cmd = self.launch_cmd.format(
            block_id=block_id,
            worker_task_port=ports[0],
            worker_result_port=ports[1])
        job_id = self.provider.submit(launch_cmd, 1)
        executor_logger.debug("Launched block {}->{}".format(block_id, job_id))
        if not job_id:
            raise ScalingFailed(
                self.provider.label,
                "Attempts to provision nodes via provider has failed")
        return job_id

    def _get_launch_command(self, block_id: str) -> str:
        if self.launch_cmd is None:
            raise ScalingFailed(self.provider.label, "No launch command")
        ports = self._get_worker_ports()
        launch_cmd = self.launch_cmd.format(
            block_id=block_id,
            worker_task_port=ports[0],
            worker_result_port=ports[1])
        return launch_cmd

    def shutdown(self, *args, **kwargs):
        ret = super().shutdown(*args, **kwargs)
        for proc in self._proxy_procs:
            proc.terminate()
        return ret

    def scale_in(self, blocks=None, block_ids=[], force=True,
                 max_idletime=None):
        if not force and not self.allow_scalein:
            return []
        if blocks == 1 and not force:
            # Workaround for parsl bug #2195
            # Assume we want to scale in as much as possible
            blocks = len(self.connected_managers)
        return super().scale_in(blocks, block_ids, force, max_idletime)
