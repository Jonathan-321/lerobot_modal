#!/usr/bin/env python3
"""
This script acts as a local gRPC server that bridges the lerobot RobotClient
to a Modal endpoint for policy inference. It receives observations from the client,
forwards them to a Modal function, and streams the resulting actions back.
"""

import logging
import pickle
import grpc
from concurrent import futures
import time
import modal
import numpy as np

# Assuming lerobot is installed in editable mode, we can import its components.
from lerobot.transport import async_inference_pb2, async_inference_pb2_grpc
from lerobot.scripts.server.helpers import TimedAction, TimedObservation, get_logger

# --- Configuration ---
MODAL_APP_NAME = "lerobot-smolvla-inference"
MODAL_FUNCTION_NAME = "SmolVLAInference.run_inference"
HOST = "127.0.0.1"
PORT = 8080

logger = get_logger("modal_policy_server")

class ModalPolicyServer(async_inference_pb2_grpc.AsyncInferenceServicer):
    """
    A gRPC server that translates requests from a RobotClient to a Modal endpoint.
    """

    def __init__(self):
        self.modal_func = None
        self.policy_config = None
        logger.info("ModalPolicyServer initialized.")

    def _connect_to_modal(self):
        """Establish connection to the Modal function."""
        if self.modal_func:
            return
        try:
            logger.info(f"Connecting to Modal app '{MODAL_APP_NAME}'...")
            self.modal_func = modal.Function.lookup(MODAL_APP_NAME, MODAL_FUNCTION_NAME)
            logger.info("✅ Successfully connected to Modal function.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Modal: {e}")
            self.modal_func = None

    def Ready(self, request, context):
        """A simple RPC to check if the server is ready."""
        logger.info("Client connected and checked Ready status.")
        return async_inference_pb2.Empty()

    def SendPolicyInstructions(self, request, context):
        """Receives policy configuration from the client."""
        self.policy_config = pickle.loads(request.data)
        logger.info("Received policy instructions from client.")
        # We can log the config for debugging, but our policy is on Modal.
        return async_inference_pb2.Empty()

    def StreamObservations(self, request_iterator, context):
        """Receives a stream of observations and returns a stream of actions."""
        self._connect_to_modal()
        if not self.modal_func:
            logger.error("Cannot stream observations, Modal connection not available.")
            return

        for request in request_iterator:
            try:
                timed_obs: TimedObservation = pickle.loads(request.data)
                logger.debug(f"Received observation with timestamp {timed_obs.timestamp}")

                # Prepare data for Modal function
                # The exact format depends on your Modal function's signature.
                # Based on our previous work, it expects image and state.
                image = timed_obs.obs.image['camera_0'] # Assuming single camera
                state = timed_obs.obs.state
                instruction = "Pick up the red block" # Example instruction

                # Call the Modal function
                start_time = time.perf_counter()
                result = self.modal_func.remote(image, state, instruction)
                end_time = time.perf_counter()
                logger.info(f"Modal inference took {end_time - start_time:.4f}s")

                # Create a TimedAction to send back to the client
                # The result from Modal should be a dictionary with an 'action' key.
                action_chunk = result['action']
                timed_action = TimedAction(
                    action=action_chunk,
                    timestamp=time.time(),
                    actions_per_chunk=len(action_chunk)
                )

                action_bytes = pickle.dumps(timed_action)
                yield async_inference_pb2.Action(data=action_bytes)

            except Exception as e:
                logger.error(f"Error processing observation: {e}")
                # Terminate the stream on error
                break

def serve():
    """Starts the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    async_inference_pb2_grpc.add_AsyncInferenceServicer_to_server(ModalPolicyServer(), server)
    server.add_insecure_port(f"{HOST}:{PORT}")
    server.start()
    logger.info(f"🚀 Server listening on {HOST}:{PORT}")
    logger.info("Ready to accept connections from a RobotClient.")
    try:
        while True:
            time.sleep(86400) # One day
    except KeyboardInterrupt:
        logger.info("Shutting down server.")
        server.stop(0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
