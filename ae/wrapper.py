import os
from typing import Any, Dict, Optional, Tuple

# import cv2
import gym
import numpy as np

from ae.autoencoder import load_ae


class AutoencoderWrapper(gym.Wrapper):
    """
    Gym wrapper to encode image and reduce input dimension
    using pre-trained auto-encoder
    (only the encoder part is used here, decoder part can be used for debug)

    :param env: Gym environment
    :param ae_path: Path to the autoencoder
    """

    def __init__(self, env: gym.Env, ae_path: Optional[str] = os.environ.get("AAE_PATH")):  # noqa: B008
        super().__init__(env)
        assert ae_path is not None, "No path to autoencoder was provided"
        self.ae = load_ae(ae_path)
        # Update observation space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.ae.z_size,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        # Important: Convert to BGR to match OpenCV convention
        return self.ae.encode_from_raw_image(self.env.reset()[:, :, ::-1]).flatten()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, infos = self.env.step(action)
        # encoded_img = self.ae.encode_from_raw_image(obs[:, :, ::-1])
        # reconstructed_img = self.ae.decode(encoded_img)[0]
        # cv2.imshow("Original", obs[:, :, ::-1])
        # cv2.imshow("Reconstruction", reconstructed_img)
        # cv2.waitKey(0)
        return self.ae.encode_from_raw_image(obs[:, :, ::-1]).flatten(), reward, done, infos
