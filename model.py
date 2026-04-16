import torch
import numpy as np
import os
import pathlib
import cv2
import tempfile
import logging

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_sdk.label_interface.objects import PredictionValue
from sam2.build_sam import build_sam2_video_predictor

logger = logging.getLogger(__name__)


DEVICE = os.getenv('DEVICE', 'cuda')
MODEL_CONFIG = os.getenv('MODEL_CONFIG', 'configs/sam2.1/sam2.1_hiera_t.yaml')
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', 'sam2.1_hiera_tiny.pt')
MAX_FRAMES_TO_TRACK = int(os.getenv('MAX_FRAMES_TO_TRACK', 10))

if DEVICE == 'cuda':
    # use bfloat16 for the entire model
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# Build path to checkpoint: always ./checkpoints/ relative to this file
sam2_checkpoint = str(pathlib.Path(__file__).parent / 'checkpoints' / MODEL_CHECKPOINT)
predictor = build_sam2_video_predictor(MODEL_CONFIG, sam2_checkpoint)


# Inference state cache — avoids re-initialising SAM2 for the same video frame directory
_predictor_state_key = ''
_inference_state = None


def get_inference_state(video_dir):
    global _predictor_state_key, _inference_state
    if _predictor_state_key != video_dir:
        _predictor_state_key = video_dir
        _inference_state = predictor.init_state(video_path=video_dir)
    return _inference_state


class SAM2VideoModel(LabelStudioMLBase):
    """Label Studio ML backend for video object tracking using SAM2."""

    def split_frames(self, video_path, temp_dir, start_frame=0, end_frame=100):
        """Extract frames from a video file into temp_dir as %05d.jpg files."""
        logger.debug(f'Opening video file: {video_path}')
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        logger.debug(f'Total frames in video: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}')

        frame_count = 0
        while True:
            success, frame = video.read()
            if frame_count < start_frame:
                frame_count += 1
                continue
            if frame_count - start_frame >= end_frame - start_frame:
                break
            if not success:
                logger.error(f'Failed to read frame {frame_count}')
                break

            # SAM2 expects zero-padded filenames: 00000.jpg, 00001.jpg, …
            relative_idx = frame_count - start_frame
            frame_filename = os.path.join(temp_dir, f'{relative_idx:05d}.jpg')
            if not os.path.exists(frame_filename):
                cv2.imwrite(frame_filename, frame)
            logger.debug(f'Frame {frame_count} → {frame_filename}')
            yield frame_filename, frame

            frame_count += 1

        video.release()

    def get_prompts(self, context) -> List[Dict]:
        """
        Parse VideoRectangle bounding boxes from Label Studio context into SAM2 keypoints.

        Each bounding box is converted to 5 keypoints (center + 4 mid-edge points)
        so SAM2 can identify the object region within the frame.
        """
        logger.debug(f'Extracting prompts from context: {context}')
        prompts = []
        for ctx in context['result']:
            obj_id = ctx['id']
            for obj in ctx['value']['sequence']:
                x = obj['x'] / 100           # normalised [0, 1]
                y = obj['y'] / 100
                box_width = obj['width'] / 100
                box_height = obj['height'] / 100
                frame_idx = obj['frame'] - 1  # Label Studio frames are 1-indexed

                # Convert bbox to 5 keypoints (all foreground)
                kps = [
                    [x + box_width / 2,         y + box_height / 2],       # centre
                    [x + box_width / 4,         y + box_height / 2],       # left-mid
                    [x + 3 * box_width / 4,     y + box_height / 2],       # right-mid
                    [x + box_width / 2,         y + box_height / 4],       # top-mid
                    [x + box_width / 2,         y + 3 * box_height / 4],   # bottom-mid
                ]

                points = np.array(kps, dtype=np.float32)
                labels = np.array([1] * len(kps), dtype=np.int32)
                prompts.append({
                    'points': points,
                    'labels': labels,
                    'frame_idx': frame_idx,
                    'obj_id': obj_id,
                })

        return prompts

    def _get_fps(self, context):
        frames_count = context['result'][0]['value']['framesCount']
        duration = context['result'][0]['value']['duration']
        return frames_count, duration

    def convert_mask_to_bbox(self, mask):
        """Convert a binary mask to a percentage-based bounding box dict."""
        mask = mask.squeeze()
        y_indices, x_indices = np.where(mask == 1)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        height, width = mask.shape

        return {
            'x':      round((xmin / width) * 100, 2),
            'y':      round((ymin / height) * 100, 2),
            'width':  round(((xmax - xmin + 1) / width) * 100, 2),
            'height': round(((ymax - ymin + 1) / height) * 100, 2),
        }

    def dump_image_with_mask(self, frame, mask, output_file, obj_id=None, random_color=False):
        """Debug helper: save a frame with the SAM2 mask overlaid."""
        from matplotlib import pyplot as plt
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = (mask_image * 255).astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGRA2BGR)
        mask_image = cv2.addWeighted(frame, 1.0, mask_image, 0.8, 0)
        cv2.imwrite(output_file, mask_image)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """
        Receive a VideoRectangle annotation on a frame and propagate tracking
        to the next MAX_FRAMES_TO_TRACK frames using SAM2 VideoPredictor.
        """
        from_name, to_name, value = self.get_first_tag_occurence('VideoRectangle', 'Video')

        task = tasks[0]
        task_id = task['id']
        video_url = task['data'][value]

        # Download / cache the video locally
        video_path = self.get_local_path(video_url, task_id=task_id)
        logger.debug(f'Video path: {video_path}')

        prompts = self.get_prompts(context)
        all_obj_ids = set(p['obj_id'] for p in prompts)
        # Map string obj_ids to consecutive integers for SAM2
        obj_id_map = {obj_id: i for i, obj_id in enumerate(all_obj_ids)}

        first_frame_idx = min(p['frame_idx'] for p in prompts) if prompts else 0
        last_frame_idx  = max(p['frame_idx'] for p in prompts) if prompts else 0
        frames_count, duration = self._get_fps(context)
        fps = frames_count / duration

        logger.debug(
            f'prompts={len(prompts)}, first_frame={first_frame_idx}, '
            f'last_frame={last_frame_idx}, obj_ids={obj_id_map}'
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            frames = list(self.split_frames(
                video_path, temp_dir,
                start_frame=first_frame_idx,
                end_frame=last_frame_idx + MAX_FRAMES_TO_TRACK + 1,
            ))
            height, width, _ = frames[0][1].shape
            logger.debug(f'Video size: {width}x{height}')

            inference_state = get_inference_state(temp_dir)
            predictor.reset_state(inference_state)

            for prompt in prompts:
                # Scale normalised coords to pixel coords
                prompt['points'][:, 0] *= width
                prompt['points'][:, 1] *= height

                predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=prompt['frame_idx'] - first_frame_idx,
                    obj_id=obj_id_map[prompt['obj_id']],
                    points=prompt['points'],
                    labels=prompt['labels'],
                )

            sequence = []
            logger.info(
                f'Propagating from frame {last_frame_idx} '
                f'for {MAX_FRAMES_TO_TRACK} frames'
            )

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=last_frame_idx - first_frame_idx,
                max_frame_num_to_track=MAX_FRAMES_TO_TRACK,
            ):
                real_frame_idx = out_frame_idx + first_frame_idx
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    bbox = self.convert_mask_to_bbox(mask)
                    if bbox:
                        sequence.append({
                            'frame':    real_frame_idx + 1,  # Label Studio is 1-indexed
                            'x':        bbox['x'],
                            'y':        bbox['y'],
                            'width':    bbox['width'],
                            'height':   bbox['height'],
                            'enabled':  True,
                            'rotation': 0,
                            'time':     out_frame_idx / fps,
                        })

            context_result_sequence = context['result'][0]['value']['sequence']

            prediction = PredictionValue(
                result=[{
                    'value': {
                        'framesCount': frames_count,
                        'duration':    duration,
                        'sequence':    context_result_sequence + sequence,
                    },
                    'from_name': from_name,
                    'to_name':   to_name,
                    'type':      'videorectangle',
                    'origin':    'manual',
                    'id':        list(all_obj_ids)[0],
                }]
            )
            logger.debug(f'Prediction: {prediction.model_dump()}')
            return [prediction.model_dump()]
