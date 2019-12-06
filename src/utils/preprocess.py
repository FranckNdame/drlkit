from skimage import transform # Preprocess the frames || scikit-image
from skimage.color import rgb2gray # Grayscale the frames
from collections import deque


def preprocess(frame, stacked_frames, crop=(5,-10,5,-10), new_size=[100,80], stack_size=4, is_new_episode=True, state_w=110, state_h=84):
    preprocessed_frame = preprocess_frame(frame, crop, new_size)
    return stack_frames(stacked_frames, preprocessed_frame, is_new_episode, stack_size, state_w, state_h)


def preprocess_frame(frame, crop, new_size):
    # Grayscale frame
    gs_frame = rgb2gray(frame)
    # Crop unnecessary pixels
    cropped_frame = gs_frame[crop[0]:crop[1], crop[2]:crop[3]]
    # Normalize pixel values 0 - 255 to 0 - 1
    normalized_frame = cropped_frame/255.0
    # Resize
    resized_frame = transform.resize(normalized_frame, new_size)
    return resized_frame

def initialize_stack_frame(width=110, height=84, max_len=4):
    return  deque([np.zeros((width, height), dtype=np.int) for _ in range(stack_size)], maxlen=max_len)

def stack_frames(stacked_frames, state, is_new_episode, stack_size=4, state_w=110, state_h=84):
    # Preprocess frame
    # frame = preprocess_frame(state)

    if is_new_episode:
        # Clear stacked frames
        stacked_frames = initialize_stack_frame(state_w, state_h)
        # since this is a new episode, we only copy the same frame 4x
        for _ in range(stack_size):
            stacked_frames.append(frame)
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2) # ??
    else:
        # Push new frame to queue and pop oldest frame
        stacked_frames.append(frame)
        # Build the stacked state (first dimensuin specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames
