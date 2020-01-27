def copy_object_to_location(frame, object_to_copy, x, y, mask):
    frame_height, frame_width, _ = frame.shape
    object_height, object_width, _ = object_to_copy.shape

    if x < 0 or x + object_width > frame_width or y < 0 or y + object_height > frame_height:
        return

    original = frame[y: y + object_height, x: x + object_width, 0:3] * mask
    new = object_to_copy[:, :, 0:3] * (1-mask)
    frame[y: y + object_height, x: x + object_width, 0:3] = original + new


