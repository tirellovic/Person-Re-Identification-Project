def parse_annotations_file(
        path_to_file: Path
    ) -> Tuple[List[str], List[List[float]]]:
    """Parse annotation file.

    Args:
        path_to_file: the path to the file with the annotations.

    Returns:
        The ID for each pedestrian in the image.
        The bounding boxes coordinates for each pedestrian in the image in [x, y, w, h] format.
    """
    pids, boxes = [], [] # output lists definition
    mat = scipy.io.loadmat(path_to_file) # convert .mat file to a Python dictionary

    # Apparently not all Matlab annotation files contain the 'box_new' key, indeed there 
    # are frames with no annotated boxes which associated annotation file has not the aftorementioned key. 
    # This case is handled by the following 'if' in conjunction with the __get_item__ method in FrameDataset.
    if 'box_new' not in mat:
        # Frame has no annotated pedestrians - return empty lists
        return pids, boxes

    annotations = mat['box_new'] # extract the annotations from the dictionary
    for ann in annotations:
        pid, x, y, w, h = ann
        pids.append(pid)
        try:
            box = [float(x), float(y), float(w), float(h)]
        except ValueError:
            print(f"Error converting annotation: {ann}")
            raise
        boxes.append(box)

    return pids, boxes


def draw_boxes(
        image: Image.Image,
        boxes: List[List[float]],
        pids: List[int],
        scores: List[float],
        add_text: bool = True
    ) -> Image.Image:
    """Draws a rectangle around each object together with the name of the category and the prediction score.

    Args:
        image: the input image.
        boxes: the bounding boxes in the format [x, y, w, h] for all the objects in the image.
        pids: the labels for all the objects in the image.
        scores: the predicted scores for all the objects in the image.
        normalized_coordinates: if true the coordinates are multiplied according to the height and width of the image.
        add_text: if true add a box with the name of the category and
                  the score.

    Returns:
        The generated image.
    """
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 24)
    image_with_bb = copy.deepcopy(image)
    painter = ImageDraw.Draw(image_with_bb)

    for i, (box, pid) in enumerate(zip(boxes, pids)):
        # coordinate transformation from [x, y, w, h] to [x_min, y_min, x_max, y_max]
        x_min, _, _, y_max = box
        painter.rectangle(box.tolist(), outline="red", width=4)

        if add_text:
            score = scores[i]
            text_in_box = f"{pid}-{score:.2f}"
            text_width, text_height = font.getbbox(text_in_box)[-2:]
            painter.rectangle(
                [(x_min, y_max - text_height), (x_min + text_width, y_max)],
                fill="red"
            )
            painter.text(
                (x_min, y_max - text_height),
                text_in_box,
                fill="black",
                font=font
            )

    return image_with_bb

def detect_pedestrians(
        image: Image.Image,
        detector: nn.Module,
    ) -> Tuple[List[List[int]], List[int]]:
    """Detects objects in the image using the provided detector. This function puts the model in the eval mode.

    Args:
        image: the input image.
        detector: the detector model.
        categories: the names of the categories of the dataset used to train the network.

    Returns:
        The predicted bounding boxes in xyxy format.
        The predicted scores.
    """
    detector.eval() # put model in evaluation mode
    with torch.no_grad():
        detections = detector(image) 
        detections = detections[0] # remove batch dimension (doing inference on a single frame -> batch size is 1)

    # Get predicted boxes and scores
    boxes = detections["boxes"].detach().cpu().numpy()
    scores = detections["scores"].detach().cpu().numpy()

    return boxes, scores

def crop_detections_from_frame( # former: crop_bounding_box
        image: Image.Image,
        boxes: List[List[float]],
        output_size: Tuple[int, int],
        crop_margin: Optional[int] = 0
    ) -> List[Image.Image]:
    """Crops the pixels in the image corresponding to the bounding boxes of detections, the crop is then resited to fit the output size.

    Args:
        image: the input image.
        boxes: the bounding boxes .
        crop_margin: the margin to add to the bounding box, in terms of pixels in the final image.
        output_size: the output image size in pixels.

    Returns:
        The detection images extracted from the frame.
    """
    detection_images = []
    width, height = image.size

    for box in boxes:
        x_min, y_min, x_max, y_max = box

        x_margin = crop_margin * (x_max - x_min) / (output_size[0] - crop_margin)
        x_margin *= 0.5
        y_margin = crop_margin * (y_max - y_min) / (output_size[1] - crop_margin)
        y_margin *= 0.5

        x_min = int(max(x_min - x_margin, 0))
        y_min = int(max(y_min - y_margin, 0))
        x_max = int(min(x_max + x_margin, width))
        y_max = int(min(y_max + y_margin, height))
        
        detection_image = np.asarray(image)[y_min:y_max, x_min:x_max]
        detection_image_resized = Image.fromarray(detection_image).resize((output_size[0], output_size[1]), Image.BILINEAR)
        detection_images.append(detection_image_resized)

    return detection_images


def set_requires_grad_for_layer(layer: torch.nn.Module, train: bool) -> None:
    """Sets the attribute requires_grad to True or False for each parameter within the layer.

        Args:
            layer: the layer to freeze.
            train: if True, trains the layer.
    """
    for p in layer.parameters():
        p.requires_grad = train