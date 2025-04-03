import cv2
import numpy as np
import sys
import os
from ultralytics import YOLO
from sklearn.cluster import KMeans

def get_grass_color(img):
    """
    Finds the color of the grass in the background of the image

    Args:
        img: np.array object of shape (WxHx3) that represents the BGR value of the
        frame pixels.

    Returns:
        grass_color
            Tuple of the BGR value of the grass color in the image
    """
    print("DEBUG: Getting grass color...")
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculate the mean value of the pixels that are not masked
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    grass_color = cv2.mean(img, mask=mask)
    print(f"DEBUG: Grass color found: {grass_color[:3]}")
    return grass_color[:3]

def get_players_boxes(result):
    """
    Finds the images of the players in the frame and their bounding boxes.

    Args:
        result: ultralytics.engine.results.Results object that contains all the
        result of running the object detection algroithm on the frame

    Returns:
        players_imgs
            List of np.array objects that contain the BGR values of the cropped
            parts of the image that contains players.
        players_boxes
            List of ultralytics.engine.results.Boxes objects that contain various
            information about the bounding boxes of the players found in the image.
    """
    players_imgs = []
    players_boxes = []
    for box in result.boxes:
        label = int(box.cls.numpy()[0])
        if label == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            player_img = result.orig_img[y1: y2, x1: x2]
            players_imgs.append(player_img)
            players_boxes.append(box)
    print(f"DEBUG: Found {len(players_imgs)} players in the frame")
    return players_imgs, players_boxes

def get_kits_colors(players, grass_hsv=None, frame=None):
    """
    Finds the kit colors of all the players in the current frame

    Args:
        players: List of np.array objects that contain the BGR values of the image
        portions that contain players.
        grass_hsv: tuple that contain the HSV color value of the grass color of
        the image background.

    Returns:
        kits_colors
            List of np arrays that contain the BGR values of the kits color of all
            the players in the current frame
    """
    kits_colors = []
    if grass_hsv is None:
        print("DEBUG: Grass HSV not provided, calculating it...")
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

    for i, player_img in enumerate(players):
        try:
            # Convert image to HSV color space
            hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

            # Define range of green color in HSV
            lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
            upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])

            # Threshold the HSV image to get only green colors
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Bitwise-AND mask and original image
            mask = cv2.bitwise_not(mask)
            upper_mask = np.zeros(player_img.shape[:2], np.uint8)
            upper_mask[0:player_img.shape[0]//2, 0:player_img.shape[1]] = 255
            mask = cv2.bitwise_and(mask, upper_mask)

            kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])

            kits_colors.append(kit_color)
        except Exception as e:
            print(f"DEBUG: Error getting kit color for player {i}: {e}")
    
    print(f"DEBUG: Found {len(kits_colors)} kit colors")
    return kits_colors

def get_kits_classifier(kits_colors):
    """
    Creates a K-Means classifier that can classify the kits accroding to their BGR
    values into 2 different clusters each of them represents one of the teams

    Args:
        kits_colors: List of np.array objects that contain the BGR values of
        the colors of the kits of the players found in the current frame.

    Returns:
        kits_kmeans
            sklearn.cluster.KMeans object that can classify the players kits into
            2 teams according to their color.
    """
    print("DEBUG: Creating kits classifier...")
    kits_kmeans = KMeans(n_clusters=2)
    kits_kmeans.fit(kits_colors)
    print("DEBUG: Kits classifier created successfully")
    return kits_kmeans

def classify_kits(kits_classifer, kits_colors):
    """
    Classifies the player into one of the two teams according to the player's kit
    color

    Args:
        kits_classifer: sklearn.cluster.KMeans object that can classify the
        players kits into 2 teams according to their color.
        kits_colors: List of np.array objects that contain the BGR values of
        the colors of the kits of the players found in the current frame.

    Returns:
        team
            np.array object containing a single integer that carries the player's
            team number (0 or 1)
    """
    team = kits_classifer.predict(kits_colors)
    return team

def get_left_team_label(players_boxes, kits_colors, kits_clf):
    """
    Finds the label of the team that is on the left of the screen

    Args:
        players_boxes: List of ultralytics.engine.results.Boxes objects that
        contain various information about the bounding boxes of the players found
        in the image.
        kits_colors: List of np.array objects that contain the BGR values of
        the colors of the kits of the players found in the current frame.
        kits_clf: sklearn.cluster.KMeans object that can classify the players kits
        into 2 teams according to their color.
    Returns:
        left_team_label
            Int that holds the number of the team that's on the left of the image
            either (0 or 1)
    """
    print("DEBUG: Determining left team label...")
    left_team_label = 0
    team_0 = []
    team_1 = []

    for i in range(len(players_boxes)):
        x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].numpy())

        team = classify_kits(kits_clf, [kits_colors[i]]).item()
        if team==0:
            team_0.append(np.array([x1]))
        else:
            team_1.append(np.array([x1]))

    if not team_0 or not team_1:
        print("DEBUG: Warning - One team not detected in frame!")
        return left_team_label

    team_0 = np.array(team_0)
    team_1 = np.array(team_1)

    if np.average(team_0) - np.average(team_1) > 0:
        left_team_label = 1
    
    print(f"DEBUG: Left team label is {left_team_label}")
    return left_team_label

def annotate_video(video_path, model):
    """
    Loads the input video and runs the object detection algorithm on its frames, finally it annotates the frame with
    the appropriate labels

    Args:
        video_path: String the holds the path of the input video
        model: Object that represents the trained object detection model
    Returns:
    """
    print(f"DEBUG: Starting video annotation for {video_path}")
    
    # Ensure output directory exists
    os.makedirs('./output', exist_ok=True)
    print("DEBUG: Ensured output directory exists")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at {video_path}")
        return
        
    # Test writing permission
    try:
        test_file = os.path.join('./output', 'test_write.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("DEBUG: Write permission confirmed for output directory")
    except Exception as e:
        print(f"ERROR: Cannot write to output directory: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Failed to open video {video_path}")
        return
    
    print(f"DEBUG: Video opened successfully")
    
    # Get video properties
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"DEBUG: Video properties - Height: {height}, Width: {width}, FPS: {fps}, Total Frames: {total_frames}")

    video_name = os.path.basename(video_path)
    output_path = os.path.join('./output', video_name.split('.')[0] + "_out.mp4")
    print(f"DEBUG: Output video will be saved to {output_path}")
    
    # Try different codecs if available
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not output_video.isOpened():
            raise Exception("Failed to open output video with mp4v codec")
    except Exception as e:
        print(f"DEBUG: Failed with mp4v codec: {e}")
        try:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = os.path.join('./output', video_name.split('.')[0] + "_out.avi")
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"DEBUG: Switched to XVID codec. Output path: {output_path}")
        except Exception as e:
            print(f"ERROR: Failed to create video writer with XVID codec: {e}")
            return

    kits_clf = None
    left_team_label = 0
    grass_hsv = None
    frame_count = 0

    # Save a sample frame to confirm format
    success, sample_frame = cap.read()
    if success:
        sample_path = os.path.join('./output', 'sample_frame.jpg')
        cv2.imwrite(sample_path, sample_frame)
        print(f"DEBUG: Saved sample frame to {sample_path}")
        # Reset video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    try:
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if not success:
                print(f"DEBUG: End of video or error reading frame {frame_count}")
                break
                
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"DEBUG: Processing frame {frame_count}/{total_frames}")

            # Run YOLOv8 inference on the frame
            try:
                annotated_frame = cv2.resize(frame, (width, height))
                result = model(annotated_frame, conf=0.5, verbose=False)[0]
                
                # Get the players boxes and kit colors
                players_imgs, players_boxes = get_players_boxes(result)
                kits_colors = get_kits_colors(players_imgs, grass_hsv, annotated_frame)

                # Run on the first frame only
                if frame_count == 1:
                    print("DEBUG: Processing first frame for team classification")
                    if len(kits_colors) < 2:
                        print("WARNING: Not enough players detected in first frame, might cause issues with team classification")
                    
                    kits_clf = get_kits_classifier(kits_colors)
                    left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf)
                    grass_color = get_grass_color(result.orig_img)
                    grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)
                    
                    # Save first frame with annotations for debugging
                    first_frame_path = os.path.join('./output', 'first_annotated_frame.jpg')
                    cv2.imwrite(first_frame_path, annotated_frame)
                    print(f"DEBUG: Saved first annotated frame to {first_frame_path}")

                # Process all detected objects
                for box in result.boxes:
                    label = int(box.cls.numpy()[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())

                    # If the box contains a player, find to which team he belongs
                    if label == 0:
                        kit_color = get_kits_colors([result.orig_img[y1: y2, x1: x2]], grass_hsv)
                        team = classify_kits(kits_clf, kit_color)
                        if team == left_team_label:
                            label = 0
                        else:
                            label = 1

                    # If the box contains a Goalkeeper, find to which team he belongs
                    elif label == 1:
                        if x1 < 0.5 * width:
                            label = 2
                        else:
                            label = 3

                    # Increase the label by 2 because of the two add labels "Player-L", "GK-L"
                    else:
                        label = label + 2

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_colors[str(label)], 2)
                    cv2.putText(annotated_frame, labels[label], (x1 - 30, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                box_colors[str(label)], 2)

                # Write the annotated frame
                output_video.write(annotated_frame)
                
                # Save a sample frame occasionally for debugging
                if frame_count % 100 == 0:
                    frame_sample_path = os.path.join('./output', f'frame_{frame_count}.jpg')
                    cv2.imwrite(frame_sample_path, annotated_frame)
                    print(f"DEBUG: Saved sample frame {frame_count} to {frame_sample_path}")
                
            except Exception as e:
                print(f"ERROR processing frame {frame_count}: {e}")
                # Save the problematic frame for debugging
                error_frame_path = os.path.join('./output', f'error_frame_{frame_count}.jpg')
                cv2.imwrite(error_frame_path, frame)
                print(f"DEBUG: Saved error frame to {error_frame_path}")

        print(f"DEBUG: Processed {frame_count} frames out of {total_frames}")
        
    except Exception as e:
        print(f"ERROR: Unexpected error during video processing: {e}")
    finally:
        # Make sure resources are properly released
        print("DEBUG: Releasing video resources...")
        
        # Check if we've successfully written frames
        if output_video.isOpened():
            frames_written = int(output_video.get(cv2.CAP_PROP_FRAMES))
            print(f"DEBUG: Wrote {frames_written} frames to output video")
        
        # Explicitly flush any buffered frames
        try:
            output_video.release()
            print("DEBUG: Output video released")
        except Exception as e:
            print(f"ERROR: Problem releasing output video: {e}")
            
        cap.release()
        print("DEBUG: Input video released")
        
        cv2.destroyAllWindows()
        print("DEBUG: All windows destroyed")
        
        # Check if output file exists and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"DEBUG: Output file exists with size {file_size} bytes")
            if file_size == 0:
                print("ERROR: Output file exists but has zero size!")
        else:
            print(f"ERROR: Output file was not created at {output_path}")
            
        print(f"DEBUG: Video processing {'completed successfully' if frame_count > 0 else 'failed'}")

if __name__ == "__main__":
    print("DEBUG: Script started")
    
    labels = ["Player-L", "Player-R", "GK-L", "GK-R", "Ball", "Main Ref", "Side Ref", "Staff"]
    box_colors = {
        "0": (150, 50, 50),
        "1": (37, 47, 150),
        "2": (41, 248, 165),
        "3": (166, 196, 10),
        "4": (155, 62, 157),
        "5": (123, 174, 213),
        "6": (217, 89, 204),
        "7": (22, 11, 15)
    }
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("ERROR: Video path not provided. Usage: python script.py video_path")
        sys.exit(1)
        
    video_path = sys.argv[1]
    print(f"DEBUG: Video path from arguments: {video_path}")
    
    # Check if model path exists and adjust if necessary
    model_path = "/kaggle/input/yolo_weights_object_detection/pytorch/default/1/last.pt"
    if not os.path.exists(model_path):
        print(f"WARNING: Model not found at {model_path}")
        # Check for local model
        if os.path.exists("./last.pt"):
            model_path = "./last.pt"
            print(f"DEBUG: Using local model at {model_path}")
        else:
            print("ERROR: YOLO model not found! Please provide a valid model path.")
            sys.exit(1)
    
    try:
        print(f"DEBUG: Loading YOLO model from {model_path}")
        model = YOLO(model_path)
        print("DEBUG: YOLO model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model: {e}")
        sys.exit(1)
        
    try:
        print("DEBUG: Starting video annotation")
        annotate_video(video_path, model)
        print("DEBUG: Video annotation complete")
    except Exception as e:
        print(f"ERROR: Unhandled exception in main process: {e}")
