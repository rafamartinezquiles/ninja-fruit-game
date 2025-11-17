import cv2
import time
import random
import math
import numpy as np
import mediapipe as mp
from collections import deque
import os  # <-- added

# ============================
# Paths: images/ is next to src/
# ============================
# Directory of this file (src/)
BASE_DIR = os.path.dirname(__file__)
# ../images
ASSETS_DIR = os.path.join(BASE_DIR, "..", "images")

# ============================
# Mediapipe setup (hand tracking)
# ============================
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Hands model (tuned for real-time video)
hand_tracker = mp_hands.Hands(
    static_image_mode=False,        # video stream, not single images
    max_num_hands=1,               # only need one hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ============================
# Game configuration constants
# ============================

# Fruit appearance
FRUIT_SIZE = 60              # diameter of fruit image in pixels
FRUIT_RADIUS = FRUIT_SIZE // 2

# Lives and scoring
INITIAL_LIVES = 15
POINTS_PER_FRUIT = 100
SCORE_PER_LEVEL = 1000       # every 1000 points, difficulty increases

# Slash (finger trail) configuration
SLASH_MAX_POINTS = 19        # how long the trail is
SLASH_THICKNESS = 15

# Fruit movement speed (initial)
INITIAL_FRUIT_VELOCITY = np.array([0, -5], dtype=np.int32)  # [vx, vy]

# Spawning
INITIAL_SPAWN_RATE = 1.0     # fruits per second

# Fruit logos - NOTE: now using ../images/ paths
FRUIT_IMAGE_FILES = [
    os.path.join(ASSETS_DIR, "apple.png"),
    os.path.join(ASSETS_DIR, "banana.png"),
    os.path.join(ASSETS_DIR, "orange.png"),
    os.path.join(ASSETS_DIR, "strawberry.png"),
]

# ============================
# Utility functions
# ============================

def load_fruit_sprites(file_list):
    """
    Load fruit images from disk.
    Returns a list of images (some may be None if loading fails).
    """
    sprites = []
    for path in file_list:
        # IMREAD_UNCHANGED keeps the alpha channel (transparency) if it exists
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            sprites.append(img)
        else:
            print(f"[WARNING] Could not load sprite: {path}")
    return sprites


def euclidean_distance(p1, p2):
    """
    Compute Euclidean distance between two 2D points (x1, y1) and (x2, y2).
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return int(math.sqrt(dx * dx + dy * dy))


def draw_sprite_centered(frame, sprite, center, size):
    """
    Draw a sprite (with possible alpha channel) on 'frame',
    centered at 'center' (x, y), scaled to 'size' x 'size'.

    If the sprite has 4 channels (BGRA), use alpha blending.
    Otherwise, just overwrite pixels.
    """
    if sprite is None:
        return

    # Resize the sprite to the desired size
    sprite_resized = cv2.resize(sprite, (size, size), interpolation=cv2.INTER_AREA)

    h, w, _ = frame.shape
    cx, cy = center

    # Calculate top-left corner such that sprite is centered
    x1 = int(cx - size / 2)
    y1 = int(cy - size / 2)
    x2 = x1 + size
    y2 = y1 + size

    # Clip to the frame boundaries
    if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
        return  # completely off-screen

    x1_clip = max(x1, 0)
    y1_clip = max(y1, 0)
    x2_clip = min(x2, w)
    y2_clip = min(y2, h)

    # Compute corresponding coordinates on the sprite
    sprite_x1 = x1_clip - x1
    sprite_y1 = y1_clip - y1
    sprite_x2 = sprite_x1 + (x2_clip - x1_clip)
    sprite_y2 = sprite_y1 + (y2_clip - y1_clip)

    # Extract regions
    roi = frame[y1_clip:y2_clip, x1_clip:x2_clip]
    sprite_roi = sprite_resized[sprite_y1:sprite_y2, sprite_x1:sprite_x2]

    if sprite_roi.shape[2] == 4:
        # Sprite has alpha channel (BGRA)
        sprite_rgb = sprite_roi[..., :3]
        alpha = sprite_roi[..., 3:] / 255.0  # normalize alpha to [0,1]
        # Blend sprite with background
        roi[:] = (1.0 - alpha) * roi + alpha * sprite_rgb
    else:
        # No alpha channel, simple overwrite
        roi[:] = sprite_roi


# ============================
# Fruit class
# ============================

class Fruit:
    """
    Represents a single fruit on screen:
    - position
    - velocity
    - color (for slash color effect)
    - sprite (image)
    """
    def __init__(self, start_position, velocity, color, sprite):
        self.position = np.array(start_position, dtype=np.int32)
        self.velocity = np.array(velocity, dtype=np.int32)
        self.color = color
        self.sprite = sprite

    def update(self):
        """Move the fruit according to its velocity."""
        self.position += self.velocity

    def draw(self, frame):
        """Draw the fruit sprite (or fallback circle) on the frame."""
        if self.sprite is not None:
            draw_sprite_centered(frame, self.sprite, tuple(self.position), FRUIT_SIZE)
        else:
            # Fallback to a simple circle if sprite missing
            cv2.circle(frame, tuple(self.position), FRUIT_RADIUS, self.color, -1)

    def is_out_of_bounds(self, frame_width, frame_height):
        """
        Check if the fruit has left the playable area.
        Here we consider 'out' if it goes above the top
        or far off to the right or left.
        """
        x, y = self.position
        if y < 0 or x < 0 or x > frame_width + 50:
            return True
        return False


# ============================
# Game state initialization
# ============================

# Camera
cap = cv2.VideoCapture(0)

# Will be updated after first frame
frame_width = 0
frame_height = 0

# Time/FPS
previous_frame_time = time.time()

# Game variables
current_score = 0
current_level = 1
player_lives = INITIAL_LIVES
game_over = False

# Difficulty / spawn control
fruit_velocity_base = INITIAL_FRUIT_VELOCITY.copy()
spawn_rate = INITIAL_SPAWN_RATE           # fruits per second
next_spawn_time = 0.0
next_level_score_threshold = SCORE_PER_LEVEL

# Slash trail: queue of recent fingertip positions
slash_points = deque(maxlen=SLASH_MAX_POINTS)
slash_color = (255, 255, 255)  # initial slash color (white)

# List of active fruit objects
active_fruits = []

# Load fruit sprites once at start
fruit_sprites = load_fruit_sprites(FRUIT_IMAGE_FILES)


def spawn_fruit():
    """
    Create a new fruit at a random x-position near the bottom of the frame.
    Uses a random color and a random sprite from 'fruit_sprites'.
    """
    global active_fruits, frame_width, frame_height

    # If size is still unknown (no frame yet), pick a reasonable default range
    if frame_width == 0:
        spawn_x = random.randint(15, 600)
        spawn_y = 440
    else:
        spawn_x = random.randint(15, max(16, frame_width - 15))
        spawn_y = frame_height - 40

    # Random color (used for slash effect)
    random_color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

    # Pick a random sprite from the loaded list (may be empty)
    sprite = random.choice(fruit_sprites) if fruit_sprites else None

    new_fruit = Fruit(
        start_position=[spawn_x, spawn_y],
        velocity=fruit_velocity_base,
        color=random_color,
        sprite=sprite
    )
    active_fruits.append(new_fruit)


# ============================
# Main game loop
# ============================

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Skipping frame (no camera input)")
        continue

    frame_height, frame_width, _ = frame.shape

    # Flip horizontally so it feels like a mirror, then convert to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False

    # Run hand tracking
    results = hand_tracker.process(frame_rgb)

    # Convert back to BGR for OpenCV drawing
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # =======================
    # Hand landmarks & slash
    # =======================
    fingertip_position = None  # index fingertip (landmark 8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on screen
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Iterate over landmarks to find index fingertip (id == 8)
            for landmark_id, lm in enumerate(hand_landmarks.landmark):
                if landmark_id == 8:
                    x_px = int(lm.x * frame_width)
                    y_px = int(lm.y * frame_height)
                    fingertip_position = (x_px, y_px)

                    # Optionally draw a small circle at the fingertip
                    cv2.circle(frame, fingertip_position, 18, slash_color, -1)

                    # Save position to slash trail
                    slash_points.append(fingertip_position)

    # =======================
    # Check fruit slicing
    # =======================
    if fingertip_position is not None:
        # We iterate over a copy of the list so we can safely remove fruits
        for fruit in list(active_fruits):
            dist = euclidean_distance(fingertip_position, fruit.position)

            # Debug: show distance on top of the fruit (optional)
            # cv2.putText(frame, str(dist), tuple(fruit.position),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            if dist < FRUIT_RADIUS:
                # Fruit sliced!
                current_score += POINTS_PER_FRUIT
                slash_color = fruit.color  # change slash color to fruit color
                active_fruits.remove(fruit)

    # =======================
    # Difficulty scaling
    # =======================
    if current_score >= next_level_score_threshold:
        # Level up
        current_level += 1
        next_level_score_threshold += SCORE_PER_LEVEL

        # Increase spawn rate and fruit speed
        spawn_rate = current_level * 4.0 / 5.0
        fruit_velocity_base[0] = int(fruit_velocity_base[0] * 1.0)  # horizontal speed (if you want)
        fruit_velocity_base[1] = int(-5 * current_level / 2)        # faster upward movement

        print(f"Level up! -> Level {current_level}")
        print("New base velocity:", fruit_velocity_base, "New spawn rate:", spawn_rate)

    # =======================
    # Update fruits (movement & despawn)
    # =======================
    if player_lives <= 0:
        game_over = True

    if not game_over:
        # Spawn fruits based on time
        current_time = time.time()
        if current_time > next_spawn_time:
            spawn_fruit()
            # Next spawn time: now + (1 / spawn_rate) seconds
            next_spawn_time = current_time + (1.0 / spawn_rate)

        # Update and draw fruits
        for fruit in list(active_fruits):
            fruit.update()
            fruit.draw(frame)

            if fruit.is_out_of_bounds(frame_width, frame_height):
                # Player missed this fruit -> lose a life
                player_lives -= 1
                active_fruits.remove(fruit)

                if player_lives <= 0:
                    game_over = True
    else:
        # Game over screen
        cv2.putText(
            frame,
            "GAME OVER",
            (int(frame_width * 0.1), int(frame_height * 0.6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 255),
            3
        )
        active_fruits.clear()

    # =======================
    # Draw slash trail
    # =======================
    if len(slash_points) > 1:
        # polylines expects an array of shape (n_points, 1, 2)
        slash_array = np.array(slash_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [slash_array], False, slash_color, SLASH_THICKNESS, lineType=cv2.LINE_AA)

    # =======================
    # FPS calculation and HUD
    # =======================
    current_frame_time = time.time()
    delta_time = current_frame_time - previous_frame_time
    previous_frame_time = current_frame_time

    # Avoid division by zero for FPS
    fps = int(1 / delta_time) if delta_time > 0 else 0

    # Heads-Up Display (HUD): FPS, score, level, lives
    cv2.putText(frame, f"FPS: {fps}", (int(frame_width * 0.82), 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 0), 2)

    cv2.putText(frame, f"Score: {current_score}", (int(frame_width * 0.35), 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)

    cv2.putText(frame, f"Level: {current_level}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 150), 5)

    cv2.putText(frame, f"Lives: {player_lives}", (200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # =======================
    # Show frame and handle quit
    # =======================
    cv2.imshow("Fruit Slash", frame)

    # Press 'q' to quit
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
