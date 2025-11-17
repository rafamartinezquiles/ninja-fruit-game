import cv2
import time
import random
import math
import numpy as np
import mediapipe as mp
from collections import deque
import os  # for paths

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
SLASH_THICKNESS = 5          # <<< thinner slash line

# Fruit movement speed (initial)
INITIAL_FRUIT_VELOCITY = np.array([0, -5], dtype=np.int32)  # [vx, vy]

# Spawning
INITIAL_SPAWN_RATE = 1.0     # fruits per second

# Fruit logos (with paths to ../images)
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
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # keep alpha
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
    """
    if sprite is None:
        return

    sprite_resized = cv2.resize(sprite, (size, size), interpolation=cv2.INTER_AREA)

    h, w, _ = frame.shape
    cx, cy = center

    # top-left of sprite
    x1 = int(cx - size / 2)
    y1 = int(cy - size / 2)
    x2 = x1 + size
    y2 = y1 + size

    # completely off screen?
    if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
        return

    # clip to frame
    x1_clip = max(x1, 0)
    y1_clip = max(y1, 0)
    x2_clip = min(x2, w)
    y2_clip = min(y2, h)

    sprite_x1 = x1_clip - x1
    sprite_y1 = y1_clip - y1
    sprite_x2 = sprite_x1 + (x2_clip - x1_clip)
    sprite_y2 = sprite_y1 + (y2_clip - y1_clip)

    roi = frame[y1_clip:y2_clip, x1_clip:x2_clip]
    sprite_roi = sprite_resized[sprite_y1:sprite_y2, sprite_x1:sprite_x2]

    if sprite_roi.shape[2] == 4:
        # BGRA
        sprite_rgb = sprite_roi[..., :3]
        alpha = sprite_roi[..., 3:] / 255.0
        roi[:] = (1.0 - alpha) * roi + alpha * sprite_rgb
    else:
        roi[:] = sprite_roi


# ============================
# Fruit class
# ============================

class Fruit:
    """
    Represents a single fruit on screen.
    """
    def __init__(self, start_position, velocity, color, sprite):
        self.position = np.array(start_position, dtype=np.int32)
        self.velocity = np.array(velocity, dtype=np.int32)
        self.color = color
        self.sprite = sprite

    def update(self):
        self.position += self.velocity

    def draw(self, frame):
        if self.sprite is not None:
            draw_sprite_centered(frame, self.sprite, tuple(self.position), FRUIT_SIZE)
        else:
            cv2.circle(frame, tuple(self.position), FRUIT_RADIUS, self.color, -1)

    def is_out_of_bounds(self, frame_width, frame_height):
        x, y = self.position
        if y < 0 or x < 0 or x > frame_width + 50:
            return True
        return False


# ============================
# Game state initialization
# ============================

cap = cv2.VideoCapture(0)

frame_width = 0
frame_height = 0

previous_frame_time = time.time()

current_score = 0
current_level = 1
player_lives = INITIAL_LIVES
game_over = False

fruit_velocity_base = INITIAL_FRUIT_VELOCITY.copy()
spawn_rate = INITIAL_SPAWN_RATE
next_spawn_time = 0.0
next_level_score_threshold = SCORE_PER_LEVEL

slash_points = deque(maxlen=SLASH_MAX_POINTS)
slash_color = (255, 255, 255)

active_fruits = []

fruit_sprites = load_fruit_sprites(FRUIT_IMAGE_FILES)

# NEW: start menu flag
game_started = False


def reset_game():
    """
    Reset all game-related variables (called when starting or restarting).
    """
    global current_score, current_level, player_lives, game_over
    global fruit_velocity_base, spawn_rate, next_spawn_time
    global next_level_score_threshold, active_fruits, slash_points, slash_color

    current_score = 0
    current_level = 1
    player_lives = INITIAL_LIVES
    game_over = False

    fruit_velocity_base = INITIAL_FRUIT_VELOCITY.copy()
    spawn_rate = INITIAL_SPAWN_RATE
    next_spawn_time = time.time()
    next_level_score_threshold = SCORE_PER_LEVEL

    active_fruits = []
    slash_points.clear()
    slash_color = (255, 255, 255)


def spawn_fruit():
    """
    Create a new fruit at a random x-position near the bottom of the frame.
    """
    global active_fruits, frame_width, frame_height

    if frame_width == 0:
        spawn_x = random.randint(15, 600)
        spawn_y = 440
    else:
        spawn_x = random.randint(15, max(16, frame_width - 15))
        spawn_y = frame_height - 40

    random_color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

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

    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = hand_tracker.process(frame_rgb)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    fingertip_position = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark_id, lm in enumerate(hand_landmarks.landmark):
                if landmark_id == 8:
                    x_px = int(lm.x * frame_width)
                    y_px = int(lm.y * frame_height)
                    fingertip_position = (x_px, y_px)

                    # thinner fingertip marker to match thinner slash
                    cv2.circle(frame, fingertip_position, 10, slash_color, -1)

                    slash_points.append(fingertip_position)

    # =======================
    # If game hasn't started: show menu and skip game logic
    # =======================
    if not game_started:
        title = "FRUIT SLASH"
        instruction = "Press ENTER to start"
        quit_msg = "Press Q to quit"

        cv2.putText(frame, title,
                    (int(frame_width * 0.1), int(frame_height * 0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.putText(frame, instruction,
                    (int(frame_width * 0.1), int(frame_height * 0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, quit_msg,
                    (int(frame_width * 0.1), int(frame_height * 0.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show current camera frame with menu text
        cv2.imshow("Fruit Slash", frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        if key == 13:  # ENTER key
            reset_game()
            game_started = True
        continue  # go to next loop iteration, skip game logic

    # =======================
    # Check fruit slicing (only when game started)
    # =======================
    if fingertip_position is not None and not game_over:
        for fruit in list(active_fruits):
            dist = euclidean_distance(fingertip_position, fruit.position)
            if dist < FRUIT_RADIUS:
                current_score += POINTS_PER_FRUIT
                slash_color = fruit.color
                active_fruits.remove(fruit)

    # =======================
    # Difficulty scaling
    # =======================
    if current_score >= next_level_score_threshold:
        current_level += 1
        next_level_score_threshold += SCORE_PER_LEVEL

        spawn_rate = current_level * 4.0 / 5.0
        fruit_velocity_base[1] = int(-5 * current_level / 2)

        print(f"Level up! -> Level {current_level}")
        print("New base velocity:", fruit_velocity_base, "New spawn rate:", spawn_rate)

    # =======================
    # Update fruits (movement & despawn)
    # =======================
    if player_lives <= 0:
        game_over = True

    if not game_over:
        current_time = time.time()
        if current_time > next_spawn_time:
            spawn_fruit()
            next_spawn_time = current_time + (1.0 / spawn_rate)

        for fruit in list(active_fruits):
            fruit.update()
            fruit.draw(frame)

            if fruit.is_out_of_bounds(frame_width, frame_height):
                player_lives -= 1
                active_fruits.remove(fruit)
                if player_lives <= 0:
                    game_over = True
    else:
        cv2.putText(
            frame,
            "GAME OVER",
            (int(frame_width * 0.1), int(frame_height * 0.6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 255),
            3
        )
        cv2.putText(
            frame,
            "Press ENTER to restart",
            (int(frame_width * 0.1), int(frame_height * 0.75)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        active_fruits.clear()

    # =======================
    # Draw slash trail
    # =======================
    if len(slash_points) > 1:
        slash_array = np.array(slash_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [slash_array], False, slash_color,
                      SLASH_THICKNESS, lineType=cv2.LINE_AA)

    # =======================
    # FPS calculation and HUD
    # =======================
    current_frame_time = time.time()
    delta_time = current_frame_time - previous_frame_time
    previous_frame_time = current_frame_time
    fps = int(1 / delta_time) if delta_time > 0 else 0

    cv2.putText(frame, f"FPS: {fps}", (int(frame_width * 0.82), 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 0), 2)

    cv2.putText(frame, f"Score: {current_score}", (int(frame_width * 0.35), 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)

    cv2.putText(frame, f"Level: {current_level}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 150), 5)

    cv2.putText(frame, f"Lives: {player_lives}", (200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # =======================
    # Show frame and handle keys
    # =======================
    cv2.imshow("Fruit Slash", frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    if game_over and key == 13:  # ENTER to restart after game over
        reset_game()
        game_started = True

# Cleanup
cap.release()
cv2.destroyAllWindows()
