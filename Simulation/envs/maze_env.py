import json
import os
import time
import pygame
import heapq
import numpy as np


# ---------------------------------------------------------------------------
# Layout constants  (map area is computed dynamically from screen size)
# ---------------------------------------------------------------------------
TOP_H    = 70    # top status bar height
LEFT_W   = 270   # left parameter panel width
RIGHT_W  = 270   # right performance panel width
# MAP_PX is set dynamically inside env_setup via get_dynamic_sizes()

# Colour palette
C_BG          = (18,  20,  30)    # window background
C_PANEL_BG    = (28,  32,  48)    # panel background
C_PANEL_EDGE  = (60,  70, 100)    # panel border
C_TOP_BG      = (22,  26,  40)    # top bar background
C_TRAIN       = (50, 220, 120)    # training mode label colour
C_TEST        = (80, 180, 255)    # testing mode label colour
C_WHITE       = (235, 235, 245)
C_LABEL       = (150, 165, 200)   # dim label text
C_VALUE       = (235, 240, 255)   # bright value text
C_TITLE       = (255, 255, 255)
C_WALL        = (40,  45,  65)
C_FREE        = (230, 232, 240)
C_PATH        = (80, 140, 255)
C_SOURCE      = (60, 220, 110)
C_DEST        = (255,  70, 140)
C_WAYPOINT    = (255, 165,  60)
C_AGENT_BODY  = (50,  200, 255)
C_AGENT_EYE   = (255, 255, 255)
C_AGENT_PUPIL = (20,   20,  20)
C_AGENT_ANT   = (255, 220,  50)


def collect_waypoints(path):
    waypoints, count = [], 0
    waypoint_interval = max(1, len(path) // 7)
    for i in range(waypoint_interval, len(path) - 1, waypoint_interval):
        waypoints.append(path[i])
        count += 1
    return waypoints, count


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class MazeEnv:
    def __init__(self):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'env_params.json', 'r') as f:
            self.env_params = json.load(f)
        with open(param_dir + 'train_params.json', 'r') as f:
            self.train_params = json.load(f)
        with open(param_dir + 'sim_params.json', 'r') as f:
            self.sim_params = json.load(f)

        # Merge for legacy compatibility
        self.params = {**self.env_params, **self.train_params, **self.sim_params}

        self.data_dir   = self.params['DATA_DIR'] + self.params['MAZE_DIR']
        self.maze_size  = self.params['SIZE']

        # cell_size / map_px computed after pygame init in env_setup
        self.cell_size  = self.params['CELL_SIZE']
        self.map_px     = self.cell_size * self.maze_size
        self.window_size = (LEFT_W + self.map_px + RIGHT_W,
                            TOP_H  + self.map_px)

        self.maze      = None
        self.is_guided_maze = bool(self.params['GUIDED_MAZE'])
        self.directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

        self.screen = None
        self.clock  = None
        self.path   = None
        self.fps    = self.params['FPS']

        self.source      = None
        self.destination = None
        self.total_waypoints = None
        self.waypoints   = [] if self.is_guided_maze else None

        # Mode set externally by Simulation before rendering
        self.mode       = 'TRAINING'
        self.start_time = None

        # Pygame font handles (set in env_setup)
        self._font_lg  = None
        self._font_md  = None
        self._font_sm  = None

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    def env_setup(self):
        pygame.init()
        pygame.font.init()

        # Open a tiny throwaway window first so display.Info() returns real dimensions
        pygame.display.set_mode((1, 1))
        info  = pygame.display.Info()
        scr_w = info.current_w  if info.current_w  > 100 else 1920
        scr_h = info.current_h  if info.current_h  > 100 else 1080

        # Window = 75 % of screen
        win_w = int(scr_w * 0.75)
        win_h = int(scr_h * 0.75)

        # Largest square map that fits in the remaining area
        avail_w = win_w - LEFT_W - RIGHT_W
        avail_h = win_h - TOP_H
        map_px  = min(avail_w, avail_h)
        cell_sz = max(1, map_px // self.maze_size)

        self.cell_size   = cell_sz
        self.map_px      = cell_sz * self.maze_size
        self.window_size = (LEFT_W + self.map_px + RIGHT_W,
                            TOP_H  + self.map_px)

        # Centre window on primary screen
        x = (scr_w - self.window_size[0]) // 2
        y = (scr_h - self.window_size[1]) // 2
        os.environ['SDL_VIDEO_WINDOW_POS'] = f'{x},{y}'

        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('Deep RL \u2014 Maze Navigation Simulation')
        self.clock      = pygame.time.Clock()
        self.start_time = time.time()

        self._font_title = pygame.font.SysFont('Segoe UI', 26, bold=True)
        self._font_mode  = pygame.font.SysFont('Segoe UI', 13, bold=True)
        self._font_lg    = pygame.font.SysFont('Segoe UI', 17, bold=True)
        self._font_md    = pygame.font.SysFont('Segoe UI', 14)
        self._font_sm    = pygame.font.SysFont('Segoe UI', 12)

    # -----------------------------------------------------------------------
    # Environment helpers (unchanged)
    # -----------------------------------------------------------------------
    def distance_to_goal(self, agent_position):
        return np.linalg.norm(agent_position - self.destination)

    def distance_to_next_point(self, agent_position):
        return np.linalg.norm(agent_position - self.waypoints[1])

    def find_path(self):
        self.a_star()

    def load_src_dst(self):
        filename = self.params['LOCATION_FILENAME']
        location = np.load(self.data_dir + filename)
        self.source, self.destination = location[0], location[1]

    def generate_maze(self, p):
        n = int((self.maze_size - 2) ** 2 * p)
        self.maze = np.ones((self.maze_size, self.maze_size), dtype=int)
        walls = []
        start_x = np.random.choice(range(1, self.maze_size - 1))
        start_y = 1
        self.maze[start_x, start_y] = 0
        for direction in self.directions:
            nx, ny = start_x + direction[0], start_y + direction[1]
            if 1 < nx < self.maze_size - 1 and 1 < ny < self.maze_size - 1:
                walls.append((nx, ny, start_x, start_y))
        while walls:
            wall = walls[np.random.choice(len(walls))]
            x, y, px, py = wall
            if self.maze[x, y] == 1:
                adjacent = sum(
                    self.maze_size - 1 > x + d[0] >= 1 and
                    self.maze_size - 1 > y + d[1] >= 1 and
                    self.maze[x + d[0], y + d[1]] == 0
                    for d in self.directions
                )
                if adjacent == 1:
                    self.maze[x, y] = 0
                    for d in self.directions:
                        nx, ny = x + d[0], y + d[1]
                        if self.maze_size - 1 > nx > 0 and self.maze_size - 1 > ny > 0 and self.maze[nx, ny] == 1:
                            walls.append((nx, ny, x, y))
            walls.remove(wall)
        internal = [(i, j) for i in range(1, self.maze_size - 1)
                    for j in range(1, self.maze_size - 1)]
        np.random.shuffle(internal)
        count = np.sum(self.maze[1:-1, 1:-1] == 1)
        while count < n:
            x, y = internal.pop()
            if self.maze[x, y] == 0:
                self.maze[x, y] = 1
                count += 1
        while count > n:
            x, y = internal.pop()
            if self.maze[x, y] == 1:
                self.maze[x, y] = 0
                count -= 1

    def generate_src_dst(self):
        self.source = np.array((1, np.random.choice(range(1, self.maze_size))))
        self.destination = np.array((self.maze_size - 2, np.random.choice(range(1, self.maze_size))))
        while not self.is_valid_position(self.source):
            self.source = np.array((1, np.random.choice(range(1, self.maze_size))))
        while not self.is_valid_position(self.destination):
            self.destination = np.array((self.maze_size - 2, np.random.choice(range(1, self.maze_size))))
        filename = self.params['LOCATION_FILENAME']
        if self.source is not None and self.destination is not None:
            location = np.vstack((self.source, self.destination))
            np.save(self.data_dir + filename, location)

    def is_valid_position(self, position):
        x, y = position
        return (0 <= x < self.maze_size and
                0 <= y < self.maze_size and
                self.maze[x, y] == 0)

    def a_star(self):
        start = tuple(self.source) if isinstance(self.source, np.ndarray) else self.source
        goal  = tuple(self.destination) if isinstance(self.destination, np.ndarray) else self.destination
        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), 0, start))
        came_from = {}
        g_score   = {start: 0}
        f_score   = {start: heuristic(start, goal)}
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            if current == goal:
                self.path = []
                while current in came_from:
                    self.path.append(current)
                    current = came_from[current]
                self.path.append(start)
                self.path.reverse()
                if self.is_guided_maze:
                    self.waypoints, self.total_waypoints = collect_waypoints(self.path)
                return self.path
            for d in self.directions:
                nb = (current[0] + d[0], current[1] + d[1])
                if self.is_valid_position(nb):
                    tg = g_score[current] + 1
                    if nb not in g_score or tg < g_score[nb]:
                        came_from[nb] = current
                        g_score[nb]   = tg
                        f_score[nb]   = tg + heuristic(nb, goal)
                        if nb not in [i[2] for i in open_set]:
                            heapq.heappush(open_set, (f_score[nb], tg, nb))

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------
    def update_display(self, agent):
        self.screen.fill(C_BG)
        self._draw_top_bar()
        self._draw_left_panel(agent)
        self._draw_right_panel()
        self._draw_maze()
        self._draw_agent(agent)
        pygame.display.update()
        self.clock.tick(self.fps)

    # --- Top bar -----------------------------------------------------------
    def _draw_top_bar(self):
        bar_rect = pygame.Rect(0, 0, self.window_size[0], TOP_H)
        pygame.draw.rect(self.screen, C_TOP_BG, bar_rect)
        pygame.draw.line(self.screen, C_PANEL_EDGE,
                         (0, TOP_H - 1), (self.window_size[0], TOP_H - 1), 2)

        # -- Large bold simulation title (centred) --
        title_surf = self._font_title.render('Deep RL  —  Maze Navigation Simulation', True, C_WHITE)
        self.screen.blit(title_surf,
                         (self.window_size[0] // 2 - title_surf.get_width() // 2,
                          TOP_H // 2 - title_surf.get_height() // 2))

        # -- Mode badge (smaller, left of map) --
        colour = C_TRAIN if self.mode == 'TRAINING' else C_TEST
        badge_text  = f'  {self.mode}  '
        mode_surf   = self._font_mode.render(badge_text, True, C_BG, colour)
        badge_x     = LEFT_W + 8
        badge_y     = TOP_H // 2 - mode_surf.get_height() // 2
        self.screen.blit(mode_surf, (badge_x, badge_y))

        # -- Elapsed timer (right of map) --
        elapsed = int(time.time() - self.start_time) if self.start_time else 0
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)
        timer_txt  = f'{h:02d}:{m:02d}:{s:02d}'
        timer_surf = self._font_mode.render(timer_txt, True, C_WHITE)
        tx = self.window_size[0] - RIGHT_W - timer_surf.get_width() - 12
        self.screen.blit(timer_surf, (tx, TOP_H // 2 - timer_surf.get_height() // 2))

    # --- Left panel --------------------------------------------------------
    def _draw_left_panel(self, agent):
        panel = pygame.Rect(0, TOP_H, LEFT_W, self.map_px)
        pygame.draw.rect(self.screen, C_PANEL_BG, panel)
        pygame.draw.line(self.screen, C_PANEL_EDGE,
                         (LEFT_W - 1, TOP_H), (LEFT_W - 1, TOP_H + self.map_px), 2)

        # Panel title strip with background
        title_strip = pygame.Rect(0, TOP_H, LEFT_W, 36)
        pygame.draw.rect(self.screen, (38, 44, 66), title_strip)
        title = self._font_lg.render('  Parameters', True, C_TITLE)
        self.screen.blit(title, (8, TOP_H + 8))
        pygame.draw.line(self.screen, C_PANEL_EDGE,
                         (0, TOP_H + 36), (LEFT_W, TOP_H + 36), 1)

        # Gather values
        eps_val   = f'{agent.model.epsilon:.4f}' if (agent.model and hasattr(agent.model, 'epsilon')) else 'N/A'
        ep_curr   = getattr(agent, 'current_episode', 0) + 1
        ep_total  = getattr(agent, 'total_episodes', '?')
        step_val  = agent.game_steps
        ep_reward = getattr(agent, 'episode_reward', 0.0)
        cum_rew   = getattr(agent, 'cumulative_reward', 0.0)
        goal_cnt  = getattr(agent, 'goal_count', 0)

        rows = [
            ('Model',       str(agent.model_name or 'N/A')),
            ('Grid',        f'{self.maze_size} x {self.maze_size}'),
            ('Run (Trial)', str(getattr(agent, 'epch', '?'))),
            ('',            ''),
            ('Episode',     f'{ep_curr} / {ep_total}'),
            ('Step',        str(step_val)),
            ('Ep. Reward',  f'{ep_reward:.3f}'),
            ('Cum. Reward', f'{cum_rew:.2f}'),
            ('Goal Reached',str(goal_cnt)),
            ('',            ''),
            ('Epsilon',     eps_val),
            ('Gamma',       str(self.train_params.get('GAMMA', '?'))),
            ('Alpha',       str(self.train_params.get('ALPHA', '?'))),
            ('Batch Size',  str(self.train_params.get('BATCH_SIZE', '?'))),
        ]

        y = TOP_H + 46
        for label, value in rows:
            if label == '':
                y += 6
                continue
            lbl_s = self._font_sm.render(label, True, C_LABEL)
            val_s = self._font_md.render(value, True, C_VALUE)
            self.screen.blit(lbl_s, (12, y))
            # Right-align value
            vx = LEFT_W - val_s.get_width() - 12
            self.screen.blit(val_s, (vx, y - 1))
            y += 22

    # --- Right panel -------------------------------------------------------
    def _draw_right_panel(self):
        rx = LEFT_W + self.map_px
        panel = pygame.Rect(rx, TOP_H, RIGHT_W, self.map_px)
        pygame.draw.rect(self.screen, C_PANEL_BG, panel)
        pygame.draw.line(self.screen, C_PANEL_EDGE,
                         (rx, TOP_H), (rx, TOP_H + self.map_px), 2)

        # Panel title strip with background
        title_strip = pygame.Rect(rx, TOP_H, RIGHT_W, 36)
        pygame.draw.rect(self.screen, (38, 44, 66), title_strip)
        title = self._font_lg.render('  Performance', True, C_TITLE)
        self.screen.blit(title, (rx + 8, TOP_H + 8))
        pygame.draw.line(self.screen, C_PANEL_EDGE,
                         (rx, TOP_H + 36), (rx + RIGHT_W, TOP_H + 36), 1)

        # Placeholder
        ph = self._font_sm.render('Plot area', True, C_LABEL)
        self.screen.blit(ph,
                         (rx + RIGHT_W // 2 - ph.get_width() // 2,
                          TOP_H + self.map_px // 2))

    # --- Maze grid ---------------------------------------------------------
    def _draw_maze(self):
        ox, oy = LEFT_W, TOP_H    # map origin offset
        for row in range(self.maze_size):
            for col in range(self.maze_size):
                colour = C_WALL if self.maze[row, col] == 1 else C_FREE
                pygame.draw.rect(self.screen, colour,
                                 pygame.Rect(ox + col * self.cell_size,
                                             oy + row * self.cell_size,
                                             self.cell_size, self.cell_size))
                # Cell border
                pygame.draw.rect(self.screen, C_PANEL_EDGE,
                                 pygame.Rect(ox + col * self.cell_size,
                                             oy + row * self.cell_size,
                                             self.cell_size, self.cell_size), 1)

        # A* optimal path
        if self.path:
            for i in range(len(self.path) - 1):
                s = (ox + self.path[i][1]     * self.cell_size + self.cell_size // 2,
                     oy + self.path[i][0]     * self.cell_size + self.cell_size // 2)
                e = (ox + self.path[i + 1][1] * self.cell_size + self.cell_size // 2,
                     oy + self.path[i + 1][0] * self.cell_size + self.cell_size // 2)
                pygame.draw.line(self.screen, C_PATH, s, e, 2)

        # Source
        pygame.draw.circle(self.screen, C_SOURCE,
                           (ox + self.source[1] * self.cell_size + self.cell_size // 2,
                            oy + self.source[0] * self.cell_size + self.cell_size // 2),
                           max(4, self.cell_size // 3))

        # Destination — pulsing star-like marker
        dx = ox + self.destination[1] * self.cell_size + self.cell_size // 2
        dy = oy + self.destination[0] * self.cell_size + self.cell_size // 2
        r  = max(4, self.cell_size // 3)
        pygame.draw.circle(self.screen, C_DEST, (dx, dy), r)
        pygame.draw.circle(self.screen, C_WHITE, (dx, dy), max(2, r // 2))

        # Waypoints
        if self.is_guided_maze and self.waypoints:
            for wp in self.waypoints:
                pygame.draw.circle(self.screen, C_WAYPOINT,
                                   (ox + wp[1] * self.cell_size + self.cell_size // 2,
                                    oy + wp[0] * self.cell_size + self.cell_size // 2),
                                   max(3, self.cell_size // 4))

    # --- Crazy agent (robot face) -----------------------------------------
    def _draw_agent(self, agent):
        ox, oy = LEFT_W, TOP_H
        cx = ox + agent.position[1] * self.cell_size + self.cell_size // 2
        cy = oy + agent.position[0] * self.cell_size + self.cell_size // 2
        r  = max(4, self.cell_size // 2 - 2)

        # Body / head
        pygame.draw.circle(self.screen, C_AGENT_BODY, (cx, cy), r)
        pygame.draw.circle(self.screen, C_WHITE, (cx, cy), r, 2)

        if r >= 7:
            # Antenna
            ant_top = (cx, cy - r - max(3, r // 2))
            pygame.draw.line(self.screen, C_AGENT_ANT, (cx, cy - r), ant_top, 2)
            pygame.draw.circle(self.screen, C_AGENT_ANT, ant_top, max(2, r // 5))

            # Eyes
            eye_r  = max(2, r // 4)
            eye_ox = max(2, r // 3)
            for ex in (cx - eye_ox, cx + eye_ox):
                ey = cy - max(1, r // 6)
                pygame.draw.circle(self.screen, C_AGENT_EYE,  (ex, ey), eye_r)
                pygame.draw.circle(self.screen, C_AGENT_PUPIL, (ex, ey), max(1, eye_r // 2))

            # Grin
            grin_rect = pygame.Rect(cx - eye_ox, cy + max(1, r // 5),
                                    eye_ox * 2, max(2, r // 3))
            pygame.draw.arc(self.screen, C_WHITE, grin_rect, 0, 3.14, max(1, r // 6))
