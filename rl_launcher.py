import json
import re
import sys
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets
import torch

ROOT = Path(__file__).resolve().parent
ROBOTIC_WAREHOUSE_DIR = ROOT / "robotic-warehouse"
SEAC_DIR = ROOT / "seac"
TRAFFIC_SCENARIO_DIR = SEAC_DIR / "seac" / "traffic" / "scenarios"
ENV_CONFIG_DIR = Path.home() / ".rl_mars_envs"


def _resolve_python_bin():
    candidates = [
        ROOT / ".venv_rl0" / "bin" / "python",
        ROOT / "venv" / "rl_env" / "bin" / "python",
        Path.home() / "venv" / "rl_env" / "bin" / "python",
        Path("/home/usern/venv/rl_env/bin/python"),
        Path("/home/utar/rl_path_planning/rl_env/bin/python"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


VENV_PYTHON = _resolve_python_bin()


def _seac_env_name(env_id):
    env_id = (env_id or "").strip()
    if env_id.startswith("rware-") and env_id.endswith("-v1"):
        return env_id[:-3] + "-v2"
    return env_id


def _env_id_from_config(path):
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return ""
    return str(data.get("env_id") or "").strip()

_LAYOUT_COLORS = {
    "x": QtGui.QColor("#6b6b6b"),
    ".": QtGui.QColor("#f7f7f7"),
    "g": QtGui.QColor("#4caf50"),
    "o": QtGui.QColor("#2f2f2f"),
    "|": QtGui.QColor("#cfe8ff"),
    "-": QtGui.QColor("#cfe8ff"),
    "^": QtGui.QColor("#cfe8ff"),
    "v": QtGui.QColor("#cfe8ff"),
    "<": QtGui.QColor("#cfe8ff"),
    ">": QtGui.QColor("#cfe8ff"),
}

_REWARD_TYPE_MAP = {
    "INDIVIDUAL": "RewardType.INDIVIDUAL",
    "GLOBAL": "RewardType.GLOBAL",
    "TWO_STAGE": "RewardType.TWO_STAGE",
}

_OBS_TYPE_MAP = {
    "FLATTENED": "ObservationType.FLATTENED",
    "IMAGE": "ObservationType.IMAGE",
    "IMAGE_DICT": "ObservationType.IMAGE_DICT",
}

_IMAGE_LAYER_OPTIONS = (
    "SHELVES",
    "REQUESTS",
    "AGENTS",
    "AGENT_DIRECTION",
    "AGENT_LOAD",
    "GOALS",
    "ACCESSIBLE",
    "AVAILABLE_DIRECTIONS",
)
_DEFAULT_IMAGE_LAYERS = (
    "SHELVES",
    "REQUESTS",
    "AGENTS",
    "GOALS",
    "ACCESSIBLE",
)

_LANE_DIR_ORDER = ("UP", "DOWN", "LEFT", "RIGHT")
_LANE_SYMBOL_BY_DIR = {
    "UP": "^",
    "DOWN": "v",
    "LEFT": "<",
    "RIGHT": ">",
}
_LANE_TOKEN_BY_DIRS = {
    ("UP",): "^",
    ("DOWN",): "v",
    ("LEFT",): "<",
    ("RIGHT",): ">",
    ("UP", "DOWN"): "|",
    ("LEFT", "RIGHT"): "-",
}
_LAYOUT_TOKEN_TO_DIRS = {
    "|": ("UP", "DOWN"),
    "-": ("LEFT", "RIGHT"),
    "^": ("UP",),
    "v": ("DOWN",),
    "<": ("LEFT",),
    ">": ("RIGHT",),
}
_SPAWN_DIR_INT_BY_TOKEN = {
    "UP": 0,
    "U": 0,
    "^": 0,
    "0": 0,
    "DOWN": 1,
    "D": 1,
    "V": 1,
    "1": 1,
    "LEFT": 2,
    "L": 2,
    "<": 2,
    "2": 2,
    "RIGHT": 3,
    "R": 3,
    ">": 3,
    "3": 3,
}
_SPAWN_DIR_LABEL_BY_INT = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


class LayoutTable(QtWidgets.QTableWidget):
    def __init__(self, rows, cols, apply_cell, parent=None):
        super().__init__(rows, cols, parent)
        self._apply_cell = apply_cell
        self._painting = False
        self._brush = "x"

        self.setMouseTracking(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setDefaultSectionSize(24)
        self.verticalHeader().setDefaultSectionSize(24)

    def set_brush(self, brush):
        self._brush = brush

    def _apply_at(self, pos):
        index = self.indexAt(pos)
        if not index.isValid():
            return
        self._apply_cell(index.row(), index.column(), self._brush)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._painting = True
            self._apply_at(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._painting:
            self._apply_at(event.pos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._painting = False
        super().mouseReleaseEvent(event)


class Launcher(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL-0 Env Launcher")
        self.resize(1180, 760)
        self._closing_after_stop = False

        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.started.connect(self._on_started)
        self.process.finished.connect(self._on_finished)
        self._force_kill_timer = QtCore.QTimer(self)
        self._force_kill_timer.setSingleShot(True)
        self._force_kill_timer.timeout.connect(self._force_kill_process)

        ENV_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.run_buttons = []
        self.env_config_combos = []

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        self.tabs = QtWidgets.QTabWidget()

        self.tabs.addTab(self._build_human_play_tab(), "Human Play")
        self.tabs.addTab(self._build_training_tab(), "Training")
        self.tabs.addTab(self._build_evaluation_tab(), "Evaluation")
        self.tabs.addTab(self._build_env_generator_tab(), "Env Generator")

        controls = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Idle")
        controls.addWidget(self.status_label)
        controls.addStretch(1)

        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_process)
        controls.addWidget(self.stop_button)

        self.clear_button = QtWidgets.QPushButton("Clear Log")
        self.clear_button.clicked.connect(self._clear_log)
        controls.addWidget(self.clear_button)

        main_layout.addLayout(controls)

        self.body_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.addWidget(self.tabs)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.body_splitter.addWidget(self.log)
        self.body_splitter.setStretchFactor(0, 5)
        self.body_splitter.setStretchFactor(1, 2)
        self.body_splitter.setSizes([520, 180])
        main_layout.addWidget(self.body_splitter, stretch=1)

        self._update_buttons()

    def _build_human_play_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)

        self.hp_env = QtWidgets.QLineEdit("rware-tiny-2ag-v2")
        self.hp_env.setToolTip("Gymnasium env id to use for human play.")
        layout.addRow("Env", self.hp_env)

        self.hp_max_steps = QtWidgets.QSpinBox()
        self.hp_max_steps.setRange(1, 100000)
        self.hp_max_steps.setValue(500)
        self.hp_max_steps.setToolTip("Max steps per episode for human play.")
        layout.addRow("Max Steps", self.hp_max_steps)

        self.hp_display_info = QtWidgets.QCheckBox()
        self.hp_display_info.setChecked(True)
        self.hp_display_info.setToolTip("Print per-step agent info to the console.")
        layout.addRow("Display Info", self.hp_display_info)

        self.hp_headless = QtWidgets.QCheckBox()
        self.hp_headless.setChecked(False)
        self.hp_headless.setToolTip("Run without a window using PYGLET_HEADLESS=1.")
        layout.addRow("Headless (PYGLET_HEADLESS=1)", self.hp_headless)

        self.hp_env_config = QtWidgets.QLineEdit("")
        self.hp_env_config.setToolTip("Path to a saved env config JSON for human play.")
        hp_env_config_row = QtWidgets.QWidget()
        hp_env_config_layout = QtWidgets.QHBoxLayout(hp_env_config_row)
        hp_env_config_layout.setContentsMargins(0, 0, 0, 0)
        hp_env_config_layout.addWidget(self.hp_env_config)
        hp_browse = QtWidgets.QPushButton("Browse")
        hp_browse.clicked.connect(lambda: self._browse_env_config(self.hp_env_config))
        hp_browse.setToolTip("Select an env config JSON from disk.")
        hp_env_config_layout.addWidget(hp_browse)
        layout.addRow("Env Config (JSON)", hp_env_config_row)

        hp_selector = self._build_env_selector()
        layout.addRow("Saved Envs", hp_selector)

        run_button = QtWidgets.QPushButton("Launch Human Play")
        run_button.clicked.connect(self._run_human_play)
        layout.addRow(run_button)
        self.run_buttons.append(run_button)

        return self._wrap_scroll(tab)

    def _build_training_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        run_group = QtWidgets.QGroupBox("Run Settings")
        run_form = QtWidgets.QFormLayout(run_group)

        self.tr_env_resolved = QtWidgets.QLineEdit("")
        self.tr_env_resolved.setReadOnly(True)
        self.tr_env_resolved.setPlaceholderText("Resolved from selected JSON")
        self.tr_env_resolved.setToolTip("Legacy SEAC env id resolved from the generated env JSON.")
        run_form.addRow("Resolved Env", self.tr_env_resolved)

        self.tr_env_config = QtWidgets.QLineEdit("")
        self.tr_env_config.setToolTip(
            "Path to a generated env config JSON. When set, training uses this config instead of Env."
        )
        tr_env_config_row = QtWidgets.QWidget()
        tr_env_config_layout = QtWidgets.QHBoxLayout(tr_env_config_row)
        tr_env_config_layout.setContentsMargins(0, 0, 0, 0)
        tr_env_config_layout.addWidget(self.tr_env_config)
        tr_browse = QtWidgets.QPushButton("Browse")
        tr_browse.clicked.connect(lambda: self._browse_env_config(self.tr_env_config))
        tr_browse.setToolTip("Select an env config JSON from disk.")
        tr_env_config_layout.addWidget(tr_browse)
        run_form.addRow("Env Config (JSON)", tr_env_config_row)

        tr_selector = self._build_env_selector()
        run_form.addRow("Saved Envs", tr_selector)

        self.tr_named_configs = QtWidgets.QLineEdit("")
        self.tr_named_configs.setPlaceholderText("e.g. rware1")
        self.tr_named_configs.setToolTip(
            "Optional Sacred named configs from seac/seac/configs (space/comma separated)."
        )
        run_form.addRow("Named Configs", self.tr_named_configs)

        self.tr_seed = QtWidgets.QSpinBox()
        self.tr_seed.setRange(0, 10_000_000)
        self.tr_seed.setValue(1)
        self.tr_seed.setToolTip("Random seed for envs and action spaces.")
        run_form.addRow("Seed", self.tr_seed)

        self.tr_time_limit = QtWidgets.QSpinBox()
        self.tr_time_limit.setRange(1, 100000)
        self.tr_time_limit.setValue(500)
        self.tr_time_limit.setToolTip("Time limit wrapper steps per episode.")
        run_form.addRow("Time Limit", self.tr_time_limit)

        self.tr_num_env_steps = QtWidgets.QLineEdit("40000000")
        self.tr_num_env_steps.setToolTip("Total number of environment steps to train.")
        run_form.addRow("Num Env Steps", self.tr_num_env_steps)

        layout.addWidget(run_group)

        algo_group = QtWidgets.QGroupBox("Algorithm Settings")
        algo_form = QtWidgets.QFormLayout(algo_group)

        self.tr_num_steps = QtWidgets.QSpinBox()
        self.tr_num_steps.setRange(1, 1000)
        self.tr_num_steps.setValue(5)
        self.tr_num_steps.setToolTip("Number of steps per update for the algorithm.")
        algo_form.addRow("Num Steps", self.tr_num_steps)

        self.tr_num_processes = QtWidgets.QSpinBox()
        self.tr_num_processes.setRange(1, 256)
        self.tr_num_processes.setValue(4)
        self.tr_num_processes.setToolTip("Parallel env count for training.")
        algo_form.addRow("Num Processes", self.tr_num_processes)

        self.tr_lr = QtWidgets.QDoubleSpinBox()
        self.tr_lr.setDecimals(6)
        self.tr_lr.setRange(1e-6, 1.0)
        self.tr_lr.setSingleStep(1e-4)
        self.tr_lr.setValue(3e-4)
        self.tr_lr.setToolTip("Optimizer learning rate.")
        algo_form.addRow("Learning Rate", self.tr_lr)

        self.tr_gamma = QtWidgets.QDoubleSpinBox()
        self.tr_gamma.setDecimals(6)
        self.tr_gamma.setRange(0.0, 1.0)
        self.tr_gamma.setSingleStep(0.001)
        self.tr_gamma.setValue(0.99)
        self.tr_gamma.setToolTip("Discount factor.")
        algo_form.addRow("Gamma", self.tr_gamma)

        self.tr_use_gae = QtWidgets.QCheckBox()
        self.tr_use_gae.setChecked(False)
        self.tr_use_gae.setToolTip("Enable generalized advantage estimation.")
        algo_form.addRow("Use GAE", self.tr_use_gae)

        self.tr_gae_lambda = QtWidgets.QDoubleSpinBox()
        self.tr_gae_lambda.setDecimals(6)
        self.tr_gae_lambda.setRange(0.0, 1.0)
        self.tr_gae_lambda.setSingleStep(0.01)
        self.tr_gae_lambda.setValue(0.95)
        self.tr_gae_lambda.setToolTip("GAE lambda parameter.")
        algo_form.addRow("GAE Lambda", self.tr_gae_lambda)

        self.tr_use_linear_lr_decay = QtWidgets.QCheckBox()
        self.tr_use_linear_lr_decay.setChecked(False)
        self.tr_use_linear_lr_decay.setToolTip(
            "Linearly decay the learning rate during training."
        )
        algo_form.addRow("Linear LR Decay", self.tr_use_linear_lr_decay)

        self.tr_recurrent_policy = QtWidgets.QCheckBox()
        self.tr_recurrent_policy.setChecked(False)
        self.tr_recurrent_policy.setToolTip(
            "Enable recurrent policy (GRU) for temporal memory."
        )
        algo_form.addRow("Recurrent Policy", self.tr_recurrent_policy)

        self.tr_entropy_coef = QtWidgets.QDoubleSpinBox()
        self.tr_entropy_coef.setDecimals(6)
        self.tr_entropy_coef.setRange(0.0, 10.0)
        self.tr_entropy_coef.setSingleStep(0.001)
        self.tr_entropy_coef.setValue(0.01)
        self.tr_entropy_coef.setToolTip("Entropy bonus coefficient.")
        algo_form.addRow("Entropy Coef", self.tr_entropy_coef)

        self.tr_value_loss_coef = QtWidgets.QDoubleSpinBox()
        self.tr_value_loss_coef.setDecimals(6)
        self.tr_value_loss_coef.setRange(0.0, 10.0)
        self.tr_value_loss_coef.setSingleStep(0.1)
        self.tr_value_loss_coef.setValue(0.5)
        self.tr_value_loss_coef.setToolTip("Value loss coefficient.")
        algo_form.addRow("Value Loss Coef", self.tr_value_loss_coef)

        self.tr_seac_coef = QtWidgets.QDoubleSpinBox()
        self.tr_seac_coef.setDecimals(6)
        self.tr_seac_coef.setRange(0.0, 10.0)
        self.tr_seac_coef.setSingleStep(0.1)
        self.tr_seac_coef.setValue(1.0)
        self.tr_seac_coef.setToolTip("SEAC coefficient for cross-agent loss.")
        algo_form.addRow("SEAC Coef", self.tr_seac_coef)

        self.tr_device = QtWidgets.QComboBox()
        self.tr_device.addItems(["cuda", "cpu"])
        self.tr_device.setCurrentText("cuda")
        self.tr_device.setToolTip("Torch device for training.")
        algo_form.addRow("Device", self.tr_device)

        layout.addWidget(algo_group)

        log_group = QtWidgets.QGroupBox("Logging & Evaluation")
        log_form = QtWidgets.QFormLayout(log_group)

        self.tr_log_interval = QtWidgets.QSpinBox()
        self.tr_log_interval.setRange(1, 1_000_000_000)
        self.tr_log_interval.setValue(2000)
        self.tr_log_interval.setToolTip("How often to log training stats (updates).")
        log_form.addRow("Log Interval", self.tr_log_interval)

        self.tr_save_interval = QtWidgets.QSpinBox()
        self.tr_save_interval.setRange(0, 1_000_000_000)
        self.tr_save_interval.setValue(1_000_000)
        self.tr_save_interval.setToolTip("Save checkpoint every N updates (0 disables).")
        log_form.addRow("Save Interval", self.tr_save_interval)

        self.tr_eval_interval = QtWidgets.QSpinBox()
        self.tr_eval_interval.setRange(0, 1_000_000_000)
        self.tr_eval_interval.setValue(1_000_000)
        self.tr_eval_interval.setToolTip("Evaluate every N updates (0 disables).")
        log_form.addRow("Eval Interval", self.tr_eval_interval)

        self.tr_episodes_per_eval = QtWidgets.QSpinBox()
        self.tr_episodes_per_eval.setRange(1, 1000)
        self.tr_episodes_per_eval.setValue(8)
        self.tr_episodes_per_eval.setToolTip("Episodes per evaluation phase.")
        log_form.addRow("Episodes Per Eval", self.tr_episodes_per_eval)

        self.tr_save_dir = QtWidgets.QLineEdit("./results/trained_models/{id}")
        self.tr_save_dir.setToolTip("Directory to save trained models.")
        log_form.addRow("Save Dir", self.tr_save_dir)

        self.tr_eval_dir = QtWidgets.QLineEdit("./results/video/{id}")
        self.tr_eval_dir.setToolTip("Directory for evaluation videos.")
        log_form.addRow("Eval Dir", self.tr_eval_dir)

        self.tr_loss_dir = QtWidgets.QLineEdit("./results/loss/{id}")
        self.tr_loss_dir.setToolTip("Directory for TensorBoard loss logs (blank disables).")
        log_form.addRow("Loss Dir", self.tr_loss_dir)

        layout.addWidget(log_group)

        run_button = QtWidgets.QPushButton("Start Training")
        run_button.clicked.connect(self._run_training)
        layout.addWidget(run_button)
        self.run_buttons.append(run_button)

        layout.addStretch(1)

        return self._wrap_scroll(tab)

    def _build_evaluation_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)

        self.ev_env_resolved = QtWidgets.QLineEdit("")
        self.ev_env_resolved.setReadOnly(True)
        self.ev_env_resolved.setPlaceholderText("Resolved from selected JSON")
        self.ev_env_resolved.setToolTip("Legacy SEAC env id resolved from the generated env JSON.")
        layout.addRow("Resolved Env", self.ev_env_resolved)

        self.ev_env_config = QtWidgets.QLineEdit("")
        self.ev_env_config.setToolTip(
            "Path to a generated env config JSON. When set, evaluation uses this config instead of Env."
        )
        ev_env_config_row = QtWidgets.QWidget()
        ev_env_config_layout = QtWidgets.QHBoxLayout(ev_env_config_row)
        ev_env_config_layout.setContentsMargins(0, 0, 0, 0)
        ev_env_config_layout.addWidget(self.ev_env_config)
        ev_browse = QtWidgets.QPushButton("Browse")
        ev_browse.clicked.connect(lambda: self._browse_env_config(self.ev_env_config))
        ev_browse.setToolTip("Select an env config JSON from disk.")
        ev_env_config_layout.addWidget(ev_browse)
        layout.addRow("Env Config (JSON)", ev_env_config_row)

        ev_selector = self._build_env_selector()
        layout.addRow("Saved Envs", ev_selector)

        self.ev_path = QtWidgets.QLineEdit("pretrained/rware-small-4ag")
        self.ev_path.setToolTip("Path to trained model weights.")
        layout.addRow("Models Path", self.ev_path)

        self.ev_time_limit = QtWidgets.QSpinBox()
        self.ev_time_limit.setRange(1, 100000)
        self.ev_time_limit.setValue(500)
        self.ev_time_limit.setToolTip("Time limit wrapper steps per episode.")
        layout.addRow("Time Limit", self.ev_time_limit)

        self.ev_episodes = QtWidgets.QSpinBox()
        self.ev_episodes.setRange(1, 1000)
        self.ev_episodes.setValue(5)
        self.ev_episodes.setToolTip("Number of evaluation episodes.")
        layout.addRow("Episodes", self.ev_episodes)

        self.ev_record_video = QtWidgets.QCheckBox("Record video for each episode")
        self.ev_record_video.setToolTip("Save an MP4 recording for each evaluation episode.")
        layout.addRow("Video", self.ev_record_video)

        self.ev_export_csv = QtWidgets.QCheckBox("Export CSV report")
        self.ev_export_csv.setToolTip("Write a CSV summary for each evaluation episode.")
        layout.addRow("CSV Report", self.ev_export_csv)

        self.ev_output_dir = QtWidgets.QLineEdit("./results/evaluation")
        self.ev_output_dir.setToolTip("Directory where evaluation videos and CSV reports are saved.")
        layout.addRow("Output Dir", self.ev_output_dir)

        run_button = QtWidgets.QPushButton("Run Evaluation")
        run_button.clicked.connect(self._run_evaluation)
        layout.addRow(run_button)
        self.run_buttons.append(run_button)

        return self._wrap_scroll(tab)

    def _build_env_generator_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, stretch=1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        splitter.addWidget(left)
        left.setMinimumWidth(520)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([620, 560])

        params_group = QtWidgets.QGroupBox("Environment Parameters")
        params_form = QtWidgets.QFormLayout(params_group)
        params_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.gen_env_id = QtWidgets.QLineEdit("rware-tiny-2ag-v2")
        self.gen_env_id.setToolTip("Gymnasium env id to instantiate (e.g., rware-tiny-2ag-v2).")
        params_form.addRow("Base Env ID", self.gen_env_id)

        self.gen_shelf_rows = QtWidgets.QSpinBox()
        self.gen_shelf_rows.setRange(1, 20)
        self.gen_shelf_rows.setValue(1)
        self.gen_shelf_rows.setToolTip("Number of shelf rows in the standard warehouse layout.")
        params_form.addRow("Shelf Rows", self.gen_shelf_rows)

        self.gen_shelf_cols = QtWidgets.QSpinBox()
        self.gen_shelf_cols.setRange(1, 30)
        self.gen_shelf_cols.setValue(3)
        self.gen_shelf_cols.setToolTip("Number of shelf columns in the standard warehouse layout.")
        params_form.addRow("Shelf Columns", self.gen_shelf_cols)

        self.gen_column_height = QtWidgets.QSpinBox()
        self.gen_column_height.setRange(1, 30)
        self.gen_column_height.setValue(8)
        self.gen_column_height.setToolTip("Height of each shelf column block.")
        params_form.addRow("Column Height", self.gen_column_height)

        self.gen_agents = QtWidgets.QSpinBox()
        self.gen_agents.setRange(1, 64)
        self.gen_agents.setValue(2)
        self.gen_agents.setToolTip("Number of agents spawned in the environment.")
        params_form.addRow("Agents", self.gen_agents)

        self.gen_msg_bits = QtWidgets.QSpinBox()
        self.gen_msg_bits.setRange(0, 32)
        self.gen_msg_bits.setValue(0)
        self.gen_msg_bits.setToolTip("Number of communication bits per agent.")
        params_form.addRow("Message Bits", self.gen_msg_bits)

        self.gen_sensor_range = QtWidgets.QSpinBox()
        self.gen_sensor_range.setRange(1, 10)
        self.gen_sensor_range.setValue(1)
        self.gen_sensor_range.setToolTip("Observation radius for each agent.")
        params_form.addRow("Sensor Range", self.gen_sensor_range)

        self.gen_request_queue = QtWidgets.QSpinBox()
        self.gen_request_queue.setRange(1, 200)
        self.gen_request_queue.setValue(2)
        self.gen_request_queue.setToolTip(
            "How many shelves are requested at once (ignored when Dedicated Requests is enabled)."
        )
        params_form.addRow("Request Queue Size", self.gen_request_queue)

        self.gen_max_inactivity = QtWidgets.QSpinBox()
        self.gen_max_inactivity.setRange(0, 100000)
        self.gen_max_inactivity.setValue(0)
        self.gen_max_inactivity.setToolTip("Max steps without a delivery before termination (0 = None).")
        params_form.addRow("Max Inactivity (0=None)", self.gen_max_inactivity)

        self.gen_max_steps = QtWidgets.QSpinBox()
        self.gen_max_steps.setRange(0, 100000)
        self.gen_max_steps.setValue(500)
        self.gen_max_steps.setToolTip("Max steps per episode (0 = None).")
        params_form.addRow("Max Steps (0=None)", self.gen_max_steps)

        self.gen_reward_type = QtWidgets.QComboBox()
        self.gen_reward_type.addItems(list(_REWARD_TYPE_MAP.keys()))
        self.gen_reward_type.setCurrentText("INDIVIDUAL")
        self.gen_reward_type.setToolTip("Reward mode: per-agent, global, or two-stage.")
        params_form.addRow("Reward Type", self.gen_reward_type)

        self.gen_obs_type = QtWidgets.QComboBox()
        self.gen_obs_type.addItems(list(_OBS_TYPE_MAP.keys()))
        self.gen_obs_type.setCurrentText("FLATTENED")
        self.gen_obs_type.setToolTip("Observation format returned by the environment.")
        params_form.addRow("Observation Type", self.gen_obs_type)

        self.gen_render_mode = QtWidgets.QComboBox()
        self.gen_render_mode.addItems(["", "human", "rgb_array"])
        self.gen_render_mode.setCurrentText("")
        self.gen_render_mode.setToolTip(
            "Optional env render_mode kwarg. Leave blank for None."
        )
        params_form.addRow("Render Mode", self.gen_render_mode)

        image_layers_widget = QtWidgets.QWidget()
        image_layers_layout = QtWidgets.QGridLayout(image_layers_widget)
        image_layers_layout.setContentsMargins(0, 0, 0, 0)
        image_layers_layout.setHorizontalSpacing(8)
        image_layers_layout.setVerticalSpacing(4)
        image_layers_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.gen_image_layer_checks = {}
        for idx, layer_name in enumerate(_IMAGE_LAYER_OPTIONS):
            checkbox = QtWidgets.QCheckBox(layer_name)
            checkbox.setChecked(layer_name in _DEFAULT_IMAGE_LAYERS)
            checkbox.setToolTip(f"Include {layer_name} in image-style observations.")
            self.gen_image_layer_checks[layer_name] = checkbox
            image_layers_layout.addWidget(checkbox, idx, 0)
        params_form.addRow(QtWidgets.QLabel("Image Obs Layers"))
        params_form.addRow(image_layers_widget)

        self.gen_image_obs_directional = QtWidgets.QCheckBox()
        self.gen_image_obs_directional.setChecked(True)
        self.gen_image_obs_directional.setToolTip(
            "Rotate image observations into each agent's local orientation."
        )
        params_form.addRow("Image Obs Directional", self.gen_image_obs_directional)

        self.gen_lane_observation = QtWidgets.QCheckBox()
        self.gen_lane_observation.setChecked(False)
        self.gen_lane_observation.setToolTip(
            "Include lane-direction feature maps in vector/image-dict observations."
        )
        params_form.addRow("Lane Observation", self.gen_lane_observation)

        self.gen_normalised = QtWidgets.QCheckBox()
        self.gen_normalised.setChecked(False)
        self.gen_normalised.setToolTip("Normalize absolute coordinates to [0, 1].")
        params_form.addRow("Normalised Coordinates", self.gen_normalised)

        self.gen_dedicated_requests = QtWidgets.QCheckBox()
        self.gen_dedicated_requests.setChecked(True)
        self.gen_dedicated_requests.setToolTip("Assign each agent a dedicated requested shelf.")
        params_form.addRow("Dedicated Requests", self.gen_dedicated_requests)

        self.gen_reward_delivery_weight = QtWidgets.QDoubleSpinBox()
        self.gen_reward_delivery_weight.setRange(-10.0, 10.0)
        self.gen_reward_delivery_weight.setSingleStep(0.1)
        self.gen_reward_delivery_weight.setValue(1.0)
        self.gen_reward_delivery_weight.setToolTip("Weight applied to delivery reward.")
        params_form.addRow("Delivery Weight", self.gen_reward_delivery_weight)

        left_layout.addWidget(params_group)

        config_group = QtWidgets.QGroupBox("Env Config")
        config_layout = QtWidgets.QVBoxLayout(config_group)
        self.env_config_path = QtWidgets.QLineEdit("")
        self.env_config_path.setReadOnly(True)
        self.env_config_path.setToolTip("Path to the last saved env config.")
        config_layout.addWidget(self.env_config_path)

        config_buttons = QtWidgets.QHBoxLayout()
        self.env_config_save_button = QtWidgets.QPushButton("Save Env Config")
        self.env_config_save_button.clicked.connect(self._save_env_config)
        self.env_config_save_button.setToolTip("Save all parameters + layout to a JSON config.")
        config_buttons.addWidget(self.env_config_save_button)
        self.env_config_copy_button = QtWidgets.QPushButton("Copy Path")
        self.env_config_copy_button.clicked.connect(self._copy_env_config_path)
        self.env_config_copy_button.setToolTip("Copy the env config file path to clipboard.")
        config_buttons.addWidget(self.env_config_copy_button)
        self.env_config_delete_button = QtWidgets.QPushButton("Delete Env Config")
        self.env_config_delete_button.clicked.connect(self._delete_env_config)
        self.env_config_delete_button.setToolTip("Delete the selected env config file.")
        config_buttons.addWidget(self.env_config_delete_button)
        self.env_config_refresh_button = QtWidgets.QPushButton("Refresh List")
        self.env_config_refresh_button.clicked.connect(self._refresh_env_config_combos)
        self.env_config_refresh_button.setToolTip("Refresh the saved env list.")
        config_buttons.addWidget(self.env_config_refresh_button)
        config_layout.addLayout(config_buttons)

        left_layout.addWidget(config_group)

        selector_group = QtWidgets.QGroupBox("Saved Envs")
        selector_layout = QtWidgets.QVBoxLayout(selector_group)
        selector_layout.addWidget(self._build_env_selector())
        left_layout.addWidget(selector_group)

        left_layout.addStretch(1)

        grid_controls = QtWidgets.QHBoxLayout()
        grid_controls.addWidget(QtWidgets.QLabel("Rows"))
        self.layout_rows = QtWidgets.QSpinBox()
        self.layout_rows.setRange(1, 50)
        self.layout_rows.setValue(7)
        self.layout_rows.setToolTip("Number of rows in the layout grid editor.")
        grid_controls.addWidget(self.layout_rows)

        grid_controls.addWidget(QtWidgets.QLabel("Cols"))
        self.layout_cols = QtWidgets.QSpinBox()
        self.layout_cols.setRange(1, 50)
        self.layout_cols.setValue(7)
        self.layout_cols.setToolTip("Number of columns in the layout grid editor.")
        grid_controls.addWidget(self.layout_cols)

        resize_button = QtWidgets.QPushButton("Resize Grid")
        resize_button.clicked.connect(self._resize_layout_grid)
        resize_button.setToolTip("Resize the layout grid to the selected rows/cols.")
        grid_controls.addWidget(resize_button)
        grid_controls.addStretch(1)
        right_layout.addLayout(grid_controls)

        brush_layout = QtWidgets.QHBoxLayout()
        brush_layout.addWidget(QtWidgets.QLabel("Brush"))
        self.brush_group = QtWidgets.QButtonGroup(self)
        self.brush_select = QtWidgets.QRadioButton("Select")
        self.brush_shelf = QtWidgets.QRadioButton("Shelf (X)")
        self.brush_corridor = QtWidgets.QRadioButton("Corridor (.)")
        self.brush_goal = QtWidgets.QRadioButton("Goal (g)")
        self.brush_obstacle = QtWidgets.QRadioButton("Obstacle (O)")
        self.brush_lane = QtWidgets.QRadioButton("Lane")
        self.brush_select.setToolTip("Select cells for inspection/edit only (no painting).")
        self.brush_shelf.setToolTip("Paint shelves (X).")
        self.brush_corridor.setToolTip("Paint corridors (.).")
        self.brush_goal.setToolTip("Paint goals (g).")
        self.brush_obstacle.setToolTip("Paint obstacles (O).")
        self.brush_lane.setToolTip("Paint lane directions using U/D/L/R toggles.")
        self.brush_group.addButton(self.brush_select)
        self.brush_group.addButton(self.brush_shelf)
        self.brush_group.addButton(self.brush_corridor)
        self.brush_group.addButton(self.brush_goal)
        self.brush_group.addButton(self.brush_obstacle)
        self.brush_group.addButton(self.brush_lane)
        self.brush_select.setChecked(True)
        brush_layout.addWidget(self.brush_select)
        brush_layout.addWidget(self.brush_shelf)
        brush_layout.addWidget(self.brush_corridor)
        brush_layout.addWidget(self.brush_goal)
        brush_layout.addWidget(self.brush_obstacle)
        brush_layout.addWidget(self.brush_lane)
        brush_layout.addWidget(QtWidgets.QLabel("Lane Dirs"))
        self.brush_lane_up = QtWidgets.QCheckBox("U")
        self.brush_lane_down = QtWidgets.QCheckBox("D")
        self.brush_lane_left = QtWidgets.QCheckBox("L")
        self.brush_lane_right = QtWidgets.QCheckBox("R")
        self.brush_lane_up.setChecked(True)
        self.brush_lane_down.setChecked(True)
        self.brush_lane_left.setChecked(True)
        self.brush_lane_right.setChecked(True)
        self.brush_lane_up.setToolTip("Enable UP direction for Lane brush.")
        self.brush_lane_down.setToolTip("Enable DOWN direction for Lane brush.")
        self.brush_lane_left.setToolTip("Enable LEFT direction for Lane brush.")
        self.brush_lane_right.setToolTip("Enable RIGHT direction for Lane brush.")
        brush_layout.addWidget(self.brush_lane_up)
        brush_layout.addWidget(self.brush_lane_down)
        brush_layout.addWidget(self.brush_lane_left)
        brush_layout.addWidget(self.brush_lane_right)
        brush_layout.addStretch(1)
        right_layout.addLayout(brush_layout)

        selected_layout = QtWidgets.QHBoxLayout()
        self.selected_cell_label = QtWidgets.QLabel("Selected Cell: none")
        selected_layout.addWidget(self.selected_cell_label)
        selected_layout.addWidget(QtWidgets.QLabel("Edit Dirs"))
        self.selected_dir_up = QtWidgets.QCheckBox("U")
        self.selected_dir_down = QtWidgets.QCheckBox("D")
        self.selected_dir_left = QtWidgets.QCheckBox("L")
        self.selected_dir_right = QtWidgets.QCheckBox("R")
        self.selected_dir_up.setToolTip("Enable UP movement for selected cell.")
        self.selected_dir_down.setToolTip("Enable DOWN movement for selected cell.")
        self.selected_dir_left.setToolTip("Enable LEFT movement for selected cell.")
        self.selected_dir_right.setToolTip("Enable RIGHT movement for selected cell.")
        selected_layout.addWidget(self.selected_dir_up)
        selected_layout.addWidget(self.selected_dir_down)
        selected_layout.addWidget(self.selected_dir_left)
        selected_layout.addWidget(self.selected_dir_right)
        selected_layout.addStretch(1)
        right_layout.addLayout(selected_layout)

        action_layout = QtWidgets.QHBoxLayout()
        fill_shelves = QtWidgets.QPushButton("Fill Shelves")
        fill_shelves.clicked.connect(lambda: self._fill_layout("x"))
        fill_shelves.setToolTip("Fill the entire grid with shelves.")
        action_layout.addWidget(fill_shelves)

        fill_corridors = QtWidgets.QPushButton("Fill Corridors")
        fill_corridors.clicked.connect(lambda: self._fill_layout("."))
        fill_corridors.setToolTip("Fill the entire grid with corridors.")
        action_layout.addWidget(fill_corridors)

        clear_goals = QtWidgets.QPushButton("Clear Goals")
        clear_goals.clicked.connect(self._clear_goals)
        clear_goals.setToolTip("Replace all goals with shelves.")
        action_layout.addWidget(clear_goals)
        action_layout.addStretch(1)
        right_layout.addLayout(action_layout)

        self.layout_table = LayoutTable(
            self.layout_rows.value(), self.layout_cols.value(), self._apply_layout_brush
        )
        self.layout_table.setToolTip(
            "Click cells to inspect or paint, depending on selected brush mode."
        )
        right_layout.addWidget(self.layout_table, stretch=1)

        self.layout_status = QtWidgets.QLabel("")
        right_layout.addWidget(self.layout_status)

        self.layout_grid = []
        self.cell_direction_constraints = {}
        self._selected_cell = None
        self._selected_dir_sync = False
        self._layout_param_syncing = False
        self._env_config_loading = False
        self._init_layout_grid(self.layout_rows.value(), self.layout_cols.value(), fill="x")
        self._update_layout_text_from_grid()

        self.brush_select.toggled.connect(lambda checked: checked and self._set_layout_brush("select"))
        self.brush_shelf.toggled.connect(lambda checked: checked and self._set_layout_brush("x"))
        self.brush_corridor.toggled.connect(lambda checked: checked and self._set_layout_brush("."))
        self.brush_goal.toggled.connect(lambda checked: checked and self._set_layout_brush("g"))
        self.brush_obstacle.toggled.connect(lambda checked: checked and self._set_layout_brush("o"))
        self.brush_lane.toggled.connect(lambda checked: checked and self._set_layout_brush("lane"))
        self.layout_table.cellPressed.connect(self._on_layout_cell_selected)
        self.selected_dir_up.toggled.connect(self._on_selected_cell_dirs_changed)
        self.selected_dir_down.toggled.connect(self._on_selected_cell_dirs_changed)
        self.selected_dir_left.toggled.connect(self._on_selected_cell_dirs_changed)
        self.selected_dir_right.toggled.connect(self._on_selected_cell_dirs_changed)
        self._set_selected_layout_cell(None, None)
        self._refresh_env_config_combos()

        return self._wrap_scroll(tab)

    def _wrap_scroll(self, widget):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def _init_layout_grid(self, rows, cols, fill="x"):
        self.cell_direction_constraints = {}
        self.layout_grid = [[fill for _ in range(cols)] for _ in range(rows)]
        self.layout_table.setRowCount(rows)
        self.layout_table.setColumnCount(cols)
        for row in range(rows):
            for col in range(cols):
                self._set_layout_cell(row, col, fill)
        self._set_selected_layout_cell(None, None)

    def _set_layout_cell(self, row, col, char):
        char = char.lower()
        self.layout_grid[row][col] = char
        self.cell_direction_constraints.pop((row, col), None)
        self._render_layout_cell(row, col)

    def _render_layout_cell(self, row, col):
        item = self.layout_table.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.layout_table.setItem(row, col, item)
        char = self.layout_grid[row][col]
        custom_dirs = self.cell_direction_constraints.get((row, col))
        if custom_dirs:
            dir_text = "".join(_LANE_SYMBOL_BY_DIR[d] for d in custom_dirs)
            if char == "g":
                item.setText(f"G{dir_text}")
            elif char == ".":
                item.setText(dir_text)
            else:
                item.setText(f"{char.upper()}{dir_text}")
            item.setBackground(_LAYOUT_COLORS.get("|", QtGui.QColor("#cfe8ff")))
        else:
            item.setText("." if char == "." else char.upper())
            item.setBackground(_LAYOUT_COLORS.get(char, QtGui.QColor("#ffffff")))

    def _effective_cell_dirs(self, row, col):
        char = self.layout_grid[row][col]
        if char == "o":
            return tuple()
        custom = self.cell_direction_constraints.get((row, col))
        if custom:
            return custom
        token_dirs = _LAYOUT_TOKEN_TO_DIRS.get(char)
        if token_dirs:
            return token_dirs
        return _LANE_DIR_ORDER

    def _set_selected_layout_cell(self, row, col):
        if (
            row is None
            or col is None
            or row < 0
            or col < 0
            or row >= len(self.layout_grid)
            or col >= len(self.layout_grid[0])
        ):
            self._selected_cell = None
            self.selected_cell_label.setText("Selected Cell: none")
            self._selected_dir_sync = True
            self.selected_dir_up.setChecked(False)
            self.selected_dir_down.setChecked(False)
            self.selected_dir_left.setChecked(False)
            self.selected_dir_right.setChecked(False)
            self.selected_dir_up.setEnabled(False)
            self.selected_dir_down.setEnabled(False)
            self.selected_dir_left.setEnabled(False)
            self.selected_dir_right.setEnabled(False)
            self._selected_dir_sync = False
            return

        self._selected_cell = (row, col)
        char = self.layout_grid[row][col]
        dirs = self._effective_cell_dirs(row, col)
        dirs_text = "".join(_LANE_SYMBOL_BY_DIR[d] for d in dirs) if dirs else "none"
        self.selected_cell_label.setText(
            f"Selected Cell: ({row},{col}) {char.upper()} | Dirs: {dirs_text}"
        )
        is_obstacle = char == "o"
        self._selected_dir_sync = True
        self.selected_dir_up.setChecked("UP" in dirs)
        self.selected_dir_down.setChecked("DOWN" in dirs)
        self.selected_dir_left.setChecked("LEFT" in dirs)
        self.selected_dir_right.setChecked("RIGHT" in dirs)
        self.selected_dir_up.setEnabled(not is_obstacle)
        self.selected_dir_down.setEnabled(not is_obstacle)
        self.selected_dir_left.setEnabled(not is_obstacle)
        self.selected_dir_right.setEnabled(not is_obstacle)
        self._selected_dir_sync = False

    def _on_layout_cell_selected(self, row, col):
        self._set_selected_layout_cell(row, col)

    def _selected_cell_dirs_from_ui(self):
        dirs = []
        if self.selected_dir_up.isChecked():
            dirs.append("UP")
        if self.selected_dir_down.isChecked():
            dirs.append("DOWN")
        if self.selected_dir_left.isChecked():
            dirs.append("LEFT")
        if self.selected_dir_right.isChecked():
            dirs.append("RIGHT")
        return tuple(dirs)

    def _on_selected_cell_dirs_changed(self, _checked):
        if self._selected_dir_sync or self._selected_cell is None:
            return
        row, col = self._selected_cell
        dirs = self._selected_cell_dirs_from_ui()
        if not dirs:
            self._append_log("[launcher] selected cell must allow at least one direction")
            self._set_selected_layout_cell(row, col)
            return
        if self._apply_lane_dirs_to_cell(row, col, dirs, log_errors=False):
            self._update_layout_text_from_grid()
            self._set_selected_layout_cell(row, col)

    def _lane_dirs_from_ui(self):
        directions = []
        if self.brush_lane_up.isChecked():
            directions.append("UP")
        if self.brush_lane_down.isChecked():
            directions.append("DOWN")
        if self.brush_lane_left.isChecked():
            directions.append("LEFT")
        if self.brush_lane_right.isChecked():
            directions.append("RIGHT")
        return tuple(directions)

    def _normalize_lane_dirs(self, dirs):
        if dirs is None:
            values = []
        elif isinstance(dirs, str):
            token = dirs.strip()
            if not token:
                values = []
            elif token == "|":
                values = ["UP", "DOWN"]
            elif token == "-":
                values = ["LEFT", "RIGHT"]
            elif any(sep in token for sep in (",", ";", "/", " ")):
                cleaned = token
                for sep in (";", "/", " "):
                    cleaned = cleaned.replace(sep, ",")
                values = [part for part in cleaned.split(",") if part]
            elif len(token) > 1 and all(ch in "^v<>" for ch in token):
                values = list(token)
            else:
                values = [token]
        elif isinstance(dirs, (list, tuple, set)):
            values = list(dirs)
        else:
            values = [dirs]

        normalized = []
        seen = set()
        for value in values:
            token = str(value).strip().upper()
            if token in ("^", "U"):
                token = "UP"
            elif token in ("V", "D"):
                token = "DOWN"
            elif token in ("<", "L"):
                token = "LEFT"
            elif token in (">", "R"):
                token = "RIGHT"
            if token not in _LANE_DIR_ORDER or token in seen:
                continue
            normalized.append(token)
            seen.add(token)
        normalized.sort(key=lambda d: _LANE_DIR_ORDER.index(d))
        return tuple(normalized)

    def _lane_token_for_dirs(self, dirs):
        normalized = self._normalize_lane_dirs(dirs)
        return _LANE_TOKEN_BY_DIRS.get(normalized)

    def _apply_lane_dirs_to_cell(self, row, col, dirs, log_errors=True):
        normalized = self._normalize_lane_dirs(dirs)
        if not normalized:
            if log_errors:
                self._append_log("[launcher] select at least one lane direction")
            return False
        current = self.layout_grid[row][col]
        if current == "o":
            if log_errors:
                self._append_log("[launcher] cannot paint lane direction on obstacle cell")
            return False

        if len(normalized) == len(_LANE_DIR_ORDER):
            self.cell_direction_constraints.pop((row, col), None)
            if current in _LAYOUT_TOKEN_TO_DIRS:
                self.layout_grid[row][col] = "."
            self._render_layout_cell(row, col)
            return True

        if current in ("x", "g"):
            self.cell_direction_constraints[(row, col)] = normalized
            self._render_layout_cell(row, col)
            return True

        token = self._lane_token_for_dirs(normalized)
        self.cell_direction_constraints.pop((row, col), None)
        if token is not None:
            self.layout_grid[row][col] = token
            self._render_layout_cell(row, col)
            return True

        self.layout_grid[row][col] = "."
        self.cell_direction_constraints[(row, col)] = normalized
        self._render_layout_cell(row, col)
        return True

    def _apply_layout_brush(self, row, col, brush):
        if brush == "select":
            self._set_selected_layout_cell(row, col)
            return
        if brush == "lane":
            if self._apply_lane_dirs_to_cell(row, col, self._lane_dirs_from_ui()):
                self._update_layout_text_from_grid()
            self._set_selected_layout_cell(row, col)
            return
        self._set_layout_cell(row, col, brush)
        self._update_layout_text_from_grid()
        self._set_selected_layout_cell(row, col)

    def _set_layout_brush(self, brush):
        self.layout_table.set_brush(brush)

    def _grid_to_layout_text(self):
        return "\n".join("".join(row) for row in self.layout_grid)

    def _update_layout_text_from_grid(self):
        self._update_layout_status()
        self._update_layout_params_from_grid()

    def _validate_layout_text(self, layout_text):
        cleaned = [line.strip().replace(" ", "") for line in layout_text.strip().splitlines() if line.strip()]
        if not cleaned:
            return None, "Layout is empty."
        width = len(cleaned[0])
        for line in cleaned:
            if len(line) != width:
                return None, "Layout must be rectangular."
        goals = 0
        grid = []
        for line in cleaned:
            row = []
            for char in line:
                char = char.lower()
                if char not in ("x", "g", ".", "o", "|", "-", "^", "v", "<", ">"):
                    return None, "Layout can only use X, G, O, ., |, -, ^, v, <, and >."
                if char == "g":
                    goals += 1
                row.append(char)
            grid.append(row)
        if goals == 0:
            return grid, "Layout needs at least one goal."
        return grid, ""

    def _apply_layout_from_string(self, layout_text):
        grid, error = self._validate_layout_text(layout_text)
        if grid is None:
            return
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        self.layout_rows.setValue(rows)
        self.layout_cols.setValue(cols)
        self.layout_table.setRowCount(rows)
        self.layout_table.setColumnCount(cols)
        self.cell_direction_constraints = {}
        self.layout_grid = grid
        for row in range(rows):
            for col in range(cols):
                self._set_layout_cell(row, col, grid[row][col])
        self._update_layout_text_from_grid()
        self._set_selected_layout_cell(None, None)

    def _update_layout_params_from_grid(self):
        if self._layout_param_syncing:
            return
        grid, error = self._validate_layout_text(self._grid_to_layout_text())
        if grid is None or error:
            return
        inferred = self._infer_layout_params(grid)
        if not inferred:
            return
        self._layout_param_syncing = True
        self._set_spin_value(self.gen_shelf_rows, inferred["shelf_rows"])
        self._set_spin_value(self.gen_shelf_cols, inferred["shelf_columns"])
        self._set_spin_value(self.gen_column_height, inferred["column_height"])
        self._layout_param_syncing = False

    def _infer_layout_params(self, grid):
        if not grid:
            return None
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        shelf_rows_idx = [r for r in range(rows) if "x" in grid[r]]
        if not shelf_rows_idx:
            return None
        row_groups = []
        current = [shelf_rows_idx[0]]
        for idx in shelf_rows_idx[1:]:
            if idx == current[-1] + 1:
                current.append(idx)
            else:
                row_groups.append(current)
                current = [idx]
        row_groups.append(current)
        shelf_rows = len(row_groups)
        lengths = [len(group) for group in row_groups]
        counts = {}
        for length in lengths:
            counts[length] = counts.get(length, 0) + 1
        column_height = max(counts.items(), key=lambda item: (item[1], item[0]))[0]

        shelf_cols_idx = []
        for c in range(cols):
            if any(grid[r][c] == "x" for r in range(rows)):
                shelf_cols_idx.append(c)
        if not shelf_cols_idx:
            return None
        col_groups = []
        current = [shelf_cols_idx[0]]
        for idx in shelf_cols_idx[1:]:
            if idx == current[-1] + 1:
                current.append(idx)
            else:
                col_groups.append(current)
                current = [idx]
        col_groups.append(current)
        shelf_columns = len(col_groups)
        return {
            "shelf_rows": shelf_rows,
            "shelf_columns": shelf_columns,
            "column_height": column_height,
        }

    def _set_spin_value(self, spinbox, value):
        value = max(spinbox.minimum(), min(spinbox.maximum(), value))
        spinbox.setValue(value)

    def _normalize_spawn_direction(self, value):
        if value is None:
            return None
        token = str(value).strip().upper()
        if token not in _SPAWN_DIR_INT_BY_TOKEN:
            return None
        direction = _SPAWN_DIR_INT_BY_TOKEN[token]
        return _SPAWN_DIR_LABEL_BY_INT[direction]

    def _parse_fixed_spawns_text(self, text, expected_count):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None, "Fixed spawns are enabled but no spawn entries were provided."

        spawns = []
        for idx, line in enumerate(lines):
            parts = [part for part in re.split(r"[,\s;]+", line) if part]
            if len(parts) != 3:
                return None, (
                    f"Invalid spawn line #{idx + 1}: '{line}'. "
                    "Use format x,y,dir."
                )
            try:
                x = int(parts[0])
                y = int(parts[1])
            except ValueError:
                return None, f"Invalid coordinates in spawn line #{idx + 1}: '{line}'."
            if x < 0 or y < 0:
                return None, f"Spawn line #{idx + 1} has negative coordinates."

            direction_label = self._normalize_spawn_direction(parts[2])
            if direction_label is None:
                return None, (
                    f"Invalid direction in spawn line #{idx + 1}: '{parts[2]}'. "
                    "Use UP/DOWN/LEFT/RIGHT or 0-3."
                )
            spawns.append({"x": x, "y": y, "dir": direction_label})

        if len(spawns) != expected_count:
            return None, (
                f"Fixed spawns count ({len(spawns)}) must equal agent count ({expected_count})."
            )
        return spawns, ""

    def _parse_fixed_nav_goals_text(self, text, expected_count):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None, "Fixed nav goals are enabled but no goal entries were provided."

        nav_goals = []
        for idx, line in enumerate(lines):
            parts = [part for part in re.split(r"[,\s;]+", line) if part]
            if len(parts) != 2:
                return None, (
                    f"Invalid nav goal line #{idx + 1}: '{line}'. "
                    "Use format x,y."
                )
            try:
                x = int(parts[0])
                y = int(parts[1])
            except ValueError:
                return None, f"Invalid coordinates in nav goal line #{idx + 1}: '{line}'."
            if x < 0 or y < 0:
                return None, f"Nav goal line #{idx + 1} has negative coordinates."
            nav_goals.append([x, y])

        if len(nav_goals) != expected_count:
            return None, (
                f"Fixed nav goals count ({len(nav_goals)}) must equal agent count ({expected_count})."
            )
        return nav_goals, ""

    def _format_fixed_spawns_text(self, spawns):
        if not isinstance(spawns, (list, tuple)):
            return ""
        lines = []
        for entry in spawns:
            if not isinstance(entry, dict):
                continue
            if "x" not in entry or "y" not in entry or "dir" not in entry:
                continue
            direction_label = self._normalize_spawn_direction(entry["dir"])
            if direction_label is None:
                continue
            lines.append(f"{int(entry['x'])},{int(entry['y'])},{direction_label}")
        return "\n".join(lines)

    def _format_fixed_nav_goals_text(self, nav_goals):
        if not isinstance(nav_goals, (list, tuple)):
            return ""
        lines = []
        for entry in nav_goals:
            if isinstance(entry, dict):
                if "x" not in entry or "y" not in entry:
                    continue
                x = int(entry["x"])
                y = int(entry["y"])
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                x = int(entry[0])
                y = int(entry[1])
            else:
                continue
            lines.append(f"{x},{y}")
        return "\n".join(lines)

    def _normalize_image_layer_name(self, value):
        if value is None:
            return None
        if isinstance(value, int):
            if 0 <= value < len(_IMAGE_LAYER_OPTIONS):
                return _IMAGE_LAYER_OPTIONS[value]
            return None
        name = str(value).strip()
        if "." in name:
            name = name.split(".")[-1]
        name = name.upper()
        if name in _IMAGE_LAYER_OPTIONS:
            return name
        return None

    def _set_image_layer_checks(self, layers):
        selected = []
        if isinstance(layers, (list, tuple)):
            for layer in layers:
                normalized = self._normalize_image_layer_name(layer)
                if normalized and normalized not in selected:
                    selected.append(normalized)
        if not selected:
            selected = list(_DEFAULT_IMAGE_LAYERS)
        for layer_name, checkbox in self.gen_image_layer_checks.items():
            checkbox.setChecked(layer_name in selected)

    def _selected_image_layers(self):
        selected = [
            layer_name
            for layer_name in _IMAGE_LAYER_OPTIONS
            if self.gen_image_layer_checks[layer_name].isChecked()
        ]
        return selected

    def _update_layout_status(self, from_text=False):
        if from_text:
            grid, error = self._validate_layout_text(self._grid_to_layout_text())
            if grid is None:
                self.layout_status.setText(error)
                return
            goals = sum(cell == "g" for row in grid for cell in row)
            self.layout_status.setText(f"Goals: {goals} | Grid: {len(grid)}x{len(grid[0])}")
            if error:
                self.layout_status.setText(f"{self.layout_status.text()} | {error}")
            return
        goals = sum(cell == "g" for row in self.layout_grid for cell in row)
        rows = len(self.layout_grid)
        cols = len(self.layout_grid[0]) if rows else 0
        status = f"Goals: {goals} | Grid: {rows}x{cols}"
        if goals == 0:
            status += " | needs >=1 goal"
        self.layout_status.setText(status)

    def _resize_layout_grid(self):
        rows = self.layout_rows.value()
        cols = self.layout_cols.value()
        prev_selected = self._selected_cell
        new_grid = [["x" for _ in range(cols)] for _ in range(rows)]
        old_constraints = dict(self.cell_direction_constraints)
        for r in range(min(rows, len(self.layout_grid))):
            for c in range(min(cols, len(self.layout_grid[0]))):
                new_grid[r][c] = self.layout_grid[r][c]
        self.cell_direction_constraints = {}
        self.layout_grid = new_grid
        self.layout_table.setRowCount(rows)
        self.layout_table.setColumnCount(cols)
        for row in range(rows):
            for col in range(cols):
                self._set_layout_cell(row, col, self.layout_grid[row][col])
                dirs = old_constraints.get((row, col))
                if dirs and self.layout_grid[row][col] != "o":
                    self.cell_direction_constraints[(row, col)] = dirs
                    self._render_layout_cell(row, col)
        self._update_layout_text_from_grid()
        if prev_selected is not None:
            self._set_selected_layout_cell(prev_selected[0], prev_selected[1])
        else:
            self._set_selected_layout_cell(None, None)

    def _fill_layout(self, char):
        for row in range(len(self.layout_grid)):
            for col in range(len(self.layout_grid[0])):
                self._set_layout_cell(row, col, char)
        self._update_layout_text_from_grid()
        self._set_selected_layout_cell(None, None)

    def _clear_goals(self):
        for row in range(len(self.layout_grid)):
            for col in range(len(self.layout_grid[0])):
                if self.layout_grid[row][col] == "g":
                    self._set_layout_cell(row, col, "x")
        self._update_layout_text_from_grid()
        self._set_selected_layout_cell(None, None)

    def _load_cell_direction_constraints(self, constraints):
        if not isinstance(constraints, (list, tuple)):
            return
        rows = len(self.layout_grid)
        cols = len(self.layout_grid[0]) if rows else 0
        for entry in constraints:
            if not isinstance(entry, dict):
                continue
            x = entry.get("x")
            y = entry.get("y")
            dirs = entry.get("dirs")
            if x is None or y is None or dirs is None:
                dirs = entry.get("directions")
            if x is None or y is None or dirs is None:
                continue
            try:
                col = int(x)
                row = int(y)
            except (TypeError, ValueError):
                continue
            if row < 0 or col < 0 or row >= rows or col >= cols:
                continue
            self._apply_lane_dirs_to_cell(row, col, dirs, log_errors=False)


    def _build_env_config_payload(self):
        layout_text = self._grid_to_layout_text().strip()
        layout_value = None
        if layout_text:
            grid, error = self._validate_layout_text(layout_text)
            if grid is None or error:
                return None, error or "Layout is invalid."
            layout_value = layout_text

        selected_image_layers = self._selected_image_layers()
        if not selected_image_layers:
            return None, "Select at least one image observation layer."

        max_inactivity = self.gen_max_inactivity.value()
        max_steps = self.gen_max_steps.value()
        constraint_list = []
        for (row, col), dirs in sorted(self.cell_direction_constraints.items()):
            constraint_list.append({"x": int(col), "y": int(row), "dirs": list(dirs)})
        payload = {
            "env_id": self.gen_env_id.text().strip() or "rware-tiny-2ag-v2",
            "kwargs": {
                "layout": layout_value,
                "shelf_rows": self.gen_shelf_rows.value(),
                "shelf_columns": self.gen_shelf_cols.value(),
                "column_height": self.gen_column_height.value(),
                "n_agents": self.gen_agents.value(),
                "msg_bits": self.gen_msg_bits.value(),
                "sensor_range": self.gen_sensor_range.value(),
                "request_queue_size": self.gen_request_queue.value(),
                "max_inactivity_steps": None if max_inactivity == 0 else max_inactivity,
                "max_steps": None if max_steps == 0 else max_steps,
                "reward_type": self.gen_reward_type.currentText(),
                "reward_delivery_weight": self.gen_reward_delivery_weight.value(),
                "observation_type": self.gen_obs_type.currentText(),
                "render_mode": self.gen_render_mode.currentText() or None,
                "image_observation_directional": self.gen_image_obs_directional.isChecked(),
                "normalised_coordinates": self.gen_normalised.isChecked(),
                "lane_observation": self.gen_lane_observation.isChecked(),
                "dedicated_requests": self.gen_dedicated_requests.isChecked(),
                "cell_direction_constraints": constraint_list or None,
                "image_observation_layers": selected_image_layers,
            },
        }
        return payload, ""

    def _save_env_config(self):
        payload, error = self._build_env_config_payload()
        if payload is None:
            QtWidgets.QMessageBox.warning(self, "Invalid layout", error)
            return
        env_id = payload.get("env_id") or "rware-env"
        safe_env_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in env_id)
        timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        suggested_name = f"{safe_env_id}_{timestamp}.json"
        suggested_path = str(ENV_CONFIG_DIR / suggested_name)
        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Env Config",
            suggested_path,
            "Env Config (*.json);;All Files (*)",
        )
        if not selected_path:
            return
        filename = Path(selected_path).name.strip()
        filename = filename.strip()
        if not filename:
            QtWidgets.QMessageBox.warning(self, "Invalid name", "Filename cannot be empty.")
            return
        if not filename.lower().endswith(".json"):
            filename = f"{filename}.json"

        config_path = ENV_CONFIG_DIR / filename
        if config_path.exists():
            reply = QtWidgets.QMessageBox.question(
                self,
                "Overwrite config",
                f"{config_path.name} already exists.\nOverwrite it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        config_path.write_text(json.dumps(payload, indent=2) + "\n")
        self.env_config_path.setText(str(config_path))
        self._append_log(f"[launcher] env config saved: {config_path}")
        self._refresh_env_config_combos()
        self._select_env_config_path(str(config_path), payload.get("env_id"))

    def _copy_env_config_path(self):
        path = self.env_config_path.text().strip()
        if not path:
            return
        QtWidgets.QApplication.clipboard().setText(path)
        self._append_log("[launcher] env config path copied to clipboard")

    def _delete_env_config(self):
        path = self.env_config_path.text().strip()
        if not path:
            QtWidgets.QMessageBox.information(self, "No selection", "No env config selected.")
            return
        path_obj = Path(path)
        if not path_obj.exists():
            QtWidgets.QMessageBox.warning(self, "Missing file", "Selected env config does not exist.")
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete env config",
            f"Delete env config?\n{path_obj}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            path_obj.unlink()
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Delete failed", str(exc))
            return
        self.env_config_path.clear()
        self._refresh_env_config_combos()
        self._append_log(f"[launcher] env config deleted: {path_obj}")

    def _apply_env_config_to_tabs(self, path, env_id=None):
        if hasattr(self, "hp_env_config"):
            self.hp_env_config.setText(path)
        if hasattr(self, "tr_env_config"):
            self.tr_env_config.setText(path)
        if hasattr(self, "ev_env_config"):
            self.ev_env_config.setText(path)
        if hasattr(self, "env_config_path"):
            self.env_config_path.setText(path)
        if env_id:
            if hasattr(self, "hp_env"):
                self.hp_env.setText(env_id)
            if hasattr(self, "tr_env_resolved"):
                self.tr_env_resolved.setText(_seac_env_name(env_id))
            if hasattr(self, "ev_env_resolved"):
                self.ev_env_resolved.setText(_seac_env_name(env_id))
        self._load_env_config_into_generator(path)

    def _browse_env_config(self, target_widget):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Env Config", str(ENV_CONFIG_DIR), "Env Config (*.json);;All Files (*)"
        )
        if not filename:
            return
        target_widget.setText(filename)
        env_id = None
        try:
            data = json.loads(Path(filename).read_text())
            env_id = data.get("env_id")
        except Exception:
            env_id = None
        self._select_env_config_path(filename, env_id)

    def _browse_traffic_scenario(self):
        start_dir = (
            str(TRAFFIC_SCENARIO_DIR)
            if TRAFFIC_SCENARIO_DIR.exists()
            else str(SEAC_DIR)
        )
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Traffic Scenario",
            start_dir,
            "JSON (*.json);;All Files (*)",
        )
        if not filename:
            return
        self.gen_traffic_scenario.setText(filename)

    def _browse_directory(self, target_widget, title):
        initial_dir = target_widget.text().strip() or str(Path.home())
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, title, initial_dir
        )
        if not directory:
            return
        target_widget.setText(directory)

    def _build_env_selector(self):
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        combo = QtWidgets.QComboBox()
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        combo.setToolTip("Pick a saved env config from the shared directory.")
        combo.currentIndexChanged.connect(lambda _: self._on_env_config_selected(combo))
        self.env_config_combos.append(combo)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_env_config_combos)
        refresh.setToolTip("Refresh the saved env list.")
        layout.addWidget(combo, stretch=1)
        layout.addWidget(refresh)
        return row

    def _scan_env_configs(self):
        entries = []
        if not ENV_CONFIG_DIR.exists():
            return entries
        for path in sorted(ENV_CONFIG_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            env_id = None
            try:
                data = json.loads(path.read_text())
                env_id = data.get("env_id")
            except Exception:
                env_id = None
            label = f"{path.name}"
            if env_id:
                label = f"{env_id} — {path.name}"
            entries.append((label, str(path), env_id))
        return entries

    def _refresh_env_config_combos(self):
        entries = self._scan_env_configs()
        for combo in self.env_config_combos:
            current_path = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Select saved env...", None)
            for label, path, _ in entries:
                combo.addItem(label, path)
            combo.blockSignals(False)
            if current_path:
                idx = combo.findData(current_path)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def _select_env_config_path(self, path, env_id=None):
        for combo in self.env_config_combos:
            idx = combo.findData(path)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self._apply_env_config_to_tabs(path, env_id)

    def _on_env_config_selected(self, combo):
        path = combo.currentData()
        if not path:
            return
        env_id = None
        try:
            data = json.loads(Path(path).read_text())
            env_id = data.get("env_id")
        except Exception:
            env_id = None
        self._apply_env_config_to_tabs(path, env_id)

    def _load_env_config_into_generator(self, path):
        if self._env_config_loading:
            return
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
        except Exception:
            return
        kwargs = data.get("kwargs") or {}
        if not isinstance(kwargs, dict):
            return
        self._env_config_loading = True
        try:
            self._reset_env_generator_defaults()
            env_id = data.get("env_id")
            if env_id:
                self.gen_env_id.setText(env_id)

            def set_spin(spinbox, key):
                if key in kwargs and kwargs[key] is not None:
                    self._set_spin_value(spinbox, int(kwargs[key]))

            set_spin(self.gen_shelf_rows, "shelf_rows")
            set_spin(self.gen_shelf_cols, "shelf_columns")
            set_spin(self.gen_column_height, "column_height")
            set_spin(self.gen_agents, "n_agents")
            set_spin(self.gen_msg_bits, "msg_bits")
            set_spin(self.gen_sensor_range, "sensor_range")
            set_spin(self.gen_request_queue, "request_queue_size")

            if "max_inactivity_steps" in kwargs:
                value = kwargs["max_inactivity_steps"]
                self._set_spin_value(self.gen_max_inactivity, int(value or 0))
            if "max_steps" in kwargs:
                value = kwargs["max_steps"]
                self._set_spin_value(self.gen_max_steps, int(value or 0))

            reward_type = kwargs.get("reward_type")
            if reward_type is not None:
                reward_name = str(reward_type)
                if "." in reward_name:
                    reward_name = reward_name.split(".")[-1]
                reward_name = reward_name.upper()
                if reward_name in _REWARD_TYPE_MAP:
                    self.gen_reward_type.setCurrentText(reward_name)

            obs_type = kwargs.get("observation_type")
            if obs_type is not None:
                obs_name = str(obs_type)
                if "." in obs_name:
                    obs_name = obs_name.split(".")[-1]
                obs_name = obs_name.upper()
                if obs_name in _OBS_TYPE_MAP:
                    self.gen_obs_type.setCurrentText(obs_name)
            render_mode = kwargs.get("render_mode")
            if render_mode is None:
                self.gen_render_mode.setCurrentText("")
            else:
                render_name = str(render_mode).strip()
                idx = self.gen_render_mode.findText(render_name)
                if idx >= 0:
                    self.gen_render_mode.setCurrentIndex(idx)

            if "image_observation_directional" in kwargs:
                self.gen_image_obs_directional.setChecked(
                    bool(kwargs["image_observation_directional"])
                )
            if "normalised_coordinates" in kwargs:
                self.gen_normalised.setChecked(bool(kwargs["normalised_coordinates"]))
            if "lane_observation" in kwargs:
                self.gen_lane_observation.setChecked(bool(kwargs["lane_observation"]))
            if "dedicated_requests" in kwargs:
                self.gen_dedicated_requests.setChecked(bool(kwargs["dedicated_requests"]))
            if "reward_delivery_weight" in kwargs:
                self.gen_reward_delivery_weight.setValue(float(kwargs["reward_delivery_weight"]))
            self._set_image_layer_checks(kwargs.get("image_observation_layers"))

            layout_value = kwargs.get("layout")
            if layout_value:
                self._apply_layout_from_string(layout_value)
            constraints = kwargs.get("cell_direction_constraints")
            if constraints:
                self._load_cell_direction_constraints(constraints)
                self._update_layout_text_from_grid()
        finally:
            self._env_config_loading = False

    def _reset_env_generator_defaults(self):
        self.gen_env_id.setText("rware-tiny-2ag-v2")
        self._set_spin_value(self.gen_shelf_rows, 1)
        self._set_spin_value(self.gen_shelf_cols, 3)
        self._set_spin_value(self.gen_column_height, 8)
        self._set_spin_value(self.gen_agents, 2)
        self._set_spin_value(self.gen_msg_bits, 0)
        self._set_spin_value(self.gen_sensor_range, 1)
        self._set_spin_value(self.gen_request_queue, 2)
        self._set_spin_value(self.gen_max_inactivity, 0)
        self._set_spin_value(self.gen_max_steps, 500)
        self.gen_reward_type.setCurrentText("INDIVIDUAL")
        self.gen_obs_type.setCurrentText("FLATTENED")
        self.gen_render_mode.setCurrentText("")
        self.gen_image_obs_directional.setChecked(True)
        self.gen_normalised.setChecked(False)
        self.gen_lane_observation.setChecked(False)
        self.gen_dedicated_requests.setChecked(True)
        self.gen_reward_delivery_weight.setValue(1.0)
        self._set_image_layer_checks(list(_DEFAULT_IMAGE_LAYERS))
        self.brush_select.setChecked(True)
        self.brush_lane_up.setChecked(True)
        self.brush_lane_down.setChecked(True)
        self.brush_lane_left.setChecked(True)
        self.brush_lane_right.setChecked(True)


    def _append_log(self, text):
        self.log.appendPlainText(text)

    def _clear_log(self):
        self.log.clear()

    def _update_buttons(self):
        running = self.process.state() != QtCore.QProcess.NotRunning
        for btn in self.run_buttons:
            btn.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.status_label.setText("Running" if running else "Idle")

    def _on_started(self):
        self._append_log("[launcher] process started")
        self._update_buttons()

    def _on_finished(self, exit_code, exit_status):
        self._force_kill_timer.stop()
        status = "ok" if exit_status == QtCore.QProcess.NormalExit else "crashed"
        self._append_log(f"[launcher] process finished ({status}), exit code {exit_code}")
        self._update_buttons()
        if self._closing_after_stop:
            self._closing_after_stop = False
            self.close()

    def _on_stdout(self):
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            self._append_log(data.rstrip())

    def _stop_process(self):
        self._request_stop_process(log_prefix="[launcher]")

    def _request_stop_process(self, log_prefix):
        if self.process.state() == QtCore.QProcess.NotRunning:
            return
        if self._force_kill_timer.isActive():
            return
        self._append_log(f"{log_prefix} stopping process...")
        self.process.terminate()
        self._force_kill_timer.start(3000)

    def _force_kill_process(self):
        if self.process.state() != QtCore.QProcess.NotRunning:
            self._append_log("[launcher] forcing process kill...")
            self.process.kill()

    def closeEvent(self, event):
        if self.process.state() == QtCore.QProcess.NotRunning:
            event.accept()
            return
        if self._closing_after_stop:
            event.ignore()
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Process running",
            "Training/evaluation is still running.\nStop it and close the launcher?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            event.ignore()
            return

        self._closing_after_stop = True
        self._request_stop_process(log_prefix="[launcher] closing: ")
        event.ignore()

    def _start_process(self, working_dir, args, extra_env=None):
        if self.process.state() != QtCore.QProcess.NotRunning:
            QtWidgets.QMessageBox.warning(self, "Busy", "A process is already running.")
            return

        python_bin = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        if extra_env:
            for key, value in extra_env.items():
                env.insert(key, value)

        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(working_dir))

        cmd_str = " ".join([str(python_bin)] + args)
        self._append_log(f"[launcher] cwd: {working_dir}")
        self._append_log(f"[launcher] cmd: {cmd_str}")

        self.process.start(str(python_bin), args)

    def _run_human_play(self):
        env_name = self.hp_env.text().strip()
        if not env_name:
            QtWidgets.QMessageBox.warning(self, "Missing env", "Please set an env name.")
            return
        args = [
            "human_play.py",
            "--env",
            env_name,
            "--max_steps",
            str(self.hp_max_steps.value()),
        ]
        env_config = self.hp_env_config.text().strip()
        if env_config:
            args.extend(["--env-config", env_config])
        if self.hp_display_info.isChecked():
            args.append("--display_info")
        extra_env = {"PYGLET_HEADLESS": "1"} if self.hp_headless.isChecked() else None
        self._start_process(ROBOTIC_WAREHOUSE_DIR, args, extra_env=extra_env)

    def _run_training(self):
        env_config = self.tr_env_config.text().strip()
        if not env_config:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing env config",
                "Please select a generated env JSON for training.",
            )
            return
        env_name = _seac_env_name(_env_id_from_config(env_config))
        if not env_name:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid env config",
                "The selected env JSON does not contain a valid env_id.",
            )
            return
        if self.tr_device.currentText() == "cuda" and not torch.cuda.is_available():
            QtWidgets.QMessageBox.warning(
                self,
                "CUDA unavailable",
                "CUDA is not available in this session. Training cannot start on GPU until the NVIDIA driver/runtime is visible.",
            )
            return
        num_env_steps = self.tr_num_env_steps.text().strip() or "40000000"
        named_configs_text = self.tr_named_configs.text().strip()
        named_configs = []
        if named_configs_text:
            named_configs = [
                token
                for token in re.split(r"[,\s]+", named_configs_text)
                if token
            ]
        args = [
            "seac/train.py",
            "with",
            *named_configs,
            f"env_name={env_name}",
            f"time_limit={self.tr_time_limit.value()}",
            f"num_env_steps={num_env_steps}",
            f"seed={self.tr_seed.value()}",
            f"log_interval={self.tr_log_interval.value()}",
            f"save_interval={self.tr_save_interval.value()}",
            f"eval_interval={self.tr_eval_interval.value()}",
            f"episodes_per_eval={self.tr_episodes_per_eval.value()}",
            f"save_dir={self.tr_save_dir.text().strip()}",
            f"eval_dir={self.tr_eval_dir.text().strip()}",
            f"loss_dir={self.tr_loss_dir.text().strip()}",
            f"algorithm.num_processes={self.tr_num_processes.value()}",
            f"algorithm.num_steps={self.tr_num_steps.value()}",
            f"algorithm.lr={self.tr_lr.value()}",
            f"algorithm.gamma={self.tr_gamma.value()}",
            f"algorithm.use_gae={self.tr_use_gae.isChecked()}",
            f"algorithm.gae_lambda={self.tr_gae_lambda.value()}",
            f"algorithm.use_linear_lr_decay={self.tr_use_linear_lr_decay.isChecked()}",
            f"algorithm.recurrent_policy={self.tr_recurrent_policy.isChecked()}",
            f"algorithm.entropy_coef={self.tr_entropy_coef.value()}",
            f"algorithm.value_loss_coef={self.tr_value_loss_coef.value()}",
            f"algorithm.seac_coef={self.tr_seac_coef.value()}",
            f"algorithm.device={self.tr_device.currentText()}",
        ]
        if env_config:
            args.append(f"env_config={env_config}")
        self._start_process(SEAC_DIR, args)

    def _run_evaluation(self):
        model_path = self.ev_path.text().strip()
        env_config = self.ev_env_config.text().strip()
        if not env_config or not model_path:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing input",
                "Please set the generated env JSON and model path.",
            )
            return
        env_name = _seac_env_name(_env_id_from_config(env_config))
        if not env_name:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid env config",
                "The selected env JSON does not contain a valid env_id.",
            )
            return
        args = [
            "seac/evaluate.py",
            "--env",
            env_name,
            "--path",
            model_path,
            "--time_limit",
            str(self.ev_time_limit.value()),
            "--episodes",
            str(self.ev_episodes.value()),
            "--output-dir",
            self.ev_output_dir.text().strip(),
        ]
        if env_config:
            args.extend(["--env-config", env_config])
        if self.ev_record_video.isChecked():
            args.append("--record-video")
        if self.ev_export_csv.isChecked():
            args.append("--export-csv")
        self._start_process(SEAC_DIR, args)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Launcher()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
