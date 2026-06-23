import cv2
import numpy as np
import pyautogui
import math
import os
import threading
import time
import keyboard
import pygetwindow as gw
import mss
import webview
import webbrowser
import sys
import platform
import subprocess
from treys import Card, Evaluator
from ultralytics import YOLO

# --- Helper for PyInstaller ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Configuration ---
ZONE_MODEL_PATH = 'model/zone_best.pt' 
CARD_MODEL_PATH = 'model/card_m_1024_best.pt'

# --- HTML/CSS Frontend ---
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Poker Vision Bot</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg:        #07070f;
            --surface:   #0e0e1c;
            --surface2:  #13131f;
            --border:    rgba(255,255,255,0.06);
            --border-hi: rgba(99,102,241,0.35);
            --text:      #e2e4f0;
            --text-dim:  #6b6f8a;
            --primary:   #6366f1;
            --primary-glow: rgba(99,102,241,0.18);
            --cyan:      #22d3ee;
            --cyan-glow: rgba(34,211,238,0.15);
            --success:   #10b981;
            --success-glow: rgba(16,185,129,0.15);
            --danger:    #f43f5e;
            --danger-glow: rgba(244,63,94,0.15);
            --warning:   #f59e0b;
        }
 
        *, *::before, *::after { box-sizing: border-box; }
 
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            padding: 16px 12px 56px;
            background-image:
                linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px);
            background-size: 40px 40px;
        }
 
        .container {
            max-width: 640px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
 
        /* ── Header bar ── */
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            padding: 14px 16px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            flex-wrap: wrap;
        }
        .header-left { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
        .header-icon {
            width: 34px; height: 34px;
            background: var(--primary-glow);
            border: 1px solid var(--border-hi);
            border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: 17px;
            flex-shrink: 0;
        }
        .header-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 17px;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.3px;
        }
        .header-sub {
            font-size: 10px;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-top: 1px;
        }
 
        /* ── Main action card ── */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
        }
        .card-label {
            font-size: 10px;
            font-weight: 600;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 16px;
        }
 
        /* ── Action button ── */
        .action-btn {
            width: 100%;
            padding: 16px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 14px;
            font-weight: 700;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            letter-spacing: 0.5px;
            transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1);
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .action-btn:last-child { margin-bottom: 0; }
        .btn-start {
            background: var(--success);
            color: #052e16;
            box-shadow: 0 0 0 0 var(--success-glow);
        }
        .btn-start:hover {
            background: #0fca8e;
            box-shadow: 0 4px 20px rgba(16,185,129,0.35);
            transform: translateY(-1px);
        }
        .btn-stop {
            background: var(--danger);
            color: #1a0009;
            box-shadow: 0 0 0 0 var(--danger-glow);
        }
        .btn-stop:hover {
            background: #f5607a;
            box-shadow: 0 4px 20px rgba(244,63,94,0.35);
            transform: translateY(-1px);
        }
        .btn-secondary {
            background: var(--primary);
            color: #fff;
            box-shadow: 0 0 0 0 var(--primary-glow);
        }
        .btn-secondary:hover {
            background: #818cf8;
            box-shadow: 0 4px 20px rgba(99,102,241,0.35);
            transform: translateY(-1px);
        }
        .action-btn:active { transform: translateY(1px); box-shadow: none; }
 
        /* ── Settings ── */
        .settings-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 0;
            border-bottom: 1px solid var(--border);
            position: relative;
        }
        .settings-row:last-child { border-bottom: none; padding-bottom: 0; }
        .settings-row:first-of-type { padding-top: 0; }
        .row-label { font-size: 13px; font-weight: 500; color: var(--text); }
        .row-hint  { font-size: 11px; color: var(--text-dim); margin-top: 2px; }
 
        /* Toggle switch */
        .switch { position: relative; display: inline-block; width: 42px; height: 23px; flex-shrink: 0; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider {
            position: absolute; inset: 0;
            background: rgba(255,255,255,0.08);
            border: 1px solid var(--border);
            border-radius: 999px;
            cursor: pointer;
            transition: 0.25s;
        }
        .slider::before {
            content: "";
            position: absolute;
            width: 17px; height: 17px;
            left: 2px; bottom: 2px;
            background: #fff;
            border-radius: 50%;
            transition: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 1px 4px rgba(0,0,0,0.4);
        }
        input:checked + .slider { background: var(--success); border-color: var(--success); }
        input:checked + .slider::before { transform: translateX(19px); }

        /* ── Logs ── */
        .log-container {
            background: #000;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            font-family: monospace;
            font-size: 11px;
            color: #a5b4fc;
            height: 180px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            user-select: text;
            -webkit-user-select: text;
        }
        .log-line { margin-bottom: 2px; line-height: 1.4; }
    </style>
</head>
<body>
<div class="container">
 
    <!-- Header -->
    <div class="header">
        <div class="header-left">
            <div class="header-icon">🃏</div>
            <div>
                <div class="header-title">Poker Vision</div>
                <div class="header-sub">WSOP Auto-Solver</div>
            </div>
        </div>
    </div>
 
    <!-- Main control -->
    <div class="card">
        <div class="card-label">Automation</div>
        <button id="start-btn" class="action-btn btn-start" onclick="toggleBot()">Start Automation</button>
        <button class="action-btn btn-secondary" onclick="openWsop()">Play WSOP in Browser</button>
    </div>
 
    <!-- Settings -->
    <div class="card">
        <div class="card-label">Settings</div>
 
        <div class="settings-row" title="Automatically finds and brings Chrome to the foreground before evaluating.">
            <div>
                <div class="row-label">Auto-Switch</div>
                <div class="row-hint">Bring Google Chrome window to foreground</div>
            </div>
            <label class="switch">
                <input type="checkbox" id="toggle-switch" onchange="updateSettings()">
                <span class="slider"></span>
            </label>
        </div>
 
        <div class="settings-row">
            <div>
                <div class="row-label">Debug Mode</div>
                <div class="row-hint">Save bounding box frames to ~/Documents/PokerVisionBot/</div>
                <button onclick="openDebugFolder()" style="margin-top: 8px; padding: 4px 10px; font-size: 11px; font-weight: 600; font-family: 'Space Grotesk', sans-serif; background: var(--surface2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; cursor: pointer; transition: all 0.2s;" onmouseover="this.style.borderColor='var(--border-hi)'; this.style.background='rgba(255,255,255,0.05)'" onmouseout="this.style.borderColor='var(--border)'; this.style.background='var(--surface2)'">📂 Open Folder</button>
            </div>
            <label class="switch">
                <input type="checkbox" id="toggle-debug" onchange="updateSettings()">
                <span class="slider"></span>
            </label>
        </div>

        <div class="settings-row">
            <div>
                <div class="row-label">Show Logs</div>
                <div class="row-hint">Display console output in dashboard</div>
            </div>
            <label class="switch">
                <input type="checkbox" id="toggle-logs" onchange="toggleLogs(this.checked)">
                <span class="slider"></span>
            </label>
        </div>
    </div>

    <!-- Logs Card -->
    <div id="log-card" class="card" style="display: none;">
        <div class="card-label">Console Logs</div>
        <div id="log-container" class="log-container"></div>
    </div>
 
</div>
 
    <script>
        window.addEventListener('pywebviewready', function() {
            // Start polling the backend for state updates once UI is loaded
            setInterval(fetchState, 500);
            fetchState();
        });
 
        async function fetchState() {
            if (!window.pywebview) return;
            const state = await window.pywebview.api.get_state();
            
            // Update Buttons
            const startBtn = document.getElementById('start-btn');
            startBtn.innerText = state.is_running ? "Stop Automation" : "Start Automation";
            startBtn.className = state.is_running ? "action-btn btn-stop" : "action-btn btn-start";

            // Update Logs
            const logToggle = document.getElementById('toggle-logs');
            if (logToggle && logToggle.checked && state.logs) {
                const logCont = document.getElementById('log-container');
                const isScrolledToBottom = logCont.scrollHeight - logCont.clientHeight <= logCont.scrollTop + 10;
                
                logCont.innerHTML = state.logs.map(l => `<div class="log-line">${l.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>`).join('');
                if (isScrolledToBottom) logCont.scrollTop = logCont.scrollHeight;
            }
        }
 
        async function toggleBot() {
            if (window.pywebview) {
                await window.pywebview.api.toggle_bot();
                fetchState();
            }
        }

        function updateSettings() {
            if (window.pywebview) {
                const autoSwitch = document.getElementById('toggle-switch').checked;
                const debugMode = document.getElementById('toggle-debug').checked;
                window.pywebview.api.set_settings(autoSwitch, debugMode);
            }
        }

        function toggleLogs(show) {
            document.getElementById('log-card').style.display = show ? 'block' : 'none';
            if (show) fetchState();
        }
 
        function openWsop() {
            if (window.pywebview) {
                window.pywebview.api.open_game();
            }
        }

        function openDebugFolder() {
            if (window.pywebview) {
                window.pywebview.api.open_debug_folder();
            }
        }
    </script>
</body>
</html>
"""

# --- Backend UI API & Logger Redirect ---
class BotApi:
    def __init__(self):
        self.is_running = False
        self.auto_switch = False
        self.debug_mode = False
        self.exit_flag = False
        self.start_time = time.time()
        self.logs = []
        
        # Stats dictionary
        self.stats = {
            'hands': 0,
            'left': 0,
            'right': 0,
            'ties': 0
        }

    def toggle_bot(self):
        self.is_running = not self.is_running
        print(f"--- BOT {'STARTED' if self.is_running else 'STOPPED'} ---")
        return self.is_running

    def set_settings(self, auto_switch, debug_mode):
        self.auto_switch = auto_switch
        self.debug_mode = debug_mode
        print(f"Settings Updated: Auto-Switch={auto_switch}, Debug={debug_mode}")

    def open_game(self):
        print("Opening WSOP game in browser...")
        webbrowser.open("https://www.playwsop.com/play")

    def open_debug_folder(self):
        documents_folder = os.path.join(os.path.expanduser('~'), 'Documents', 'PokerVisionBot')
        os.makedirs(documents_folder, exist_ok=True)
        print(f"Opening debug folder: {documents_folder}")
        
        if platform.system() == "Windows":
            os.startfile(documents_folder)
        elif platform.system() == "Darwin": # macOS
            subprocess.Popen(["open", documents_folder])
        else: # Linux
            subprocess.Popen(["xdg-open", documents_folder])

    def get_state(self):
        # Syncs state with frontend Javascript
        return {
            'is_running': self.is_running,
            'stats': {
                'elapsed': int(time.time() - self.start_time) if self.is_running else 0,
                'hands': self.stats['hands'],
                'left': self.stats['left'],
                'right': self.stats['right'],
                'ties': self.stats['ties']
            },
            'logs': self.logs
        }

class LoggerRedirect:
    def __init__(self, api_ref):
        self.api = api_ref
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        if message.strip():
            self.api.logs.append(message.strip())
            # Keep only the last 60 lines of logs so the UI doesn't lag
            if len(self.api.logs) > 60:
                self.api.logs.pop(0)

    def flush(self):
        self.terminal.flush()

# --- Helper Functions ---
def switch_to_chrome():
    try:
        windows = gw.getWindowsWithTitle('Chrome')
        if windows:
            for win in windows:
                if win.title: # Ignore invisible background windows
                    try:
                        if not win.isActive:
                            win.activate()
                            time.sleep(0.15) # Brief pause to allow OS animation
                        return
                    except Exception:
                        continue
    except Exception as e:
        print(f"Failed to switch to Chrome: {e}")

def save_debug_image(img_rgb):
    documents_folder = os.path.join(os.path.expanduser('~'), 'Documents', 'PokerVisionBot')
    os.makedirs(documents_folder, exist_ok=True)
    filepath = os.path.join(documents_folder, 'debug_cascaded_output.jpg')
    cv2.imwrite(filepath, img_rgb)
    print(f">>> SAVED DEBUG IMAGE TO '{filepath}'")

def normalize_card(label):
    label = label.upper()
    if len(label) == 3 and label.startswith('10'):
        return 'T' + label[-1].lower()
    elif len(label) == 2:
        return label[0] + label[1].lower()
    return label

# --- Core Computer Vision Logic ---
def scan_board_cascaded(img_rgb, zone_model, card_model, conf_zone=0.25, conf_card=0.10, debug=False):
    print(f"\n--- STAGE 1: DETECTING ZONES ---")
    if img_rgb is None:
        return [], None
        
    h, w = img_rgb.shape[:2]
    zone_results = zone_model(img_rgb, conf=conf_zone, imgsz=640, verbose=False)
    
    detected_zones = {}
    for result in zone_results:
        for box in result.boxes:
            zx1, zy1, zx2, zy2 = map(int, box.xyxy[0])
            label = zone_model.names[int(box.cls[0])].lower()
            
            if label in ['community_card', 'left_hand', 'right_hand']:
                detected_zones[label] = (zx1, zy1, zx2, zy2)

    print(f"Found {len(detected_zones)}/3 required zones.")
    if len(detected_zones) < 3:
        print("Waiting for all 3 zones to be visible...")
        return [], None

    all_cards = []
    valid_ranks = {'2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'}
    valid_suits = {'S', 'H', 'D', 'C', 'SPADE', 'HEART', 'DIAMOND', 'CLUB', 'SPADES', 'HEARTS', 'DIAMONDS', 'CLUBS'}

    print(f"--- STAGE 2: READING CARDS ---")
    
    for zone_label, (zx1, zy1, zx2, zy2) in detected_zones.items():
        padding = 15
        p_zx1 = max(0, zx1 - padding)
        p_zy1 = max(0, zy1 - padding)
        p_zx2 = min(w, zx2 + padding)
        p_zy2 = min(h, zy2 + padding)
        
        zone_crop = img_rgb[p_zy1:p_zy2, p_zx1:p_zx2]
        crop_h, crop_w = zone_crop.shape[:2]
        
        card_results = card_model(zone_crop, conf=conf_card, imgsz=1024, verbose=False)
        
        ranks = []
        suits = []
        
        raw_detections = [card_model.names[int(box.cls[0])].upper() for result in card_results for box in result.boxes]
        print(f"   [{zone_label}] Raw Detections: {raw_detections}")
        
        for result in card_results:
            for box in result.boxes:
                cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                c_label = card_model.names[int(box.cls[0])].upper()
                
                ccx = (cx1 + cx2) // 2
                ccy = (cy1 + cy2) // 2
                
                item_data = {
                    'label': c_label, 'score': confidence,
                    'cx': ccx, 'cy': ccy,
                    'box': (cx1, cy1, cx2, cy2), 'used': False
                }
                
                if c_label in valid_ranks:
                    ranks.append(item_data)
                elif c_label in valid_suits:
                    suits.append(item_data)

        for r in ranks:
            best_suit = None
            min_dist = float('inf')
            
            for s in suits:
                if s['used']: continue
                
                is_below = s['cy'] > (r['cy'] - 10)
                is_aligned = abs(s['cx'] - r['cx']) < (crop_w * 0.08)
                
                if is_below and is_aligned:
                    dist = math.hypot(r['cx'] - s['cx'], r['cy'] - s['cy'])
                    if dist < min_dist:
                        min_dist = dist
                        best_suit = s
            
            if best_suit:
                best_suit['used'] = True
                suit_char = best_suit['label'][0]
                combined_label = r['label'] + suit_char 
                
                global_x1 = min(r['box'][0], best_suit['box'][0]) + p_zx1
                global_y1 = min(r['box'][1], best_suit['box'][1]) + p_zy1
                global_x2 = max(r['box'][2], best_suit['box'][2]) + p_zx1
                global_y2 = max(r['box'][3], best_suit['box'][3]) + p_zy1
                
                global_cx = (global_x1 + global_x2) // 2
                global_cy = (global_y1 + global_y2) // 2
                
                target_zone = 'board' if zone_label == 'community_card' else ('left' if zone_label == 'left_hand' else 'right')
                
                all_cards.append({
                    'label': combined_label, 'score': (r['score'] + best_suit['score']) / 2, 
                    'cx': global_cx, 'cy': global_cy,
                    'zone': target_zone, 'box': (global_x1, global_y1, global_x2, global_y2)
                })

    matches = sorted(all_cards, key=lambda x: x['cx'])

    debug_img = None
    if debug:
        debug_img = img_rgb.copy()
        for z_label, (zx1, zy1, zx2, zy2) in detected_zones.items():
            cv2.rectangle(debug_img, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
            cv2.putText(debug_img, z_label, (zx1, zy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        for m in matches:
            gx1, gy1, gx2, gy2 = m['box']
            color = (0, 255, 0) if m['zone'] == 'board' else (255, 0, 0)
            cv2.rectangle(debug_img, (gx1, gy1), (gx2, gy2), color, 2)
            cv2.putText(debug_img, f"{m['label']} {m['score']:.2f}", (gx1, gy1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return matches, debug_img

def evaluate_and_click(matches, offset_x=0, offset_y=0):
    print("\n--- EVALUATING HANDS ---")
    board_matches = [m for m in matches if m['zone'] == 'board']
    left_matches = [m for m in matches if m['zone'] == 'left']
    right_matches = [m for m in matches if m['zone'] == 'right']

    if len(board_matches) != 5 or len(left_matches) != 2 or len(right_matches) != 2:
        return False, None

    evaluator = Evaluator()
    try:
        board = [Card.new(normalize_card(m['label'])) for m in board_matches]
        left_hand = [Card.new(normalize_card(m['label'])) for m in left_matches]
        right_hand = [Card.new(normalize_card(m['label'])) for m in right_matches]
    except Exception as e:
        return False, None

    left_score = evaluator.evaluate(board, left_hand)
    right_score = evaluator.evaluate(board, right_hand)

    print(f"Left Hand: {evaluator.class_to_string(evaluator.get_rank_class(left_score))} (Score: {left_score})")
    print(f"Right Hand: {evaluator.class_to_string(evaluator.get_rank_class(right_score))} (Score: {right_score})")

    winner_str = ""
    if left_score < right_score:
        print("Winner: LEFT HAND")
        winner_str = "left"
        winning_matches = left_matches
    elif right_score < left_score:
        print("Winner: RIGHT HAND")
        winner_str = "right"
        winning_matches = right_matches
    else:
        print("Winner: TIE")
        winner_str = "ties"
        winning_matches = left_matches

    target_x = sum(m['cx'] for m in winning_matches) / len(winning_matches)
    target_y = sum(m['cy'] for m in winning_matches) / len(winning_matches)

    final_x = int(target_x + offset_x)
    final_y = int(target_y + offset_y)

    print(f"Action: Clicking coordinates X:{final_x}, Y:{final_y}")
    pyautogui.click(x=final_x, y=final_y)
    return True, winner_str

# --- Background Worker Thread ---
def run_bot_loop(api):
    try:
        print("Loading AI Models... (This might take a second)")
        zone_model = YOLO(resource_path(ZONE_MODEL_PATH))
        card_model = YOLO(resource_path(CARD_MODEL_PATH))
        print("Both models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    was_running = False

    with mss.MSS() as sct:
        monitor = sct.monitors[1] 
        
        while not api.exit_flag:
            if not api.is_running:
                was_running = False
                time.sleep(0.1)
                continue

            if not was_running:
                api.start_time = time.time() # Reset clock when start clicked
                print("\n--- Bot Started: Preparing to scan ---")
                if api.auto_switch:
                    switch_to_chrome()
                else:
                    print("Auto-switch disabled. Waiting 5 seconds for you to manually switch to the browser window...")
                    time.sleep(5.0)
                was_running = True
            
            sct_img = sct.grab(monitor)
            img_rgb = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            
            matches, debug_img = scan_board_cascaded(img_rgb, zone_model, card_model, conf_zone=0.25, conf_card=0.10, debug=api.debug_mode) 
            
            if api.debug_mode and debug_img is not None:
                save_debug_image(debug_img)
            
            if matches and len(matches) == 9:
                success, winner = evaluate_and_click(matches, offset_x=monitor['left'], offset_y=monitor['top'])
                
                if success:
                    # Update internal stats for UI
                    api.stats['hands'] += 1
                    api.stats[winner] += 1
                    
                    print("\n✅ Hand evaluated & clicked! Waiting 3 seconds for the next round...")
                    time.sleep(3.0)
                else:
                    time.sleep(0.5)
            else:
                if matches is not None and len(matches) > 0:
                    print(f"Stage 2 Incomplete: Paired {len(matches)}/9 cards. Scanning again...")
                time.sleep(0.5) 
            
            time.sleep(0.01)

if __name__ == "__main__":
    pyautogui.FAILSAFE = True 
    
    # Instantiate the API and bind the sys.stdout interceptor
    api = BotApi()
    sys.stdout = LoggerRedirect(api)
    
    print("Poker Vision UI Initialized.")
    
    # Start the OpenCV / YOLO bot loop as a Daemon Thread
    bot_thread = threading.Thread(target=run_bot_loop, args=(api,), daemon=True)
    bot_thread.start()

    # Create the standalone GUI window
    webview.create_window(
        title="Poker Vision Bot",
        html=HTML_CONTENT,
        js_api=api,
        width=540,
        height=660,
        resizable=False,
        background_color='#07070f'
    )
    
    # Start the UI loop and load the app_icon if supported
    icon_path = resource_path('app_icon.ico') if os.path.exists(resource_path('app_icon.ico')) else None
    webview.start(icon=icon_path)
    
    # Clean shutdown when the window is closed
    api.exit_flag = True