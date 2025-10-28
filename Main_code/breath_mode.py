#!/usr/bin/env python
# coding: utf-8

# In[31]:


# breath_mode.py  (dual_reader_v2.py 대체)
# 결과 시점 LED+음성 출력 / 겹침 방지 / 파일 없이 삡! 생성 / macOS afplay 우선
# -*- coding: utf-8 -*-

import re, time, math, threading, queue, collections, statistics, sys, os, platform, shutil, struct, wave
from pathlib import Path

# =========================
# (1) 오디오: afplay 우선 + playsound 폴백 + 삡! 톤(파일 없이 생성)
# =========================
try:
    from playsound import playsound
except Exception as e:
    print("[WARN] playsound import 실패(폴백 비활성):", e)
    playsound = None

# __file__ 기준 절대경로 (터미널/서브프로세스/Jupyter 모두 대응)
try:
    _BASE = Path(__file__).parent.resolve()
except NameError:
    _BASE = Path.cwd().resolve()

_TTS = (_BASE / "tts").resolve()

def _first_existing(*names):
    for n in names:
        p = (_TTS / n).resolve()
        if p.exists():
            return str(p)
    return None

AUDIO_FILES = {
    "normal":  _first_existing("normal_tts.mp3", "normal.mp3", "_normal_tts.mp3"),
    "smoking": _first_existing("smoking_tts.mp3", "smoking.mp3"),
    "alcohol": _first_existing("alcohol_tts.mp3", "Alcohol_tts.mp3", "alchol_tts.mp3"),
    "both":    _first_existing("smoking_and_alcohol_tts.mp3", "both.mp3"),
    "warn":    _first_existing("uncertain.mp3", "warn_tts.mp3"),
}

# 디버그: 찾은 파일 보여주기
print("[AUDIO MAP]")
for k, v in AUDIO_FILES.items():
    print(f"  {k}: {v if v else 'NONE'}")

PREFER_AFPLAY = (platform.system() == "Darwin" and shutil.which("afplay"))
AFPLAY_VOLUME = None  # 예: 0.8 (None이면 시스템 기본 볼륨)

_last_audio_token = None
_last_audio_ts = 0.0
AUDIO_ENABLE = True
AUDIO_COOLDOWN_SEC = 8.0  # 같은 결과 반복재생 방지
_audio_lock = threading.Lock()

def _play_with_afplay(path_str: str):
    args = ["afplay"]
    if AFPLAY_VOLUME is not None:
        args += ["-v", str(AFPLAY_VOLUME)]
    args.append(path_str)
    # 블로킹 실행(겹침 방지 목적). 필요하면 별도 쓰레드 사용.
    try:
        subprocess = __import__("subprocess")
        subprocess.run(args, check=True)
        return True
    except Exception as e:
        print(f"[AUDIO] afplay 실패: {path_str} | {e}")
        return False

def _play_with_playsound(path_str: str):
    if not playsound:
        return False
    try:
        playsound(path_str)
        return True
    except Exception as e:
        print(f"[AUDIO] playsound 실패: {path_str} | {e}")
        return False

def _play_file_blocking(path_str: str):
    # afplay 우선 → 실패 시 playsound
    if PREFER_AFPLAY:
        if _play_with_afplay(path_str):
            return
        _play_with_playsound(path_str)
    else:
        if not _play_with_playsound(path_str):
            _play_with_afplay(path_str)

def _playsound_async(path_str: str):
    def _worker():
        _play_file_blocking(path_str)
    th = threading.Thread(target=_worker, daemon=True)
    th.start()

def play_audio(filepath: str, *, token: str = None):
    """
    결과 음성 재생(비동기). token/쿨다운으로 겹침 방지.
    """
    global _last_audio_token, _last_audio_ts
    if not AUDIO_ENABLE or not filepath:
        return
    p = Path(filepath)
    if not p.exists():
        print(f"[AUDIO] 파일 없음: {filepath}")
        return

    tnow = time.monotonic()
    with _audio_lock:
        if token is not None and _last_audio_token == token:
            return
        if (tnow - _last_audio_ts) < AUDIO_COOLDOWN_SEC:
            return
        _last_audio_token = token
        _last_audio_ts = tnow
    _playsound_async(str(p))

def beep_tone(duration=0.4, freq=850.0, volume=0.25, rate=44100):
    """
    파일 없이 삡! 소리 생성 후 임시 WAV로 재생 (afplay/playsound 공통 사용 가능).
    재생 후 임시 파일은 자동 삭제.
    """
    import tempfile, math
    nframes = int(rate * duration)
    # 16-bit PCM
    frames = bytearray()
    for i in range(nframes):
        t = i / rate
        # 간단한 attack/decay로 팝 제거
        env = 1.0
        a = min(0.02, duration*0.2)
        d = min(0.05, duration*0.3)
        if t < a:
            env = t / a
        elif t > (duration - d):
            env = max(0.0, (duration - t) / d)
        sample = int(max(-1.0, min(1.0, volume * env * math.sin(2.0*math.pi*freq*t))) * 32767)
        frames += struct.pack('<h', sample)

    with tempfile.NamedTemporaryFile(prefix="beep_", suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    # WAV write
    with wave.open(tmp_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(rate)
        wf.writeframes(frames)

    # 재생 비동기
    def _play_and_cleanup(pth):
        try:
            _play_file_blocking(pth)
        finally:
            try:
                os.remove(pth)
            except Exception:
                pass
    threading.Thread(target=_play_and_cleanup, args=(tmp_path,), daemon=True).start()

# =========================
# (2) 알코올 의심 간단 감지
# =========================
ALC_RATIO_PEAK_MIN = 1.25  
ALC_RATIO_AUC_MIN  = 1.35  
ALC_RH_DELTA_MAX   = 6.0 

def classify_alcohol(*, ratio_peak, ratio_auc, rh_delta):
    """
    알코올 의심 판단:
    - 에탄올 지표(ETH/ACE 비율 또는 AUC)가 완화된 기준 이상일 때
    - 습도 변화량(RH delta)이 너무 크지 않을 때
    """

    # (1) 기준 완화 (기존 값보다 낮게 설정)
    ALC_RATIO_PEAK_MIN = 1.15
    ALC_RATIO_AUC_MIN = 1.00
    ALC_RH_DELTA_MAX  = 5.0

    # (2) ETH 지표가 우세한지 확인 (peak 또는 AUC 중 하나라도 기준 이상)
    strong_eth = (
        (ratio_peak is not None and ratio_peak >= ALC_RATIO_PEAK_MIN) or
        (ratio_auc  is not None and ratio_auc  >= ALC_RATIO_AUC_MIN)
    )

    # (3) RH 변화량이 너무 크지 않은지
    rh_ok = True if rh_delta is None else (rh_delta <= ALC_RH_DELTA_MAX)

    # (4) 최종 판단
    alcohol_flag = bool(strong_eth and rh_ok)

    # (5) 디버그 로그
    peak_str = f"{ratio_peak:.2f}" if ratio_peak is not None else "None"
    auc_str  = f"{ratio_auc:.2f}" if ratio_auc is not None else "None"
    rh_str   = f"{rh_delta:.2f}" if rh_delta is not None else "None"
    print(f"[ALC_CHECK] ratio_peak={peak_str}, ratio_auc={auc_str}, RHΔ={rh_str}, flag={alcohol_flag}")

    return alcohol_flag

# =========================
# (3) LED 시리얼 설정 
# =========================
PORT_LED = "/dev/cu.usbmodem11401" 
BAUD_LED = 9600
LED_SEND_MIN_INTERVAL = 0.20
LED_AUTO_ON = True

LED_RESULT_FLASH_SEC = 10.0
from serial.tools import list_ports
print("[INFO] Available ports:")
for p in list_ports.comports():
    print(" ", p.device, "-", p.description or "", "-", p.manufacturer or "")

try:
    import serial  # pip install pyserial
except ImportError:
    print("pyserial이 필요합니다: pip install pyserial")
    sys.exit(1)

# =========================
# (5) CONFIG (네가 준 mac 포트)
# =========================
PORT_VOC = "/dev/cu.usbmodem11101"    # BME688
PORT_ACE = "/dev/cu.usbmodem1401"    # MICS-6814
BAUD_VOC = 9600
BAUD_ACE = 9600

PRINT_HZ      = 2.0
DEBUG_SPIKES  = False
SENSOR_WATCHDOG_SEC = 3.5
WARMUP_SEC    = 25.0
BASELINE_SEC  = 30.0

EMA_ALPHA_WAIT   = 0.02
EMA_ALPHA_EXHALE = 0.0
ROLLWIN_SEC      = 1.0

MIN_OHM              = 1.0
ACE_VALID_OHM_MIN    = 1000.0
ACE_VALID_OHM_MAX    = 800000.0

VOC_LOG_DROP_START   = 0.015
VOC_LOG_SLOPE_START  = 0.020
VOC_Z_START          = -2.0
HUM_RISE_ABS         = 0.70
TEMP_RISE_ABS        = 0.15

ACE_IDX_START        = 1.60
TRIG_HOLD_MS         = 300

VOC_LOG_DROP_REL     = 0.005
VOC_Z_RELAX          = -0.3
ACE_IDX_REL          = 1.08
REL_HOLD_MS          = 500

EXHALE_MIN_SEC       = 0.40
EXHALE_MAX_SEC       = 8.0
REFRACTORY_SEC       = 3.0

BURST_MULTIPLIER     = 1.5
ETH_DOMINANCE_RATIO  = 1.4

SMK_ETH_PEAK_MIN     = 1.35
SMK_ETH_AUC_MIN      = 0.30
SMK_DOM_PEAK_MIN     = 1.20 
SMK_DOM_AUC_MIN      = 1.25
SMK_SLOPE_MIN        = 0.25
SMK_EXHALE_MIN       = 0.70
SMK_EXCLUDE_ACE_DOM_RATIO = 1.0/ETH_DOMINANCE_RATIO

# === 한 번만 측정하고 끝낼지 여부 ===
EXIT_AFTER_FIRST_EXHALE = True      # True면 EXHALE 결과 후 종료
AUDIO_GRACE_BEFORE_EXIT_SEC = 0.0   # 음성 끝까지 잠깐 기다리려면 초 단위로 지정(예: 1.0)

# =========================
# (6) 유틸
# =========================
def fmt_dur(sec: float) -> str:
    sec = int(round(sec)); m, s = divmod(sec, 60)
    return (f"{m}분 {s}초" if m else f"{s}초")

def now(): return time.monotonic()

def fmt_kohm(ohm):
    if ohm is None: return "--"
    return f"{ohm/1000.0:.2f} KOhms"

def safe_float(x, default=None):
    try: return float(x)
    except: return default

def ohm_is_valid(v: float) -> bool:
    if v is None: return False
    return (ACE_VALID_OHM_MIN <= v <= ACE_VALID_OHM_MAX)

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def scale_linear(x, x0: float, x1: float) -> float:
    if x is None or x0 == x1: return 0.0
    t = (x - x0) / (x1 - x0)
    return clamp(t, 0.0, 1.0)

def tri_score(x, low: float, ideal: float, high: float) -> float:
    if x is None: return 0.0
    if x <= low or x >= high: return 0.0
    if x == ideal: return 1.0
    if x < ideal:  return (x - low) / (ideal - low)
    return (high - x) / (high - ideal)

def fmt_ago(tnow, last_ts):
    if not last_ts: return "NA"
    return f"{max(0.0, tnow - last_ts):.2f}s ago"

def sensor_stat(tnow, last_ts):
    if not last_ts: return "BOOT"
    return "OK" if (tnow - last_ts) <= SENSOR_WATCHDOG_SEC else "TIMEOUT"

class RollingSlope:
    def __init__(self, window_s=1.0):
        self.w = window_s
        self.deq = collections.deque()
    def push(self, ts, val):
        self.deq.append((ts, val))
        while self.deq and (ts - self.deq[0][0] > self.w):
            self.deq.popleft()
    def slope_per_s(self):
        if len(self.deq) < 2: return 0.0
        t0, v0 = self.deq[0]; t1, v1 = self.deq[-1]
        dt = max(1e-6, t1 - t0)
        return (v1 - v0) / dt

class EMA:
    def __init__(self, alpha, init=None):
        self.a = alpha
        self.v = init
    def update(self, x):
        if x is None: return self.v
        if self.a <= 0.0:
            return self.v if self.v is not None else x
        self.v = x if self.v is None else (self.a * x + (1 - self.a) * self.v)
        return self.v

class HoldGate:
    def __init__(self, hold_ms):
        self.hold = hold_ms / 1000.0
        self.t0 = None
    def tick(self, cond: bool, tnow: float) -> bool:
        if cond:
            if self.t0 is None:
                self.t0 = tnow
            return (tnow - self.t0) >= self.hold
        else:
            self.t0 = None
            return False
    def reset(self):
        self.t0 = None

# =========================
# 시리얼 리더 & 파서
# =========================
class SerialReader(threading.Thread):
    def __init__(self, port, baud, out_q, tag):
        super().__init__(daemon=True)
        self.port = port; self.baud = baud; self.tag = tag
        self.q = out_q; self.keep = True; self.ser = None
    def run(self):
        while self.keep:
            try:
                if self.ser is None:
                    self.ser = serial.Serial(self.port, self.baud, timeout=1)
                    self.q.put((self.tag, 'PORT', f"open: {self.port} @ {self.baud}"))
                line = self.ser.readline().decode(errors='ignore').strip()
                if line:
                    self.q.put((self.tag, 'LINE', line))
            except Exception as e:
                self.q.put((self.tag, 'ERR', f"{e}"))
                try:
                    if self.ser: self.ser.close()
                except: pass
                self.ser = None
                time.sleep(1.0)
    def stop(self):
        self.keep = False
        try:
            if self.ser: self.ser.close()
        except: pass

class VOCParser:
    def __init__(self):
        self.cur = {'T':None,'P':None,'RH':None,'Rgas':None}
        self.last_update = None
    def feed(self, line):
        if "Temperature" in line:
            v = re.findall(r"([-+]?\d+\.?\d*)", line)
            if v: self.cur['T'] = float(v[0])
        elif "Pressure" in line:
            v = re.findall(r"([-+]?\d+\.?\d*)", line)
            if v: self.cur['P'] = float(v[0])
        elif "Humidity" in line:
            v = re.findall(r"([-+]?\d+\.?\d*)", line)
            if v: self.cur['RH'] = float(v[0])
        elif "Gas" in line:
            vals = re.findall(r"([-+]?\d+\.?\d*)", line)
            if vals:
                val = float(vals[0])
                if "KOhm" in line or "KΩ" in line or "kOhm" in line:
                    self.cur['Rgas'] = val*1000.0
                else:
                    self.cur['Rgas'] = val if val>500 else val*1000.0
        else:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                t,p,rh,gas = map(safe_float, parts)
                if t is not None:  self.cur['T']=t
                if p is not None:  self.cur['P']=p
                if rh is not None: self.cur['RH']=rh
                if gas is not None:
                    self.cur['Rgas'] = gas*1000.0 if gas<500 else gas
        if any(v is not None for v in self.cur.values()):
            self.last_update = now()
    def snapshot(self):
        return dict(self.cur)

class ACEParser:
    """MICS-6814 저항값: 중앙값(0.5s) + EMA(α=0.2) 평활, 급변비 1.4 제한"""
    def __init__(self):
        self.co_ohm = None
        self.nh3_ohm = None
        self.last_update = None

        self._co_raw = None
        self._nh3_raw = None

        self.med_win_s = 0.5
        self.med_buf_co = collections.deque()
        self.med_buf_nh3 = collections.deque()

        self.ema_co = EMA(0.20, None)
        self.ema_nh3 = EMA(0.20, None)

        self._max_ratio = 1.4

    def _plausible(self, prev, cur):
        if prev is None or cur is None: return True
        r = cur / max(prev, 1e-9)
        return (1.0/self._max_ratio) <= r <= self._max_ratio

    def _push_med(self, buf, ts, val):
        buf.append((ts, val))
        # 창 범위 벗어난 값 제거
        while buf and (ts - buf[0][0] > self.med_win_s):
            buf.popleft()
        if not buf: return None
        return statistics.median([v for _, v in buf])

    def feed(self, line):
        if "RAW_ADC_RED" in line or "RAW_ADC_NH3" in line: return
        if line.lstrip().startswith("time_ms"): return

        parts = [p.strip() for p in line.split(",")]
        rsr = rsn = None
        if len(parts) >= 5:
            rsr = safe_float(parts[3]); rsn = safe_float(parts[4])
        elif len(parts) == 2:
            rsr = safe_float(parts[0]); rsn = safe_float(parts[1])

        t = now(); updated = False
        if ohm_is_valid(rsr) and (self._co_raw is None or self._plausible(self._co_raw, rsr)):
            self._co_raw = max(MIN_OHM, rsr)
            med = self._push_med(self.med_buf_co, t, self._co_raw)
            if med is not None:
                self.co_ohm = self.ema_co.update(med); updated = True
        if ohm_is_valid(rsn) and (self._nh3_raw is None or self._plausible(self._nh3_raw, rsn)):
            self._nh3_raw = max(MIN_OHM, rsn)
            med = self._push_med(self.med_buf_nh3, t, self._nh3_raw)
            if med is not None:
                self.nh3_ohm = self.ema_nh3.update(med); updated = True
        if updated:
            self.last_update = t

    def snapshot(self):
        return self.co_ohm, self.nh3_ohm

# =========================
# 흡연 판단 / 메트릭
# =========================
def classify_smoking(*, peak_ace, peak_eth, auc_ace, auc_eth,
                     ratio_peak, ratio_auc, s_idx_peak, s_voc_peak, exhale_dur):
    if ratio_peak <= SMK_EXCLUDE_ACE_DOM_RATIO and ratio_auc <= SMK_EXCLUDE_ACE_DOM_RATIO:
        return ("비의심", "ACE(≈NH3/수분) 우세")
    if exhale_dur < SMK_EXHALE_MIN:
        return ("모호", f"호기 시간 부족({exhale_dur:.2f}s<{SMK_EXHALE_MIN:.2f}s)")

    score = 0; reasons = []
    if peak_eth >= SMK_ETH_PEAK_MIN: score += 2; reasons.append(f"ETH_peak≥{SMK_ETH_PEAK_MIN:.2f}")
    if auc_eth  >= SMK_ETH_AUC_MIN:  score += 1; reasons.append(f"ETH_AUC≥{SMK_ETH_AUC_MIN:.2f}")
    if ratio_peak >= SMK_DOM_PEAK_MIN: score += 1; reasons.append(f"ETH/ACE_peak≥{SMK_DOM_PEAK_MIN:.2f}")
    if ratio_auc  >= SMK_DOM_AUC_MIN:  score += 1; reasons.append(f"ETH/ACE_AUC≥{SMK_DOM_AUC_MIN:.2f}")
    if s_idx_peak is not None and s_idx_peak >= SMK_SLOPE_MIN:
        score += 1; reasons.append(f"idx_slope_max≥{SMK_SLOPE_MIN:.2f}/s")

    if score >= 3:
        return ("흡연 의심", ", ".join(reasons) or "규칙충족")
    elif score == 2:
        return ("모호", ", ".join(reasons))
    else:
        return ("비의심", ", ".join(reasons) or "규칙미충족")

def compute_srs(*, ratio_peak, eth_peak, ratio_auc, idx_slope_max):
    eth_dom      = scale_linear(ratio_peak,    1.05, 1.35)
    eth_strength = scale_linear(eth_peak,      1.05, 1.45)
    slope_spike  = scale_linear(idx_slope_max, 0.05, 0.20)
    auc_dom      = scale_linear(ratio_auc,     1.15, 2.00)
    srs_raw = (0.4*eth_dom) + (0.3*eth_strength) + (0.2*slope_spike) + (0.1*auc_dom)
    srs = round(100.0 * clamp(srs_raw, 0.0, 1.0), 1)
    if srs >= 60: label = "의심"
    elif srs >= 30: label = "모호"
    else: label = "비의심"
    return srs, label

def compute_rqs(*, exhale_dur, rh_delta, idx_slope_max):
    dur_score   = tri_score(exhale_dur, low=3.0, ideal=6.0, high=10.0)
    rh_score    = scale_linear(rh_delta if rh_delta is not None else 0.0, 2.0, 6.0)
    slope_score = tri_score(idx_slope_max, low=0.02, ideal=0.06, high=0.12)
    rqs_raw = (0.4*dur_score) + (0.4*rh_score) + (0.2*slope_score)
    return round(100.0 * clamp(rqs_raw, 0.0, 1.0), 1)

# =========================
# LED 시리얼 헬퍼
# =========================
class LEDSerial:
    def __init__(self, port: str, baud: int, enabled: bool=True):
        self.port = port; self.baud = baud
        self.enabled = enabled and bool(port)
        self.ser = None; self.last_cmd = None; self.last_ts = 0.0
    def _ensure(self):
        if not self.enabled: return False
        if self.ser and self.ser.is_open: return True
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
            time.sleep(1.2)
            try: self.ser.reset_input_buffer()
            except: pass
            print(f"[LED] connected: {self.port} @ {self.baud}")
            return True
        except Exception as e:
            print(f"[LED] open fail: {e}")
            self.ser = None
            return False
    def send(self, cmd: str):
        if not self.enabled: return
        t = now()
        if cmd == self.last_cmd and (t - self.last_ts) < LED_SEND_MIN_INTERVAL:
            return
        if not self._ensure(): return
        try:
            self.ser.write((cmd.strip() + "\n").encode())
            self.last_cmd = cmd; self.last_ts = t
        except Exception as e:
            print(f"[LED] send fail: {e}")
            try:
                if self.ser: self.ser.close()
            except: pass
            self.ser = None

def led_show_state(led: 'LEDSerial', state: str, sensor_ok: bool=True):
    return

def led_show_timeout(led: 'LEDSerial'):
    return

def led_show_result(led: 'LEDSerial', final_label: str, smoke_label: str, alcohol_flag=False):
    print(f"[LED_DEBUG] smoke_label={smoke_label}, final_label={final_label}, alcohol_flag={alcohol_flag}")
    danger = (smoke_label == "흡연 의심") or (final_label in ("강한 비정상", "위험"))
    warn   = (smoke_label == "모호") or (final_label in ("약한 반응", "모호", "주의")) or alcohol_flag
    print(f"[LED_DEBUG] danger={danger}, warn={warn}")
    if danger:
        led.send("R")
    elif warn:
        led.send("Y")
    else:
        led.send("G")

def determine_result_label(smoke_label, final_label, alcohol_flag):
    """
    LED와 TTS 진단 기준을 통일하는 함수
    (흡연·알코올·모호·약한 반응까지 모두 동일한 기준으로 LED/TTS 일치시킴)
    """
    # 1️⃣ 위험 단계 (빨강)
    if smoke_label == "흡연 의심" and alcohol_flag:
        return ("흡연+알코올", "R", AUDIO_FILES.get("both"))
    elif smoke_label == "흡연 의심":
        return ("흡연", "R", AUDIO_FILES.get("smoking"))
    elif alcohol_flag:
        return ("알코올", "R", AUDIO_FILES.get("alcohol"))

    # 2️⃣ 주의 단계 (노랑)
    elif smoke_label == "모호" or final_label in ("약한 반응", "모호", "주의"):
        return ("주의", "Y", AUDIO_FILES.get("warn"))

    # 3️⃣ 정상 단계 (초록)
    else:
        return ("정상", "G", AUDIO_FILES.get("normal"))

# =========================
# 메인 상태머신
# =========================
def main():
    q = queue.Queue()
    voc_r = SerialReader(PORT_VOC, BAUD_VOC, q, "VOC")
    ace_r = SerialReader(PORT_ACE, BAUD_ACE, q, "ACE")
    voc_r.start(); ace_r.start()

    led = LEDSerial(PORT_LED, BAUD_LED, enabled=LED_AUTO_ON)
    if LED_AUTO_ON:
        led.send(f"HOLD {int(LED_RESULT_FLASH_SEC)}")

    print("[BOOT] starting dual reader...")
    print("[LEGEND] ACE DATA: CO_Rs_ohm=CO, NH3_Rs_ohm=NH3 (Ω)")
    print("[LEGEND] IDX: ACE_idx≈NH3, ETH_idx≈CO (1.00 baseline)")
    print(f"[INFO] Start (WARMUP {fmt_dur(WARMUP_SEC)} + BASELINE {fmt_dur(BASELINE_SEC)})")

    voc = VOCParser()
    ace = ACEParser()

    state = "WARMUP"
    state_ts = now()
    last_print = 0.0
    refractory_until = 0.0
    freeze_env_until = 0.0

    voc_log_baseline = EMA(EMA_ALPHA_WAIT, None)
    voc_raw_baseline = EMA(EMA_ALPHA_WAIT, None)
    hum_baseline     = EMA(EMA_ALPHA_WAIT, None)
    temp_baseline    = EMA(EMA_ALPHA_WAIT, None)

    co_baseline_ohm  = EMA(EMA_ALPHA_WAIT, None)
    nh3_baseline_ohm = EMA(EMA_ALPHA_WAIT, None)

    gas_log_window = collections.deque(maxlen=200)

    slope_voc_log = RollingSlope(ROLLWIN_SEC)
    slope_ace_idx = RollingSlope(1.2)

    hold_enter = HoldGate(TRIG_HOLD_MS)
    hold_exit  = HoldGate(REL_HOLD_MS)

    peak_ace_idx = 1.0
    peak_eth_idx = 1.0
    auc_ace      = 0.0
    auc_eth      = 0.0
    last_auc_t   = None
    exhale_ts    = None

    s_ace_idx_peak = 0.0
    s_voc_peak     = 0.0

    rh_pre = None
    rh_peak = None
    exhale_idx = 0

    idx_max_hist = collections.deque(maxlen=256)
    def idx_max_median(last_s=1.2):
        if not idx_max_hist: return None
        t_end = idx_max_hist[-1][0]
        vals = [v for (tt, v) in idx_max_hist if t_end - tt <= last_s]
        if not vals: return None
        vals.sort()
        return vals[len(vals)//2]

    DRIFT_ALPHA = 0.001
    def maybe_slow_drift_update(t, idx_med, s_idx_v, dRH, dTmp):
        if idx_med is None: return
        env_ok = (dRH is None or abs(dRH) < 0.20) and (dTmp is None or abs(dTmp) < 0.05)
        idx_ok = 0.90 <= idx_med <= 1.10 and (s_idx_v is None or abs(s_idx_v) < 0.02)
        if env_ok and idx_ok and t >= refractory_until:
            co_rs, nh3_rs = ace.snapshot()
            if co_rs and nh3_rs:
                co_baseline_ohm.a = DRIFT_ALPHA; nh3_baseline_ohm.a = DRIFT_ALPHA
                co_baseline_ohm.update(co_rs);   nh3_baseline_ohm.update(nh3_rs)
                co_baseline_ohm.a = EMA_ALPHA_WAIT; nh3_baseline_ohm.a = EMA_ALPHA_WAIT

    def compute_idxs(co_ohm, nh3_ohm, co_base, nh3_base):
        if not (co_ohm and nh3_ohm and co_base and nh3_base):
            return None, None, None
        ace_idx = nh3_base / max(MIN_OHM, nh3_ohm)
        eth_idx = co_base  / max(MIN_OHM, co_ohm)
        idx_max = max(ace_idx, eth_idx)
        return ace_idx, eth_idx, idx_max

    while True:
        try:
            try:
                tag, typ, payload = q.get(timeout=0.05)
            except queue.Empty:
                tag, typ, payload = None, None, None

            if typ == 'PORT':
                print(f"[PORT] {payload}")
            elif typ == 'LINE':
                if tag == "VOC": voc.feed(payload)
                elif tag == "ACE": ace.feed(payload)
            elif typ == 'ERR':
                print(f"[{tag}] ERROR: {payload}")

            t = now()
            voc_status = sensor_stat(t, voc.last_update)
            ace_status = sensor_stat(t, ace.last_update)

            snap_v = voc.snapshot()
            T = snap_v['T']; P = snap_v['P']; RH = snap_v['RH']; Rgas = snap_v['Rgas']
            Rgas_log = math.log(Rgas) if Rgas and Rgas > 0 else None

            co_rs, nh3_rs = ace.snapshot()

            if Rgas is not None:
                if Rgas_log is not None:
                    if voc_log_baseline.v is None and state != "WARMUP":
                        voc_log_baseline.v = Rgas_log
                    gas_log_window.append(Rgas_log)
                    slope_voc_log.push(t, Rgas_log)
                if voc_raw_baseline.v is None and state != "WARMUP":
                    voc_raw_baseline.v = Rgas

            if co_rs and nh3_rs and co_baseline_ohm.v is None and state != "WARMUP":
                co_baseline_ohm.v  = co_rs
                nh3_baseline_ohm.v = nh3_rs

            ace_idx, eth_idx, idx_max = compute_idxs(co_rs, nh3_rs, co_baseline_ohm.v, nh3_baseline_ohm.v)
            if idx_max:
                idx_max_hist.append((t, idx_max))
                med_for_slope = idx_max_median(1.2)
                if med_for_slope is not None:
                    slope_ace_idx.push(t, med_for_slope)

            elapsed_state = t - state_ts

            # --- 상태 전이 ---
            if state == "WARMUP":
                if elapsed_state >= WARMUP_SEC and Rgas and co_rs and nh3_rs:
                    state = "BASELINE"; state_ts = t
                    print("[STATE] BASELINE (센서 고정 공기에서 유지하세요)")

            elif state == "BASELINE":
                if Rgas is not None:         voc_raw_baseline.update(Rgas)
                if Rgas_log is not None:     voc_log_baseline.update(Rgas_log)
                if RH is not None:           hum_baseline.update(RH)
                if T is not None:            temp_baseline.update(T)

                if co_rs is not None:        co_baseline_ohm.update(co_rs)
                if nh3_rs is not None:       nh3_baseline_ohm.update(nh3_rs)

                if elapsed_state >= BASELINE_SEC and voc_log_baseline.v and co_baseline_ohm.v and nh3_baseline_ohm.v:
                    state = "WAIT"; state_ts = t
                    print(f"[STATE] WAIT (ready) | gas0={fmt_kohm(voc_raw_baseline.v)}, co0={int(co_baseline_ohm.v)} Ω, nh30={int(nh3_baseline_ohm.v)} Ω")

                    # ✨ 파일 없이 즉석 삡! (비동기)
                    try:
                        beep_tone(duration=0.55, freq=820.0, volume=0.22)
                        print("[BEEP] (generated) WAIT 상태 진입")
                    except Exception as e:
                        print(f"[BEEP FAIL] {e}")

            z_voc = None
            if len(gas_log_window) >= 20 and voc_log_baseline.v and (Rgas_log is not None):
                sd = statistics.pstdev(gas_log_window) or 1e-6
                z_voc = (Rgas_log - voc_log_baseline.v) / sd

            dgas_log = None
            if (Rgas_log is not None) and (voc_log_baseline.v is not None):
                dgas_log = voc_log_baseline.v - Rgas_log

            s_voc_log = -slope_voc_log.slope_per_s()
            s_ace_idx_now = slope_ace_idx.slope_per_s()

            dRH  = (RH - hum_baseline.v)   if (RH is not None and hum_baseline.v  is not None) else None
            dTmp = (T  - temp_baseline.v)  if (T  is not None and temp_baseline.v is not None) else None

            if state == "WAIT":
                if EMA_ALPHA_WAIT > 0:
                    if Rgas is not None:         voc_raw_baseline.update(Rgas)
                    if Rgas_log is not None:     voc_log_baseline.update(Rgas_log)
                    if t >= freeze_env_until:
                        if RH is not None: hum_baseline.update(RH)
                        if T is not None: temp_baseline.update(T)

                cond_voc = (
                    (dgas_log is not None and dgas_log >= VOC_LOG_DROP_START) or
                    (s_voc_log is not None and s_voc_log >= VOC_LOG_SLOPE_START) or
                    (z_voc is not None and z_voc <= VOC_Z_START)
                )
                cond_env = (
                    (dRH  is not None and dRH  >= HUM_RISE_ABS) or
                    (dTmp is not None and dTmp >= TEMP_RISE_ABS)
                )

                idx_max_med = idx_max_median(1.2)
                cond_ace_strong = (
                    (idx_max_med is not None and idx_max_med >= ACE_IDX_START) and
                    (s_ace_idx_now is not None and s_ace_idx_now >= 0.15)
                )

                maybe_slow_drift_update(t, idx_max_med, s_ace_idx_now, dRH, dTmp)
                arm = (t >= refractory_until)

                enter_cond = arm and (sensor_stat(t, voc.last_update)=="OK") and (sensor_stat(t, ace.last_update)=="OK") and cond_env and (cond_voc or cond_ace_strong)
                if not (sensor_stat(t, voc.last_update)=="OK" and sensor_stat(t, ace.last_update)=="OK"):
                    hold_enter.reset()

                if hold_enter.tick(enter_cond, t):
                    state = "EXHALE"; state_ts = t; exhale_ts = t
                    hold_exit.reset()
                    exhale_idx += 1
                    peak_ace_idx = 1.0; peak_eth_idx = 1.0
                    auc_ace = 0.0; auc_eth = 0.0; last_auc_t = t
                    s_ace_idx_peak = 0.0; s_voc_peak = 0.0
                    rh_pre = hum_baseline.v
                    rh_peak = RH if RH is not None else hum_baseline.v

            elif state == "EXHALE":
                if ace_idx: peak_ace_idx = max(peak_ace_idx, ace_idx)
                if eth_idx: peak_eth_idx = max(peak_eth_idx, eth_idx)
                if ace_idx and eth_idx and last_auc_t is not None:
                    dt = max(0.0, t - last_auc_t)
                    auc_ace += max(0.0, ace_idx - 1.0) * dt
                    auc_eth += max(0.0, eth_idx - 1.0) * dt
                    last_auc_t = t

                if s_ace_idx_now is not None:
                    s_ace_idx_peak = max(s_ace_idx_peak, s_ace_idx_now)
                if s_voc_log is not None:
                    s_voc_peak = max(s_voc_peak, s_voc_log)

                if RH is not None:
                    if rh_peak is None: rh_peak = RH
                    else: rh_peak = max(rh_peak, RH)

                rel_voc = ((dgas_log is not None and dgas_log <= VOC_LOG_DROP_REL) and (z_voc is None or z_voc >= VOC_Z_RELAX))
                rel_ace = ((idx_max is not None and idx_max <= ACE_IDX_REL) and (s_ace_idx_now is None or s_ace_idx_now <= 0.05))
                timeup  = (t - exhale_ts) >= EXHALE_MAX_SEC
                minok   = (t - exhale_ts) >= EXHALE_MIN_SEC

                exit_cond = (minok and (rel_voc and rel_ace)) or timeup
                if hold_exit.tick(exit_cond, t) or timeup:
                    exhale_dur = t - exhale_ts
                    ace_delta = max(0.0, peak_ace_idx - 1.0)
                    eth_delta = max(0.0, peak_eth_idx - 1.0)
                    ratio_peak = (peak_eth_idx / peak_ace_idx) if peak_ace_idx > 0 else float('inf')
                    ratio_auc  = (auc_eth / max(1e-6, auc_ace)) if auc_ace > 0 else float('inf')
                    ratio_any  = max(ratio_peak, ratio_auc)

                    final_label = "정상"
                    pattern_comment = "일반 호기 패턴 (수분/희석 우세)"
                    if ace_delta >= 1.0 or eth_delta >= 1.0:
                        final_label = "강한 비정상"
                        if ratio_any >= ETH_DOMINANCE_RATIO:
                            pattern_comment = f"CO/EtOH 우세 (AUC={ratio_auc:.2f}, peak={ratio_peak:.2f})"
                        elif ratio_any <= 1.0 / ETH_DOMINANCE_RATIO:
                            inv_auc = (auc_ace / max(1e-6, auc_eth)) if auc_eth > 0 else float('inf')
                            inv_peak= (peak_ace_idx / max(1e-6, peak_eth_idx)) if peak_eth_idx > 0 else float('inf')
                            pattern_comment = f"NH3/수분 우세 (AUC={inv_auc:.2f}, peak={inv_peak:.2f})"
                        else:
                            pattern_comment = "VOC 또는 복합 성분 반응"
                    elif ace_delta > 0.15 or eth_delta > 0.15:
                        final_label = "약한 반응"

                    print(f"[RESULT] {final_label} | ACEΔ={ace_delta:.2f}, ETHΔ={eth_delta:.2f} | peak(A/E)={peak_ace_idx:.2f}/{peak_eth_idx:.2f} AUC(A/E)={auc_ace:.2f}/{auc_eth:.2f}")
                    print(f"[PATTERN] {pattern_comment}")

                    smoke_label, smoke_reason = classify_smoking(
                        peak_ace=peak_ace_idx, peak_eth=peak_eth_idx,
                        auc_ace=auc_ace,     auc_eth=auc_eth,
                        ratio_peak=ratio_peak, ratio_auc=ratio_auc,
                        s_idx_peak=s_ace_idx_peak, s_voc_peak=s_voc_peak,
                        exhale_dur=exhale_dur
                    )
                    print(f"[SMOKE] {smoke_label} | dur={exhale_dur:.2f}s | ETH/ACE peak={ratio_peak:.2f}, AUC={ratio_auc:.2f} | idx_slope_max={s_ace_idx_peak:.2f}/s | {smoke_reason}")

                    rh_delta = None
                    if rh_peak is not None and rh_pre is not None:
                        rh_delta = rh_peak - rh_pre
                    srs, srs_label = compute_srs(ratio_peak=ratio_peak, eth_peak=peak_eth_idx, ratio_auc=ratio_auc, idx_slope_max=s_ace_idx_peak)
                    rqs = compute_rqs(exhale_dur=exhale_dur, rh_delta=rh_delta, idx_slope_max=s_ace_idx_peak)
                    rh_delta_txt = f"{rh_delta:.2f}" if rh_delta is not None else "NA"
                    print(f"[METRIC] EXHALE #{exhale_idx} -> SRS={srs} ({srs_label}), RQS={rqs} | dur={exhale_dur:.2f}s, RHΔ={rh_delta_txt}%, ETH/ACE peak={ratio_peak:.2f}, AUC={ratio_auc:.2f}, idx_slope_max={s_ace_idx_peak:.2f}/s")

                    # === LED + TTS 통합 진단 ===
                    try:
                        alcohol_flag = classify_alcohol(ratio_peak=ratio_peak, ratio_auc=ratio_auc, rh_delta=rh_delta)
                    except Exception:
                        alcohol_flag = False
                    
                    tts_label, led_code, tts_file = determine_result_label(smoke_label, final_label, alcohol_flag)
                    print(f"[RESULT_LOGIC] tts_label={tts_label}, led={led_code}, tts_file={tts_file}")
                    
                    # TTS 출력
                    if tts_file:
                        play_audio(tts_file, token=f"exhale#{exhale_idx}")
                    
                    # LED 출력
                    led.send(led_code)

                    if EXIT_AFTER_FIRST_EXHALE:
                        print("[EXIT] Single exhale complete — quitting this mode.")
                        if AUDIO_GRACE_BEFORE_EXIT_SEC > 0:
                            time.sleep(AUDIO_GRACE_BEFORE_EXIT_SEC)
                        break  # 메인 while 루프 탈출 → 아래 정리 코드 실행

                    state = "WAIT"; state_ts = t
                    refractory_until = t + REFRACTORY_SEC
                    freeze_env_until = refractory_until
                    hold_enter.reset()
                    print("[EVENT] EXHALE end")

            # --- 주기 출력 ---
            if t - last_print >= 1.0/PRINT_HZ:
                last_print = t
                print(f"[STATUS] VOC={voc_status} ({fmt_ago(t, voc.last_update)}), ACE={ace_status} ({fmt_ago(t, ace.last_update)})")
                if T is not None: print(f"[VOC] Temperature: {T:.2f} *C")
                if P is not None: print(f"[VOC] Pressure: {P:.2f} hPa")
                if RH is not None: print(f"[VOC] Humidity: {RH:.2f} %")
                if Rgas is not None: print(f"[VOC] Gas Resistance: {Rgas/1000.0:.2f} KOhms")

                st = state
                arm = "Y" if t >= refractory_until else "N"
                ref_left = max(0, refractory_until - t)
                ref_txt = f"{int(ref_left)}s" if arm=="N" else "N"

                ace_idx_txt = f"{ace_idx:.3f}" if 'ace_idx' in locals() and ace_idx else '--'
                eth_idx_txt = f"{eth_idx:.3f}" if 'eth_idx' in locals() and eth_idx else '--'

                idx_max_med_now = idx_max_median(1.2)
                settle = "N" if (idx_max_med_now is not None and (idx_max_med_now > 1.15 or idx_max_med_now < 0.85)) else "Y"

                print(
                    f"[ACE] CO_Rs={int(co_rs) if co_rs else '--'}Ω  "
                    f"NH3_Rs={int(nh3_rs) if nh3_rs else '--'}Ω  "
                    f"ACE_idx={ace_idx_txt}  ETH_idx={eth_idx_txt}  "
                    f"arm={arm} ref={ref_txt}  state={st} settle={settle}"
                )

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[CRIT ERR] {e}")

    voc_r.stop(); ace_r.stop()
    print("\n[INFO] Stopped by user")

if __name__ == "__main__":
    main()


# In[ ]:




