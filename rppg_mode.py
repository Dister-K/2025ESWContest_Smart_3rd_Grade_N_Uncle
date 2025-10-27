#!/usr/bin/env python
# coding: utf-8

# In[68]:


import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json, os, time, re, serial
from collections import deque
from datetime import datetime
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import subprocess, shutil, platform, threading, os
from pathlib import Path
import tempfile

time.sleep(6)

# --- TTS 폴더 (TTS/ 또는 tts/ 어느 쪽이든) ---
_BASE = Path(__file__).parent.resolve() if "__file__" in globals() else Path.cwd().resolve()
_TTS_DIR = (_BASE / "TTS")
if not _TTS_DIR.exists():
    _TTS_DIR = (_BASE / "tts")
_TTS_DIR = _TTS_DIR.resolve()

def first_existing(*names):
    for n in names:
        p = (_TTS_DIR / n).resolve()
        if p.exists():
            return str(p)
    return None

# 고정 멘트 파일 (있으면 사용, 없으면 자동 폴백)
AUDIO_END_PREFIX = first_existing("end_prefix.mp3")  # 예: "측정이 종료되었습니다. 평균 심박수"
AUDIO_END_SUFFIX = first_existing("end_suffix.mp3")  # 예: "입니다."
AUDIO_END_FAIL   = first_existing("end_fail.mp3")    # 예: 실패 멘트

def play_audio_file(path_str: str) -> bool:
    """블로킹 재생(겹침 방지). macOS면 afplay 우선, 그 외 playsound."""
    if not path_str or not os.path.exists(path_str):
        return False
    try:
        if shutil.which("afplay"):
            subprocess.run(["afplay", path_str], check=True)
            return True
        else:
            try:
                from playsound import playsound
                playsound(path_str)
                return True
            except Exception:
                pass
    except Exception as e:
        print(f"[AUDIO] play failed: {e}")
    return False

# macOS 'say'를 파일로 합성 → 블로킹 재생(순차 보장)
PREFER_SAY = (platform.system() == "Darwin" and shutil.which("say") is not None)
SAY_VOICE = "Yuna"   # macOS 한국어 기본 여성 음성 (남성 기본 없음)
SAY_RATE  = 190

def tts_say_blocking(text: str, *, voice: str | None = SAY_VOICE, rate: int | None = SAY_RATE) -> bool:
    if not text:
        return False
    if not PREFER_SAY:
        print("[TTS]", text)  # macOS가 아니면 프린트로 대체
        return False
    fd, tmp = tempfile.mkstemp(prefix="tts_", suffix=".aiff")
    os.close(fd)
    try:
        args = ["say", "-o", tmp]
        if voice: args += ["-v", voice]
        if rate:  args += ["-r", str(rate)]
        args.append(text)
        subprocess.run(args, check=True)   # 합성(블로킹)
        ok = play_audio_file(tmp)          # 재생(블로킹)
        return ok
    except Exception as e:
        print("[TTS FAIL]", e, "|", text)
        return False
    finally:
        try: os.remove(tmp)
        except: pass

# ===== 9가지 피부 케이스 파일 매핑 =====
# TTS/ 또는 tts/ 폴더에 아래 파일명을 넣으면 자동 사용됩니다.
# 파일이 없으면 동일 문장으로 TTS 폴백합니다.
SKIN9_FILES = {
    # 1~3: 붉음 + 균일도 Good/Mid/Low
    "red_good":  first_existing("rPPG_Red_even.mp3"),
    "red_mid":   first_existing("rPPG_Red_Slight_even.mp3"),
    "red_low":   first_existing("rPPG_Red_Not_even.mp3"),
    # 4~6: 정상 + 균일도 Good/Mid/Low  (정상일 때는 '정상' 언급 없이 균일도만 안내)
    "norm_good": first_existing("rPPG_Normal.mp3"),
    "norm_mid":  first_existing("rPPG_Normal_Slight_even.mp3"),
    "norm_low":  first_existing("rPPG_Normal_Not_even.mp3"),
    # 7~9: 창백 + 균일도 Good/Mid/Low
    "pale_good": first_existing("rPPG_Pale_even.mp3"),
    "pale_mid":  first_existing("rPPG_Pale_Slight_even.mp3"),
    "pale_low":  first_existing("rPPG_Pale_Not_even.mp3"),
}

# (선택) 안내 앞/뒤에 붙일 프리/서픽스 mp3 (있으면 재생)
SKIN_PREFIX = first_existing("skin_prefix.mp3")  # 예: "피부 상태 요약을 안내합니다."
SKIN_SUFFIX = first_existing("skin_suffix.mp3")

# 라벨 표준화 + 균일도 버킷
def _normalize_redness(label: str) -> str:
    if not label:
        return "정상"
    if label in ("홍조", "붉음", "Red"):
        return "붉음"
    if label in ("창백", "Pale"):
        return "창백"
    return "정상"

def _uniformity_bucket(u: float) -> str:
    # u는 0~1 (이미 코드에서 그렇게 산출)
    if u is None:
        return "불균일"
    if u >= 0.8:
        return "좋음"
    elif u >= 0.6:
        return "살짝 불균일"
    else:
        return "불균일"

def _skin_case_id(red: str, ub: str) -> int:
    table = {
        ("붉음","좋음"):1, ("붉음","살짝 불균일"):2, ("붉음","불균일"):3,
        ("정상","좋음"):4, ("정상","살짝 불균일"):5, ("정상","불균일"):6,
        ("창백","좋음"):7, ("창백","살짝 불균일"):8, ("창백","불균일"):9,
    }
    return table.get((red, ub), 6)  # 못맞추면 보수적으로 6(정상-불균일)

# 각 케이스별 TTS 폴백 문장 (파일 없을 때만 사용)
SKIN9_FALLBACK_TEXT = {
    1: "피부가 다소 붉습니다. 피부 자극이나 스트레스 상태일 수 있어요. 피부톤이 매우 균일합니다.",
    2: "피부가 다소 붉습니다. 피부 자극이나 스트레스 상태일 수 있어요. 피부톤이 살짝 불균일합니다.",
    3: "피부가 다소 붉습니다. 피부 자극이나 스트레스 상태일 수 있어요. 피부톤이 불균일합니다.",
    4: "피부톤이 매우 균일합니다.",
    5: "피부톤이 살짝 불균일합니다.",
    6: "피부톤이 불균일합니다.",
    7: "피부가 창백해 보입니다. 컨디션 저하나 수면 부족일 수 있어요. 피부톤이 매우 균일합니다.",
    8: "피부가 창백해 보입니다. 컨디션 저하나 수면 부족일 수 있어요. 피부톤이 살짝 불균일합니다.",
    9: "피부가 창백해 보입니다. 컨디션 저하나 수면 부족일 수 있어요. 피부톤이 불균일합니다.",
}

def _skin_file_key(case_id: int) -> str:
    return {
        1:"red_good", 2:"red_mid", 3:"red_low",
        4:"norm_good",5:"norm_mid",6:"norm_low",
        7:"pale_good",8:"pale_mid",9:"pale_low",
    }[case_id]

def speak_skin_case_9way(redness_label: str, uniformity_val: float):
    """9가지 조합 중 맞는 mp3를 재생(있으면), 없으면 동일 문장으로 TTS."""
    red = _normalize_redness(redness_label)
    ub  = _uniformity_bucket(uniformity_val)
    cid = _skin_case_id(red, ub)
    key = _skin_file_key(cid)
    # 프리픽스(있으면)
    if SKIN_PREFIX:
        play_audio_file(SKIN_PREFIX)
    # 본문 파일 → 없으면 폴백 TTS
    mp3 = SKIN9_FILES.get(key)
    if not mp3 or not play_audio_file(mp3):
        tts_say_blocking(SKIN9_FALLBACK_TEXT.get(cid, "피부 상태를 해석할 수 없습니다."))
    # 서픽스(있으면)
    if SKIN_SUFFIX:
        play_audio_file(SKIN_SUFFIX)

def print_skin_case_9way(redness_label: str, uniformity_val: float):
    """콘솔 출력도 9가지 규칙에 맞게."""
    red = _normalize_redness(redness_label)
    ub  = _uniformity_bucket(uniformity_val)

    # '정상'일 땐 균일도만 출력
    if red == "붉음":
        print("📛 피부가 다소 붉습니다. 피부 자극이나 스트레스 상태일 수 있어요.")
    elif red == "창백":
        print("⚠️ 피부가 창백해 보입니다. 컨디션 저하나 수면 부족일 수 있어요.")

    if ub == "좋음":
        print("🟢 피부톤이 매우 균일합니다. 좋은 컨디션입니다.")
    elif ub == "살짝 불균일":
        print("⚪ 피부톤이 살짝 불균일합니다. 약간의 피로나 건조함이 있을 수 있어요.")
    else:
        print("🎯 피부톤이 불균일합니다. 색소침착, 피로 누적 가능성이 있어요.")
            
# === 피부톤 트래커 (밝기/붉기/균일도/잡티 + z-score + EMA) ===
class SkinToneTracker:
    def __init__(self, baseline_sec=3, ema_alpha=0.3, fps_hint=30):
        self.N = int(baseline_sec * max(1, fps_hint))
        self.buf_b, self.buf_r = [], []
        self.mu = None
        self.sd = None
        self.a = ema_alpha
        self.sm = dict(b=None, r=None, u=None, bl=None)

    @staticmethod
    def _ema(prev, x, a):
        return x if prev is None else a*x + (1-a)*prev

    @staticmethod
    def _metrics_from_bgr_pixels(bgr_pixels):
        if bgr_pixels.size == 0:
            return 0.0, 1.0, 0.0, 0.0
        px = bgr_pixels.astype(np.float32)
        mb, mg, mr = px[:,0].mean(), px[:,1].mean(), px[:,2].mean()
        m = (mb+mg+mr)/3.0
        kb, kg, kr = m/(mb+1e-6), m/(mg+1e-6), m/(mr+1e-6)
        px[:,0]*=kb; px[:,1]*=kg; px[:,2]*=kr
        px = np.clip(px,0,255)
        B,G,R = px[:,0],px[:,1],px[:,2]
        V = np.max(px,axis=1)
        brightness = float(np.mean(V))
        redness = float((np.mean(R)+1e-6)/(np.mean(G)+1e-6))
        uniformity = float(1.0 - (np.std(V)/255.0))
        uniformity = np.clip(uniformity,0,1)
        thr = np.mean(V)-12.0
        blemish_ratio = float(np.mean((V<thr).astype(np.float32)))
        return brightness, redness, uniformity, blemish_ratio

    def update(self, bgr_pixels):
        b, r, u, bl = self._metrics_from_bgr_pixels(bgr_pixels)
        if self.mu is None:
            self.buf_b.append(b); self.buf_r.append(r)
            if len(self.buf_b) >= self.N:
                mu_b, sd_b = np.mean(self.buf_b), np.std(self.buf_b)+1e-6
                mu_r, sd_r = np.mean(self.buf_r), np.std(self.buf_r)+1e-6
                self.mu, self.sd = np.array([mu_b,mu_r]), np.array([sd_b,sd_r])
        if self.mu is not None:
            z_b = (b-self.mu[0])/self.sd[0]
            z_r = (r-self.mu[1])/self.sd[1]
        else: z_b=z_r=0.0
        self.sm["b"]=self._ema(self.sm["b"],b,self.a)
        self.sm["r"]=self._ema(self.sm["r"],r,self.a)
        self.sm["u"]=self._ema(self.sm["u"],u,self.a)
        self.sm["bl"]=self._ema(self.sm["bl"],bl,self.a)
        tone_lbl = "밝음" if z_b>0.7 else ("어두움" if z_b<-0.7 else "중간")
        red_lbl  = "홍조" if z_r>0.8 else ("창백" if z_r<-0.8 else "정상")
        uni_lbl  = "양호" if (self.sm["u"] or 0)>=0.8 else ("보통" if (self.sm["u"] or 0)>=0.6 else "불균일")
        return dict(brightness=self.sm["b"] or 0.0, redness=self.sm["r"] or 1.0,
                    uniformity=self.sm["u"] or 0.0, blemish=self.sm["bl"] or 0.0,
                    z_bright=z_b, z_red=z_r), dict(tone=tone_lbl, redness=red_lbl, uniformity=uni_lbl)

# === POS 기반 rPPG 신호 생성 + 필터 ===
def calculate_pos_signal(rgb_values, wG=1.2):
    C = np.asarray(rgb_values,dtype=np.float32)
    mean_rgb = np.mean(C,axis=0)+1e-6
    Cn = C/mean_rgb - 1.0
    S1 = Cn[:,1]-Cn[:,2]
    S2 = Cn[:,1]+Cn[:,2]-2.0*Cn[:,0]
    alpha = np.std(S1)/(np.std(S2)+1e-6)
    return (S1+alpha*S2) - np.mean(S1+alpha*S2)

def bandpass_filter(signal, fs, lowcut=0.7, highcut=3.0, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def speak_skin_feedback_by_files(redness_label: str, uniformity_val: float):
    # 0) 인트로(있으면)
    if SKIN_FILES.get("intro"):
        play_audio_file(SKIN_FILES["intro"])

    # 1) 붉기 파트
    if redness_label == "붉음":
        if not play_audio_file(SKIN_FILES.get("red_high")):
            tts_say_blocking("피부가 다소 붉습니다. 피부 자극이나 스트레스 상태일 수 있어요.")
    elif redness_label == "창백":
        if not play_audio_file(SKIN_FILES.get("red_pale")):
            tts_say_blocking("피부가 창백해 보입니다. 컨디션 저하나 수면 부족일 수 있어요.")
    # '정상'이면 생략(원하면 여기서 추가 가능)

    # 2) 균일도 파트
    if uniformity_val >= 0.8:
        if not play_audio_file(SKIN_FILES.get("uni_good")):
            tts_say_blocking("피부톤이 매우 균일합니다. 좋은 컨디션입니다.")
    elif 0.6 <= uniformity_val < 0.8:
        if not play_audio_file(SKIN_FILES.get("uni_mid")):
            tts_say_blocking("피부톤이 살짝 균일하지 못합니다. 약간의 피로나 건조함이 있을 수 있어요.")
    else:
        if not play_audio_file(SKIN_FILES.get("uni_low")):
            tts_say_blocking("피부톤이 고르지 못해요. 색소침착, 피로 누적 가능성이 있어요.")

# === 오디오/보이스 진단 ===
print("[DEBUG] _TTS_DIR:", _TTS_DIR)
print("[DEBUG] end_prefix:", AUDIO_END_PREFIX, "exists:", os.path.exists(AUDIO_END_PREFIX) if AUDIO_END_PREFIX else None)
print("[DEBUG] end_suffix:", AUDIO_END_SUFFIX, "exists:", os.path.exists(AUDIO_END_SUFFIX) if AUDIO_END_SUFFIX else None)
print("[DEBUG] end_fail  :", AUDIO_END_FAIL,   "exists:", os.path.exists(AUDIO_END_FAIL)   if AUDIO_END_FAIL   else None)

print("[DEBUG] macOS?:", platform.system() == "Darwin")
print("[DEBUG] afplay:", shutil.which("afplay") is not None)
print("[DEBUG] say   :", shutil.which("say") is not None)

def _list_korean_voices():
    if shutil.which("say") is None: 
        return []
    try:
        out = subprocess.run(["say","-v","?"], capture_output=True, text=True)
        lines = out.stdout.splitlines()
        return [ln for ln in lines if ("Korean" in ln or "ko_KR" in ln)]
    except Exception as e:
        print("[DEBUG] say -v ? error:", e)
        return []


# In[69]:


# === Baseline 저장/불러오기 + 비교/진단 유틸 ===
BASELINE_DIR = "baselines"
os.makedirs(BASELINE_DIR, exist_ok=True)

def save_baseline(user_id, baseline_data):
    path = os.path.join(BASELINE_DIR, f"{user_id}.json")
    with open(path,"w",encoding="utf-8") as f:
        json.dump(baseline_data,f,ensure_ascii=False,indent=2)
    print(f"[저장 완료] {path}")

def load_baseline(user_id):
    path = os.path.join(BASELINE_DIR, f"{user_id}.json")
    if not os.path.exists(path):
        print(f"[경고] Baseline not found for {user_id}")
        return None
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

def compare_to_baseline(current, baseline):
    comp={}
    for k in ["bpm","sdnn","rmssd","brightness","redness","uniformity"]:
        b0, c0 = baseline.get(k), current.get(k)
        if b0 is None or c0 is None: continue
        if k=="bpm":
            z=(c0-b0)/(0.1*b0+1e-6)
            comp["bpm_z"]=z
        else:
            comp[f"{k}_delta"]=(c0-b0)/(b0+1e-6)
    return comp

def diagnose_state(comp):
    bpm_z = comp.get("bpm_z", 0)
    brightness_delta = comp.get("brightness_delta", 0)
    redness_delta = comp.get("redness_delta", 0)
    uniformity_delta = comp.get("uniformity_delta", 0)

    # 심박수가 많이 증가하고 피부 밝기가 감소하면 스트레스 가능성
    if bpm_z > 1.5 and brightness_delta < -0.15:
        return "⚠️ 스트레스 또는 혈류 저하 가능"
    # 붉기 상승 + 균일도 저하 시 피부 긴장 or 순환 문제
    elif redness_delta > 0.2 and uniformity_delta < -0.15:
        return "⚠️ 피부톤 변화 감지됨"
    elif bpm_z < -1.5:
        return "⚠️ 비정상적으로 느린 맥박"
    else:
        return "🟢 안정된 상태"

# ----- 피부 피드백 함수  -----
def skin_feedback(tone, redness_label, uniformity_val):
    feedback = []

    #if tone == "밝음":
        #feedback.append("☀️ 피부가 밝은 상태입니다. 실내 활동이 많거나 피부가 건조할 수 있어요.")     보류
    #elif tone == "중간":
        #feedback.append("✅ 피부 밝기는 정상 범위입니다. 현재 상태가 안정적이에요.")
    #elif tone == "어두움":
        #feedback.append("🌞 피부가 어두운 편이에요. 햇빛 노출이 많거나 피로 누적일 수 있어요.")

    if redness_label == "붉음":
        feedback.append("📛 피부가 다소 붉습니다. 피부 자극이나 스트레스 상태일 수 있어요.")
    elif redness_label == "창백":
        feedback.append("⚠️ 피부가 창백해 보입니다. 컨디션 저하나 수면 부족일 수 있어요.")

    if uniformity_val >= 0.8:
        feedback.append("🟢 피부톤이 매우 균일합니다. 좋은 컨디션입니다.")
    elif 0.6 <= uniformity_val < 0.8:
        feedback.append("⚪ 피부톤이 살짝 균일하지 못합니다. 약간의 피로나 건조함이 있을 수 있어요.")
    else:
        feedback.append("🎯 피부톤이 고르지 못해요. 색소침착, 피로 누적 가능성이 있어요.")

    return "\n".join(feedback)

def show_fft_spectrum(sig, fps):
    # POS 신호에 대한 FFT 계산
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0/fps)
    fft_vals = np.abs(np.fft.rfft(sig - np.mean(sig)))

    # BPM 범위만 시각화 (40~180BPM → 약 0.66~3Hz)
    bpm_freq_range = (freqs >= 0.6) & (freqs <= 3.0)
    bpm_freqs = freqs[bpm_freq_range]
    bpm_amplitudes = fft_vals[bpm_freq_range]

    # 주파수 → BPM 변환
    bpm_ticks = bpm_freqs * 60

    # 실시간 업데이트
    clear_output(wait=True)
    plt.figure(figsize=(8, 3))
    plt.plot(bpm_ticks, bpm_amplitudes, color='blue')
    plt.title("실시간 주파수 스펙트럼 (BPM 단위)")
    plt.xlabel("BPM")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[72]:


width = 1200
height = 960
fps_hint = 30
last_advice_text = "" 
last_skin_labels = None
last_skin_metrics = None
# ==== 실시간 카메라 측정 + 개인화 진단 통합 ====
cam_index = 0  # 외장 카메라
mirror = 1     # 좌우반전 ON
subject = "subject"  # 사용자 ID 고정
mode = 0       # 0: 진단모드 / 1: Baseline 저장

BPM_TTS_ENABLE         = True
BPM_TTS_COOLDOWN_SEC   = 10.0
BPM_TTS_DELTA_TO_SPEAK = 3.0
BPM_VALID_RANGE        = (40, 180)

_last_bpm_tts_time = 0.0
_last_bpm_spoken   = None

# baseline 불러오기 + None 방지 처리
baseline = load_baseline(subject) if mode == 0 else None
if baseline is None:
    print(f"[경고] Baseline not found for {subject}")
    baseline = {}

cap = cv2.VideoCapture(cam_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps_hint)
if not cap.isOpened(): raise RuntimeError("카메라 열기 실패")

skin_tracker = SkinToneTracker(baseline_sec=3, ema_alpha=0.3, fps_hint=fps_hint)
win_sec = 12
rgb_buf = deque(maxlen=fps_hint * win_sec)
t_buf = deque(maxlen=fps_hint * win_sec)
skin_log = []

mp_face_mesh = mp.solutions.face_mesh
LEFT_CHEEK_IDX = [111,117,118,119,120,121,47,126,209,49,203,206,216,214,192,213,177,137,227]
RIGHT_CHEEK_IDX = [340,346,347,348,349,350,277,355,429,279,423,426,436,434,416,433,401,366,447]


with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as fm:
    t0 = time.perf_counter(); frame_idx = 0; last_log_time = 0; last_bpm_est = None

    cv2.namedWindow("rPPG+Diagnosis", cv2.WINDOW_NORMAL)
    
    while True:
        ok, img = cap.read()
        if not ok:
            print("[경고] 프레임 읽기 실패, 0.5초 후 재시도...")
            time.sleep(0.5)
            continue
        if mirror: img = cv2.flip(img, 1)
        h, w, _ = img.shape
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask = np.zeros((h, w), dtype=np.uint8); cheek_pixels = None

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            lp = [(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in LEFT_CHEEK_IDX]
            rp = [(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in RIGHT_CHEEK_IDX]
            cv2.fillPoly(mask, [np.array(lp)], 255)
            cv2.fillPoly(mask, [np.array(rp)], 255)
            mb, mg, mr, _ = cv2.mean(img, mask=mask)
            rgb_buf.append([mr, mg, mb]); t_buf.append(time.perf_counter())
            cheek_pixels = cv2.bitwise_and(img, img, mask=mask)[mask == 255]
            cv2.polylines(img, [np.array(lp)], True, (0,255,0), 2)
            cv2.polylines(img, [np.array(rp)], True, (0,255,0), 2)

        now = time.perf_counter()
        if now - t0 > 60:
            print("[INFO] 1분 측정 완료. 결과를 출력합니다.")
            break
        if cheek_pixels is not None and cheek_pixels.size:
            skin_metrics, skin_labels = skin_tracker.update(cheek_pixels)
            current_metrics = {
                "bpm": last_bpm_est,
                "brightness": skin_metrics["brightness"],
                "redness": skin_metrics["redness"],
                "uniformity": skin_metrics["uniformity"]
            }
            advice_text = skin_feedback(
                tone=skin_labels["tone"],
                redness_label=skin_labels["redness"],
                uniformity_val=skin_metrics["uniformity"]
            )
            last_advice_text = advice_text
            last_skin_labels = skin_labels
            last_skin_metrics = skin_metrics

            if mode == 0 and baseline:
                comp = compare_to_baseline(current_metrics, baseline)
                diagnosis = diagnose_state(comp)
            else:
                diagnosis = "기준 저장 중"

            if now - last_log_time >= 1.0:
                last_log_time = now
                print(f"[{int(now - t0)}s] Tone:{skin_labels['tone']} Red:{skin_labels['redness']} BPM:{last_bpm_est}")
                print(f" >> 상태 진단: {diagnosis}")  # ✅ 이거 추가!
                skin_log.append({
                    "time_sec": now - t0, "bpm": last_bpm_est,
                    "brightness": skin_metrics["brightness"],
                    "redness": skin_metrics["redness"],
                    "uniformity": skin_metrics["uniformity"],
                    "diagnosis": diagnosis
                })

            cv2.putText(img, f"State: {diagnosis}", (12, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(img, f"State: {diagnosis}", (12, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if len(rgb_buf) >= int(fps_hint * 8):
            t_arr = np.array(t_buf)
            fs = 1.0 / np.median(np.diff(t_arr))
            pos = calculate_pos_signal(np.array(rgb_buf))
            fpos = bandpass_filter(pos, fs=fs, lowcut=0.7, highcut=3.0)
            freqs = np.fft.rfftfreq(len(fpos), 1 / fs)
            mag = np.abs(np.fft.rfft(fpos))
            valid = (freqs >= 0.8) & (freqs <= 3.0)
            if np.any(valid):
                peak = freqs[valid][np.argmax(mag[valid])]
                new_bpm = peak * 60

                # ==== 튐 방지 조건 추가 ====
                if last_bpm_est is not None:
                    if abs(new_bpm - last_bpm_est) > 0.25 * last_bpm_est:
                        new_bpm = last_bpm_est  # 변화 폭 너무 크면 이전 BPM 유지
                last_bpm_est = new_bpm
               
        cv2.imshow("rPPG+Diagnosis", img)
        if cv2.waitKey(1) & 0xFF == 27: break


    cap.release()
    cv2.destroyWindow("rPPG+Diagnosis")
    cv2.waitKey(100)

# ---- 종료 후 baseline 저장 ----
if mode == 1 and len(skin_log) > 0:
    avg_vals = pd.DataFrame(skin_log).mean(numeric_only=True).to_dict()
    baseline_data = dict(
        bpm=avg_vals.get("bpm", np.nan),
        brightness=avg_vals.get("brightness", 0),
        redness=avg_vals.get("redness", 1),
        uniformity=avg_vals.get("uniformity", 0)
    )
    save_baseline(subject, baseline_data)
    print(f"[저장 완료] baselines/{subject}.json")

# ---- 종료 요약 ----
if len(rgb_buf) >= int(fps_hint * 10):
    df = pd.DataFrame(skin_log)
    print("\n[측정 요약 결과]")
    print(f" - 평균 BPM        : {df['bpm'].mean():.2f}")
    print(f" - 평균 밝기        : {df['brightness'].mean():.3f}")
    print(f" - 평균 붉기        : {df['redness'].mean():.3f}")
    print(f" - 평균 균일도      : {df['uniformity'].mean():.3f}")

    final_bpm = None
    if 'bpm' in df and df['bpm'].notna().sum() > 0:
        final_bpm = float(df['bpm'].mean())
    elif last_bpm_est is not None:
        final_bpm = float(last_bpm_est)

    if final_bpm is not None and 40 <= final_bpm <= 180:
        ok1 = play_audio_file(AUDIO_END_PREFIX) if AUDIO_END_PREFIX else False
        
        num_spoken = tts_say_blocking(f"{int(round(final_bpm))}")  # 숫자만 TTS
        if not num_spoken:
            print("[WARN] 숫자 TTS 실패. (보이스 미설치/권한 문제일 수 있음)")
        
        ok3 = play_audio_file(AUDIO_END_SUFFIX) if AUDIO_END_SUFFIX else False
        
        # 최소한 뭔가는 나왔는지 확인 (모두 실패라면 마지막 폴백)
        if not (ok1 or num_spoken or ok3):
            # 'say'가 아예 불가면 이 폴백도 실패할 수 있으니, 여기서는 mp3만 다시 시도하거나 로그만 남기도록 선택
            print("[FALLBACK] 오디오 재생 전부 실패. 파일명/경로/보이스를 확인하세요.")
    else:
        if not play_audio_file(AUDIO_END_FAIL):
            tts_say_blocking("측정이 종료되었습니다. 유효한 심박수 데이터를 얻지 못했습니다.")
            
    # ==== 피부 피드백: 9가지 케이스 규칙으로 출력 + 파일우선(TTS폴백) 안내 ====
    if last_skin_labels and last_skin_metrics:
        print("\n[피부 피드백]")
        print_skin_case_9way(
            redness_label=last_skin_labels["redness"],
            uniformity_val=float(last_skin_metrics["uniformity"] or 0.0)
        )
        # 음성 출력(파일 없으면 자동 TTS)
        speak_skin_case_9way(
            redness_label=last_skin_labels["redness"],
            uniformity_val=float(last_skin_metrics["uniformity"] or 0.0)
        )

    # ✅ 최종 주파수 스펙트럼 (심박수 추정용)
    print("\n[주파수 스펙트럼]")
    t_arr = np.array(t_buf)
    fs = 1.0 / np.median(np.diff(t_arr))
    pos = calculate_pos_signal(np.array(rgb_buf))
    fpos = bandpass_filter(pos, fs=fs, lowcut=0.7, highcut=3.0)
    freqs = np.fft.rfftfreq(len(fpos), 1 / fs)
    mag = np.abs(np.fft.rfft(fpos))
    valid = (freqs >= 0.8) & (freqs <= 3.0)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs[valid], mag[valid], color='tab:blue')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("d")
    plt.title("Spectrum")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
else:
    print("\n[알림] 수집된 데이터가 10초 미만이라 결과 요약을 생략합니다.")

