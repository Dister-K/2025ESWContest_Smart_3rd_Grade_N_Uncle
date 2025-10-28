#!/usr/bin/env python
# coding: utf-8

# In[34]:


# main_controller_with_tts.py
# 키패드 메인 컨트롤러 + TTS (중도 중지 지원, 비동기 큐) + 8번 강제 종료(모드 포함)
# macOS(M2) 기준. 필요 패키지: playsound==1.2.2, pyserial

import serial
import time
import re
import subprocess
import sys
import threading
import queue
from pathlib import Path
import platform
import shutil
import os
import signal

# =========================
# (0) 기본 설정
# =========================
PORT = '/dev/cu.usbserial-130'  # 환경에 맞게 수정
BAUD = 9600
started = False  # 시스템 시작 여부

PYTHON_EXE = sys.executable if sys.executable else "python3"

# 실행 중 모드 프로세스 관리
RUN_PROCS = {}                 # {'rppg': Popen, 'breath': Popen}
RUN_PROCS_LOCK = threading.Lock()

# =========================
# (1) TTS (afplay 우선 + playsound 폴백)
#  - 비동기 재생 큐(순차 재생)
#  - stop_audio()로 현재 재생 즉시 중단 + 큐 비우기
#  - 토큰+쿨다운으로 같은 이벤트 과다 재생 방지
# =========================
try:
    from playsound import playsound
except Exception:
    def playsound(*args, **kwargs):
        raise RuntimeError("playsound 사용 불가")

_BASE = Path(__file__).parent.resolve() if "__file__" in globals() else Path.cwd().resolve()
_TTS  = (_BASE / "tts").resolve()

# TTS 파일 매핑 (존재하지 않으면 자동으로 건너뜀)
TTS_FILES = {
    # 시스템
    "start":        _TTS / "start_tts.mp3",       # "시스템을 시작합니다"
    "start_guide":  _TTS / "start_guide.mp3",     # "측정 전 준비 안내..."
    "need_start":   _TTS / "need_start.mp3",      # "먼저 1번을 눌러 시작하세요"
    "stop":         _TTS / "switch_8_tts.mp3",    # "시스템을 종료합니다"
    # 모드
    "rppg_select":   _TTS / "switch_2_tts.mp3",
    "breath_select": _TTS / "switch_3_tts.mp3",
    "both_select":   _TTS / "switch_4_tts.mp3",
}

# 재생 제어 옵션
PREFER_AFPLAY   = True     # macOS에서 afplay 우선
AFPLAY_VOLUME   = None     # 예: 0.8 (80%). None이면 기본 볼륨
AUDIO_COOLDOWN_SEC = 2.0   # 같은 토큰 반복 재생 방지

# 재생 상태/큐
_audio_q = queue.Queue()
_audio_lock = threading.Lock()
_audio_proc = None  # 현재 재생 중인 afplay 프로세스 (중간중지 용)
_last_token = None
_last_ts = 0.0

def _clear_audio_queue():
    try:
        while True:
            _audio_q.get_nowait()
            _audio_q.task_done()
    except queue.Empty:
        pass

def stop_audio(clear_queue: bool = True):
    """현재 재생 즉시 중단. 필요시 대기열도 비움."""
    global _audio_proc
    with _audio_lock:
        if _audio_proc is not None and _audio_proc.poll() is None:
            try:
                _audio_proc.terminate()
                try:
                    _audio_proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    _audio_proc.kill()
            except Exception:
                pass
        _audio_proc = None
    if clear_queue:
        _clear_audio_queue()

def _play_with_afplay(p: str):
    args = ["afplay"]
    if AFPLAY_VOLUME is not None:
        args += ["-v", str(AFPLAY_VOLUME)]
    args.append(p)
    proc = subprocess.Popen(args)
    # 현재 실행 프로세스 기록 (stop_audio로 끊기 위함)
    global _audio_proc
    with _audio_lock:
        _audio_proc = proc
    proc.wait()
    with _audio_lock:
        _audio_proc = None

def _play_file(path: Path):
    """afplay(중단 가능) 우선, 안 되면 playsound(중단 불가)"""
    p = str(path)
    if PREFER_AFPLAY and platform.system() == "Darwin" and shutil.which("afplay"):
        try:
            _play_with_afplay(p)   # ✅ Popen으로 실행 → stop_audio로 중단 가능
            return
        except Exception as e:
            print(f"[AUDIO] afplay 실패: {path.name} | {e}")

    # 폴백: playsound (이 경로는 중간중지 불가)
    try:
        playsound(p)
        return
    except Exception as e1:
        print(f"[AUDIO] 재생 실패: {path.name} | {e1}")

def _audio_worker():
    while True:
        path = _audio_q.get()
        try:
            _play_file(path)
        finally:
            _audio_q.task_done()

threading.Thread(target=_audio_worker, daemon=True).start()

def enqueue_audio(files, *, token: str = None, cooldown: float = AUDIO_COOLDOWN_SEC):
    """
    파일(들)을 순차 재생 큐에 넣는다. (토큰/쿨다운 적용)
    - token: 이벤트 식별자(같은 token은 cooldown 내 재생 방지)
    """
    global _last_token, _last_ts
    now = time.monotonic()

    if token is not None:
        if _last_token == token and (now - _last_ts) < cooldown:
            return
        _last_token, _last_ts = token, now

    paths = files if isinstance(files, (list, tuple)) else [files]
    any_enqueued = False
    for p in map(Path, paths):
        if p.exists() and p.is_file():
            _audio_q.put(p)
            any_enqueued = True
        else:
            print(f"[AUDIO] 파일 없음(건너뜀): {p}")
    if not any_enqueued:
        print("[AUDIO] 재생할 파일이 없습니다 (모두 누락).")

# =========================
# (2) 모드 프로세스 제어
# =========================
def spawn_mode(name: str, cmd: list[str]):
    """모드 프로세스를 비동기로 실행하고 RUN_PROCS에 등록."""
    with RUN_PROCS_LOCK:
        # 같은 이름이 이미 돌고 있으면 먼저 종료
        if name in RUN_PROCS:
            stop_proc(name)
        # 새 세션으로 실행 → 필요 시 프로세스 그룹 종료 가능
        p = subprocess.Popen(cmd, start_new_session=True)
        RUN_PROCS[name] = p
        print(f"[PROC] {name} started (pid={p.pid})")

def stop_proc(name: str, grace: float = 0.7):
    """특정 모드 종료(SIGTERM→KILL)."""
    with RUN_PROCS_LOCK:
        p = RUN_PROCS.pop(name, None)
    if not p:
        return
    if p.poll() is None:
        try:
            # 우선 프로세스 그룹에 TERM (macOS/Unix)
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                p.terminate()
            try:
                p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    p.kill()
        except Exception as e:
            print(f"[PROC] stop {name} error: {e}")
    print(f"[PROC] {name} stopped")

def stop_all_modes():
    """실행 중인 모든 모드 종료."""
    with RUN_PROCS_LOCK:
        names = list(RUN_PROCS.keys())
    for n in names:
        stop_proc(n)

def any_mode_running() -> bool:
    with RUN_PROCS_LOCK:
        # 죽은 프로세스는 정리
        dead = [n for n, p in RUN_PROCS.items() if p.poll() is not None]
        for n in dead:
            RUN_PROCS.pop(n, None)
        return bool(RUN_PROCS)

# =========================
# (3) 모드 런처
#  - 키 입력 시마다 먼저 stop_audio(clear_queue=True)로 현재 재생 중단
#  - 8번: 모든 모드 종료 + 시스템 종료 음성
# =========================
def launch_mode(key: str):
    global started

    # 어떤 키를 누르든, 이전 음성은 즉시 끊고 큐 비움
    stop_audio(clear_queue=True)

    if key == '1':
        if started:
            print("[INFO] 시스템은 이미 시작되어 있습니다.")
        else:
            print("[1] 시스템 시작됨.")
            started = True
            enqueue_audio(TTS_FILES["start"], token="start")
            if TTS_FILES["start_guide"].exists():
                enqueue_audio(TTS_FILES["start_guide"], token="start_guide")

    elif key == '8':
        # 시스템 종료: 실행 중 모드가 있으면 모두 종료
        if any_mode_running():
            stop_all_modes()
        if started:
            print("[8] 시스템 종료됨. 대기 상태로 전환됩니다.")
            enqueue_audio(TTS_FILES["stop"], token="stop")
            started = False
        else:
            print("[INFO] 시스템이 이미 종료된 상태입니다.")
            enqueue_audio(TTS_FILES["already_on"], token="already_on")

    elif not started:
        print(f"[{key}] ⚠ 시스템이 꺼져 있습니다. 먼저 [1]번을 눌러 시작해주세요.")
        enqueue_audio(TTS_FILES["need_start"], token="need_start")

    elif key == '2':
        print("[2] rPPG 모드 실행")
        # 다른 모드가 돌고 있으면 충돌 방지 위해 모두 정리 후 시작
        if any_mode_running():
            stop_all_modes()
        enqueue_audio(TTS_FILES["rppg_select"], token="rppg")
        spawn_mode('rppg', [PYTHON_EXE, "rppg_mode.py"])

    elif key == '3':
        print("[3] 호흡 건강 진단 모드 실행")
        if any_mode_running():
            stop_all_modes()
        enqueue_audio(TTS_FILES["breath_select"], token="breath")
        spawn_mode('breath', [PYTHON_EXE, "breath_mode.py"])

    elif key == '4':
        print("[4] rPPG + 호흡 모드 동시 실행")
        if any_mode_running():
            stop_all_modes()
    
        enqueue_audio(TTS_FILES["both_select"], token="both")  # "동시 모드를 시작합니다"
        
        spawn_mode('rppg', [PYTHON_EXE, "rppg_mode.py"])
    
        # rPPG 결과 TTS가 끝나기까지 대기
        time.sleep(38)
    
        spawn_mode('breath', [PYTHON_EXE, "breath_mode.py"])

    else:
        print(f"[{key}] 아직 설정되지 않은 모드입니다.")

# =========================
# (4) 메인 루프
# =========================
def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"[INFO] 아두이노 연결됨: {PORT}")
        print("[INFO] 키패드 입력을 기다리는 중...")

        while True:
            if ser.in_waiting:
                raw = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"[KEYPAD] 입력됨: {raw}")

                m = re.search(r'(\d)', raw)
                if m:
                    key = m.group(1)
                    launch_mode(key)
                    print("[INFO] 다시 입력 대기 중...\n")
                else:
                    print("[WARN] 숫자 키가 감지되지 않았습니다.")

    except serial.SerialException:
        print(f"[ERROR] 포트 {PORT} 를 열 수 없습니다.")
    except KeyboardInterrupt:
        print("\n[EXIT] 프로그램 종료됨.")
    finally:
        try:
            if 'ser' in locals():
                ser.close()
        except:
            pass
        # 안전망: 종료 시 모든 모드/오디오 정리
        stop_audio(clear_queue=True)
        stop_all_modes()

if __name__ == "__main__":
    main()


# In[ ]:




