// status_rgb_serial.ino (자동 유지시간 버전)
// 시리얼 명령으로 RGB LED 제어 + 극성/핀순서 교정 + 자동 OFF(기본 10초)
// 명령: G/Y/R/0, 정상/주의/위험/OFF, RTEST/GTEST/BTEST/TEST,
//       BRIGHT n(0~255), HOLD s(초; 0이면 자동꺼짐 비활성)

#define BAUDRATE 9600

// ======= 하드웨어 설정 =======
// 커먼 애노드(공통 +)면 true, 커먼 캐소드(공통 GND)면 false
const bool COMMON_ANODE = false;   // 필요시 true로 바꾸세요

// 핀 매핑
const int PIN_R = 9;   // 흰색,PWM
const int PIN_G = 10;  // 검정,PWM
const int PIN_B = 11;  // 군색,PWM

// ======= 채널별 밝기 보정 (G 채널이 너무 밝을 때 유용) =======
float R_GAIN = 1.0;
float G_GAIN = 0.8;   // 초록 LED 세기 약간 줄이기
float B_GAIN = 1.0;

// ======= 밝기(최대 세기) =======
int BRIGHT_MAX = 255;  // 필요시 낮춰서 눈부심 줄이기

// ======= 자동 유지시간(초) =======
unsigned long holdMs = 10000UL;    // 기본 10초 유지
bool autoHoldEnabled = true;       // HOLD 0 => false 로 꺼짐

// 내부 상태
enum Color { C_OFF, C_GREEN, C_YELLOW, C_RED, C_BLUE };
Color currentColor = C_OFF;
unsigned long lastOnMs = 0;        // 마지막으로 색을 켠 시각(ms)

// ======= 헬퍼들 =======
void pwmWritePolarity(int pin, int val /*0~255*/) {
  val = constrain(val, 0, 255);
  if (COMMON_ANODE) val = 255 - val; // 커먼 애노드면 반전
  analogWrite(pin, val);
}

void setRGBVal(int r, int g, int b) {
  // 채널별 보정값 적용
  r = constrain((int)(r * R_GAIN), 0, 255);
  g = constrain((int)(g * G_GAIN), 0, 255);
  b = constrain((int)(b * B_GAIN), 0, 255);

  pwmWritePolarity(PIN_R, r);
  pwmWritePolarity(PIN_G, g);
  pwmWritePolarity(PIN_B, b);
}

void setRGB(bool r, bool g, bool b) {
  setRGBVal(r ? BRIGHT_MAX : 0,
            g ? BRIGHT_MAX : 0,
            b ? BRIGHT_MAX : 0);
}

void showOff()    { setRGB(false, false, false); currentColor = C_OFF; }
void showGreen()  { setRGB(false, true,  false); currentColor = C_GREEN; }
void showYellow() { 
  setRGBVal(BRIGHT_MAX, BRIGHT_MAX * 0.6, 0);  // R 100%, G 60%
  currentColor = C_YELLOW; 
}
void showRed()    { setRGB(true,  false, false); currentColor = C_RED; }
void showBlue()   { setRGB(false, false, true ); currentColor = C_BLUE; }

void turnOnAndStartHold(Color c) {
  switch (c) {
    case C_GREEN:  showGreen();  break;
    case C_YELLOW: showYellow(); break;
    case C_RED:    showRed();    break;
    case C_BLUE:   showBlue();   break;
    default:       showOff();    break;
  }
  lastOnMs = millis(); // 유지 타이머 시작/리셋
}

// ======= 시리얼 유틸 =======
String readLine() {
  String s = "";
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') break;
    s += c;
    delay(1);
  }
  return s;
}

void applyCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  String up = cmd; up.toUpperCase();

  // 밝기 명령: BRIGHT 0~255
  if (up.startsWith("BRIGHT")) {
    int sp = cmd.indexOf(' ');
    if (sp > 0) {
      int val = cmd.substring(sp + 1).toInt();
      BRIGHT_MAX = constrain(val, 0, 255);
      // Serial.print("ACK BRIGHT "); Serial.println(BRIGHT_MAX);
      // 현재 색을 다시 적용해서 밝기 반영(옵션)
      switch (currentColor) {
        case C_GREEN:  showGreen();  break;
        case C_YELLOW: showYellow(); break;
        case C_RED:    showRed();    break;
        case C_BLUE:   showBlue();   break;
        default: break;
      }
    }
    return;
  }

  // 유지시간: HOLD s(초). 0 => 자동꺼짐 비활성
  if (up.startsWith("HOLD")) {
    int sp = cmd.indexOf(' ');
    if (sp > 0) {
      long sec = cmd.substring(sp + 1).toInt();
      if (sec <= 0) {
        autoHoldEnabled = false;   // 자동 꺼짐 비활성
        // Serial.println("ACK HOLD OFF");
      } else {
        autoHoldEnabled = true;
        holdMs = (unsigned long)sec * 1000UL;
        // Serial.print("ACK HOLD "); Serial.println(holdMs);
      }
    }
    return;
  }

  // 개별 채널 테스트
  if (up == "RTEST") { setRGBVal(BRIGHT_MAX, 0, 0); currentColor = C_RED;    lastOnMs = millis(); return; }
  if (up == "GTEST") { setRGBVal(0, BRIGHT_MAX, 0); currentColor = C_GREEN;  lastOnMs = millis(); return; }
  if (up == "BTEST") { setRGBVal(0, 0, BRIGHT_MAX); currentColor = C_BLUE;   lastOnMs = millis(); return; }
  if (up == "TEST") {
    setRGBVal(BRIGHT_MAX, 0, 0); delay(400);
    setRGBVal(0, BRIGHT_MAX, 0); delay(400);
    setRGBVal(0, 0, BRIGHT_MAX); delay(400);
    showOff();
    return;
  }

  // 일반 상태 명령 (수신 시점부터 holdMs 동안 유지)
  if (up == "G" || up == "GREEN" || cmd == "정상") {
    turnOnAndStartHold(C_GREEN);
  } else if (up == "Y" || up == "YELLOW" || cmd == "주의") {
    turnOnAndStartHold(C_YELLOW);
  } else if (up == "R" || up == "RED" || cmd == "위험") {
    turnOnAndStartHold(C_RED);
  } else if (up == "0" || up == "OFF" || cmd == "꺼짐" || up == "OFFLINE") {
    showOff(); // 즉시 소거
  }
  // 필요시 디버그
  // Serial.print("ACK: "); Serial.println(cmd);
}

void setup() {
  pinMode(PIN_R, OUTPUT);
  pinMode(PIN_G, OUTPUT);
  pinMode(PIN_B, OUTPUT);
  showOff();

  Serial.begin(BAUDRATE);
  delay(1200);
}

void loop() {
  // 시리얼 명령 처리
  if (Serial.available()) {
    String line = readLine();
    if (line.length() > 0) {
      applyCommand(line);
    }
  }

  // 자동 OFF 처리
  if (autoHoldEnabled && currentColor != C_OFF) {
    if (millis() - lastOnMs >= holdMs) {
      showOff();
    }
  }
}