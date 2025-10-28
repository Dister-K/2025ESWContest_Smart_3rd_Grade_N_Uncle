/* MICS-6814 2채널(RED=CO, NH3) ppm 추정기
   - NO2 미사용
   - 로그–로그(거듭제곱) 근사, 구간별 2점으로 자동 a,b 도출
   - CSV 출력
   - Baud Rate: 9600
*/

struct PowLawSeg {
  float r1, c1;  // (Rs/Ro, ppm) 첫 점
  float r2, c2;  // (Rs/Ro, ppm) 둘째 점
  float a, b;    // log10(C)=a*log10(r)+b
  float rmin, rmax;
};

//////////////// USER SETTINGS ////////////////
const int PIN_RED = A0;    // CO 채널
const int PIN_NH3 = A1;    // NH3 채널

const float VREF    = 5.0;     // 보드 기준전압(UNO=5.0, 3.3V 보드면 3.3)
const float ADC_FS  = 1023.0;  // UNO/Nano 10bit
const float RL      = 10000.0; // 실제 로드저항(Ω)

const float RO_RED  = 17255.99137; // << RED/CO 채널 R0(Ω)
const float RO_NH3  = 17937.47004; // << NH3 채널 R0(Ω)

// 데이터시트 그래프에서 ‘두 점’씩 읽어 넣기(예시 숫자! 반드시 교체)
PowLawSeg CO_SEGS[] = {
  //  r1,  c1,    r2,    c2,  a,b(자동),    rmin, rmax
  { 1.0,  60,    0.5, 300,  0,0,          0.4,  1.5 }
};
PowLawSeg NH3_SEGS[] = {
  { 2.0,  25,    1.0,  80,  0,0,          0.8,  3.0 }
};
const int N_CO  = sizeof(CO_SEGS)/sizeof(CO_SEGS[0]);
const int N_NH3 = sizeof(NH3_SEGS)/sizeof(NH3_SEGS[0]);

// ===== 디버그: 원시 ADC 출력 토글 =====
#define DEBUG_RAW 0            // 1로 바꾸면 RAW_ADC_* 출력
const unsigned long DEBUG_RAW_PERIOD_MS = 200; // 디버그 출력 주기(느리게)

/////////////// HELPERS //////////////////
float adcToVolt(int adc){ return adc*(VREF/ADC_FS); }
float rsFromVout(float vout){
  if(vout<=0.001) vout=0.001;
  return RL*(VREF/vout - 1.0);
}
void deriveAB(PowLawSeg &s){
  float x1=log10(s.r1), x2=log10(s.r2);
  float y1=log10(s.c1), y2=log10(s.c2);
  s.a=(y2-y1)/(x2-x1); s.b=y1 - s.a*x1;
}
float estimatePPM(float ratio, PowLawSeg segs[], int n){
  PowLawSeg* seg=nullptr;
  for(int i=0;i<n;i++) if(ratio>=segs[i].rmin && ratio<segs[i].rmax){ seg=&segs[i]; break; }
  if(!seg){ // 범위 밖이면 가장 가까운 구간 선택
    float best=1e9;
    for(int i=0;i<n;i++){
      float d = (ratio<segs[i].rmin)?(segs[i].rmin-ratio):
                (ratio>segs[i].rmax)?(ratio-segs[i].rmax):0.0;
      if(d<best){ best=d; seg=&segs[i]; }
    }
  }
  float clog = seg->a*log10(ratio) + seg->b;
  float ppm  = pow(10.0,clog);
  return (ppm<0)?0:ppm;
}

void setup(){
  Serial.begin(9600);
  for(int i=0;i<N_CO;i++)  deriveAB(CO_SEGS[i]);
  for(int i=0;i<N_NH3;i++) deriveAB(NH3_SEGS[i]);
  Serial.println("time_ms,Vred,Vnh3,Rs_red,Rs_nh3,ratio_red,ratio_nh3,CO_ppm,NH3_ppm");
}

void loop(){
  unsigned long t=millis();

  // 1) ADC 읽기
  int a_red = analogRead(PIN_RED);
  int a_nh3 = analogRead(PIN_NH3);

  // 1-1) (옵션) 원시 ADC 디버그 출력: analogRead() "직후"
  #if DEBUG_RAW
    static unsigned long last_dbg=0;
    if (t - last_dbg >= DEBUG_RAW_PERIOD_MS) {
      last_dbg = t;
      Serial.print("RAW_ADC_RED="); Serial.print(a_red);
      Serial.print(", RAW_ADC_NH3="); Serial.println(a_nh3);
    }
  #endif

  // 2) 전압/저항/비율/PPM 계산
  float v_red   = adcToVolt(a_red);
  float v_nh3   = adcToVolt(a_nh3);
  float rs_red  = rsFromVout(v_red);
  float rs_nh3  = rsFromVout(v_nh3);
  float ratio_red  = rs_red/RO_RED;
  float ratio_nh3  = rs_nh3/RO_NH3;
  float co_ppm  = estimatePPM(ratio_red, CO_SEGS, N_CO);
  float nh3_ppm = estimatePPM(ratio_nh3, NH3_SEGS, N_NH3);

  // 3) CSV 출력(파이썬 파서는 이 줄만 읽게 설계)
  Serial.print(t); Serial.print(",");
  Serial.print(v_red,3);  Serial.print(",");
  Serial.print(v_nh3,3);  Serial.print(",");
  Serial.print(rs_red,1); Serial.print(",");
  Serial.print(rs_nh3,1); Serial.print(",");
  Serial.print(ratio_red,3); Serial.print(",");
  Serial.print(ratio_nh3,3); Serial.print(",");
  Serial.print(co_ppm,1);  Serial.print(",");
  Serial.println(nh3_ppm,1);

  delay(500);
}