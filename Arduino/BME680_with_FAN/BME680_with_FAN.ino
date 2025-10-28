#include <Wire.h>
#include <SPI.h>
#include <Adafruit_Sensor.h>
#include "Adafruit_BME680.h"

#define SEALEVELPRESSURE_HPA (1013.25)

Adafruit_BME680 bme; // I2C 통신 사용
const int FAN_PIN = 3;


void setup() {
  pinMode(FAN_PIN, OUTPUT);   // 팬 제어 핀을 출력으로 설정
  digitalWrite(FAN_PIN, LOW); // 시작할 때 팬 OFF
  Serial.begin(9600);
  
  while (!Serial); // 시리얼 연결 대기

  // BME680 센서 초기화
  if (!bme.begin()) {
    Serial.println("Could not find a valid BME680 sensor, check wiring!");
    while (1);
  }

  // 센서 설정 (사용자 필요에 따라 조정)
  bme.setTemperatureOversampling(BME680_OS_8X);
  bme.setHumidityOversampling(BME680_OS_2X);
  bme.setPressureOversampling(BME680_OS_4X);
  bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
  bme.setGasHeater(320, 150); // 가열 온도 320°C, 가열 시간 150ms
}

void loop() {
  // 팬 ON
  digitalWrite(FAN_PIN, HIGH);
  // 센서 데이터 읽기
  if (!bme.performReading()) {
    Serial.println("Failed to perform reading :(");
    return;
  }

  // 데이터 출력
  Serial.print("Temperature: ");
  Serial.print(bme.temperature);
  Serial.println(" *C");

  Serial.print("Pressure: ");
  Serial.print(bme.pressure / 100.0);
  Serial.println(" hPa");

  Serial.print("Humidity: ");
  Serial.print(bme.humidity);
  Serial.println(" %");

  Serial.print("Gas Resistance: ");
  Serial.print(bme.gas_resistance / 1000.0);
  Serial.println(" KOhms");

  Serial.println();
  delay(2000); // 2초 간격으로 측정
}