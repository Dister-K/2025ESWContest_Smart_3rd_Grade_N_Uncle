#include <Keypad.h>

const byte ROWS = 2;
const byte COLS = 4;

char keys[ROWS][COLS] = {
  {'1','2','3','4'},
  {'5','6','7','8'}
};

byte rowPins[ROWS] = {4, 5}; 
byte colPins[COLS] = {6, 7, 8, 9}; 

Keypad keypad = Keypad(makeKeymap(keys), rowPins, colPins, ROWS, COLS);

void setup(){
  Serial.begin(9600);
}

void loop(){
  char key = keypad.getKey();

  if (key) {
    Serial.print("Key Pressed = ");
    Serial.println(key);
  }
}