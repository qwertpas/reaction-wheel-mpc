#include <Arduino.h>

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.printf("SETUP\n");
    Serial.flush();
}

void loop() {
    static unsigned long tick = 0;
    Serial.printf("TICK %lu: ms=%lu\n", tick++, millis());
    Serial.flush();
    delay(500);
}
